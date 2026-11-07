"""Person B matcher: reference + party + amount -> MatchProposal for one bank line.

The matcher never mutates balances. It reads them through `ledger.open(i)` (cents, item
index) and returns a proposal; the resolver (Person A) commits it.

Decision order (each step only fires on an exact cent match unless stated):
  1. referenced items (strong keys) of a compatible entity/side sum to the cash -> 'ref_exact'
  2. per candidate party (refs' parties first, then resolver ranking): subset-sum over the
     party's open items in the cash currency, referenced items forced in -> 'party_subset'
     (single-item hits on the same amount are ordered FIFO, or by service month if the text
     names one, POLICIES §5.6)
  3. parent with subsidiaries: same search on the widened pool -> 'group_subset'
  4. single strongly-referenced item with cash < open -> PARTIAL hint (amount = cash)
  else abstain (Person A / C decide: category, remittance, exceptions).
"""
from __future__ import annotations

import bisect
import re

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .normalize import doc_keys, text_reference_tokens, service_month
from .party_resolver import PartyResolver
from .subset_sum import find_subsets

FULL_CLEAR_HINTS = ("FX_DIFFERENCE",)


@dataclass
class ItemCandidate:
    item_id: str
    party_id: str
    entity: str
    open_amount_before: float
    item_currency: str
    score: float = 0.0
    reference_score: float = 0.0
    party_score: float = 0.0
    amount_score: float = 0.0
    date_score: float = 0.0
    evidence: dict = field(default_factory=dict)


# charge option written in the text of a foreign transfer ("GASTOS SHA", "CHARGES: BEN", "SHA CHARGES")
_CHARGE_OPT = re.compile(r"\b(?:GASTOS|CHARGES?|CHGS?)\s*:?\s*(?:SHA|BEN)\b|\b(?:SHA|BEN)\s+CHARGES\b")


def _round_units(cents: int) -> bool:
    """1, 2 or 5 x 10^n whole currency units (1, 2, 5, 10, 20, 50, 100 ...)."""
    if cents <= 0 or cents % 100:
        return False
    u = cents // 100
    while u % 10 == 0:
        u //= 10
    return u in (1, 2, 5)


@dataclass
class MatchProposal:
    bank_txn_id: str
    payer_party_id: str | None
    match_method: str
    confidence: float
    candidates: list
    allocations: list            # [{item_id, amount (item ccy, positive), exception, source, confidence}]
    abstained: bool
    evidence: dict = field(default_factory=dict)


class ItemStore:
    """Static item attributes as numpy arrays + lookup indexes. Balances live in the ledger."""

    def __init__(self, items: pd.DataFrame):
        it = items.reset_index(drop=True)
        self.item_id = it.item_id.to_numpy()
        self.kind = it.kind.to_numpy()
        self.entity = it.entity.to_numpy()
        self.party = it.party_id.to_numpy()
        self.doc = it.document_number.to_numpy()
        self.issue = it.issue_date.to_numpy()
        self.currency = it.currency.to_numpy()
        self.total_c = (pd.to_numeric(it.total_amount, errors="coerce").fillna(0).to_numpy() * 100).round().astype(np.int64)
        self.terms = it.payment_terms.fillna("").to_numpy() if "payment_terms" in it else np.full(len(it), "", object)
        self.issue_ord = pd.to_datetime(it.issue_date, errors="coerce").map(lambda d: d.toordinal() if pd.notna(d) else 0).to_numpy()
        self.sign = np.where(np.isin(self.kind, ["ARC", "APC"]), -1, 1)
        self.side = np.where(np.isin(self.kind, ["ARI", "ARC"]), "AR", "AP")
        self.pos = {iid: i for i, iid in enumerate(self.item_id)}
        self.strong = defaultdict(list)       # (entity, key) -> [idx]
        self.digits = defaultdict(list)
        self.tails = defaultdict(list)
        for i, (e, d) in enumerate(zip(self.entity, self.doc)):
            k = doc_keys(d)
            self.strong[(e, k.strong)].append(i)
            if k.digits:
                self.digits[(e, k.digits)].append(i)
            for t in k.tails:
                self.tails[(e, t)].append(i)
            # customers abbreviate a run of invoices to their last six digits ("FA250112467 112647 112745")
            suf = k.digits[-6:].lstrip("0") if len(k.digits) > 6 else ""
            if len(suf) >= 5 and suf not in k.tails:
                self.tails[(e, suf)].append(i)
        if "po" in it:
            self.po = defaultdict(list)
            for i, (e, p) in enumerate(zip(self.entity, it.po.to_numpy())):
                if p and len(p) >= 5:
                    self.po[(e, doc_keys(p).strong)].append(i)
        by_party = defaultdict(list)
        order = np.lexsort((self.item_id, self.issue))
        for i in order:
            by_party[self.party[i]].append(int(i))
        self.by_party = dict(by_party)

    def lookup_document(self, entity: str, document_number: str) -> list[str]:
        """(entity, document_number) -> item ids (for Person C remittances / PDF clauses)."""
        k = doc_keys(document_number)
        idx = self.strong.get((entity, k.strong), []) or self.digits.get((entity, k.digits), [])
        return [self.item_id[i] for i in idx]


class CustomerMatcher:
    def __init__(self, store: ItemStore, resolver: PartyResolver, entity_scope: dict, cfg: dict | None = None,
                 terms: pd.DataFrame | None = None, fx: pd.DataFrame | None = None):
        self.s = store
        # payment_terms code -> (discount basis points, discount days); POLICIES §3 SHORT_DISCOUNT
        self.disc = {}
        if terms is not None:
            for r in terms.itertuples(index=False):
                pct = str(r.early_payment_discount_pct or "").strip()
                if pct:
                    self.disc[r.code] = (int(round(float(pct) * 100)), int(r.discount_days or 0))
        self.r = resolver
        self.units_per_eur = {}          # currency -> sorted [(date, units per EUR)] for the 60 EUR bank-charge cap
        if fx is not None:
            for r in fx.sort_values("date").itertuples(index=False):
                self.units_per_eur.setdefault(r.currency, []).append((str(r.date), float(r.units_per_eur)))
        self.scope = entity_scope        # line entity -> set of item entities it may settle
        self.closed = defaultdict(list)  # party -> [(cash cents, item idx, booking ordinal)] of lines that fully closed one item
        self.cfg = {"use_iban": True, "use_names": True, "use_refs": True, "use_digit_refs": True,
                    "use_subset": True, "use_group": True, "use_month": True, "partial_hint": True,
                    "fx": True, "max_pool": 60, "node_budget": 20000, "min_party_score": 0.55,
                    "top_parties": 8, "use_discount": True, "use_withholding": True, "grace_days": 3,
                    "withholding_bp": [125, 150, 1000], "withholding_countries": ["MX", "BR", "MA"], "mode_node_budget": 500, "ambiguity_abstain": False,
                    "use_bank_charges": True, "charges_cap_eur": 60.0, "charges_cap_pct": 2.0,
                    "use_duplicates": True, "duplicate_days": 31,
                    "use_overpayment": True, "over_max_pct": 10.0}
        if cfg:
            self.cfg.update(cfg)

    # ------------------------------------------------------------------ references
    def reference_hits(self, blob, entities, side):
        s = self.s
        strong, digit = text_reference_tokens(blob)
        hits = {}
        for e in entities:
            for k in strong:
                for i in s.strong.get((e, k), ()):
                    hits[i] = max(hits.get(i, 0), 1.0)
            if self.cfg["use_digit_refs"]:
                for k in digit:
                    for i in s.digits.get((e, k), ()):
                        hits.setdefault(i, 0.7)
                    for i in s.tails.get((e, k), ()):
                        hits.setdefault(i, 0.5)
        return {i: v for i, v in hits.items() if s.side[i] == side}

    # ------------------------------------------------------------------ value modes
    def _modes(self, booking_ord):
        """Ways the cash can relate to the open amounts. Each mode maps item idx -> (value_cents, exception).
        'full' pays open amounts; 'disc' = early-payment discount within terms (+grace), cash = total - pct*total;
        'whtNNN' = tax withheld at source, cash = total - rate*total (POLICIES §3). Discount/withholding
        only apply to untouched invoices (open == total)."""
        s, cfg = self.s, self.cfg
        modes = [("full", None)]
        if cfg["use_discount"] and self.disc:
            def disc(i, o):
                d = self.disc.get(s.terms[i])
                if not d or s.sign[i] < 0 or o != s.total_c[i] or booking_ord > s.issue_ord[i] + d[1] + cfg["grace_days"]:
                    return None
                return s.total_c[i] - (s.total_c[i] * d[0] + 5000) // 10000
            modes.append(("disc", disc))
        if cfg["use_withholding"]:
            for bp in cfg["withholding_bp"]:
                wc = set(cfg["withholding_countries"] or ())
                def wht(i, o, bp=bp, wc=wc):
                    if s.sign[i] < 0 or o != s.total_c[i] or (wc and self.r.country.get(s.party[i], "") not in wc):
                        return None
                    return s.total_c[i] - (s.total_c[i] * bp + 5000) // 10000
                modes.append((f"wht{bp}", wht))
        return modes

    @staticmethod
    def _mode_exc(mode):
        return "SHORT_DISCOUNT" if mode == "disc" else ("SHORT_WITHHOLDING" if mode.startswith("wht") else "")

    def _values(self, pool, fn, ledger):
        """(values, exceptions) for a pool under a mode; None for items the mode does not change."""
        s = self.s
        vals, excs = [], []
        for i in pool:
            o = ledger.open(i)
            v = fn(i, o) if fn else None
            if v is None:
                vals.append(int(s.sign[i] * o)); excs.append("")
            else:
                vals.append(int(v)); excs.append("*")
        return vals, excs

    # ------------------------------------------------------------------ bank charges (POLICIES §3)
    XBORDER_CODES = ("PMNT/RCDT/XBCT",)

    @staticmethod
    def _cross_border(line) -> bool:
        """Cross-border transfer evidence: camt XBCT, Norma 43 'transferencias en divisa' (code 13), or SHA/BEN charge
        options in the text. A shortfall on any other line is not treated as bank charges."""
        code = (getattr(line, "bank_code", "") or "").strip()
        if code in CustomerMatcher.XBORDER_CODES or (getattr(line, "statement_format", "") == "n43" and code == "13"):
            return True
        return bool(_CHARGE_OPT.search(" ".join((line.text, line.remittance_info))))

    def _cap_cents(self, cur, day):
        """60 EUR equivalent in `cur` cents at the last reference rate on or before `day`."""
        cap = self.cfg["charges_cap_eur"]
        if cur != "EUR":
            rates = self.units_per_eur.get(cur)
            if not rates:
                return 0
            k = bisect.bisect_right(rates, (day, float("inf"))) - 1
            cap *= rates[max(k, 0)][1]
        return int(cap * 100)

    def _bank_charges(self, target, cur, side, entities, parties, strong_refs, ledger, day):
        """Single open invoice of the payer, fully cleared, with 0 < open - cash <= min(60 EUR eq., 2 % of open).
        Unique over the candidate payers (or over the strongly referenced items when there are some)."""
        s, cfg = self.s, self.cfg
        cap = self._cap_cents(cur, day)
        pool = strong_refs or [i for p in parties[: cfg["top_parties"]] for i in s.by_party.get(p, ())]
        hits = set()
        for i in pool:
            o = ledger.open(i)
            if s.side[i] != side or s.currency[i] != cur or s.entity[i] not in entities or s.sign[i] < 0 or o <= 0:
                continue
            short = o - target
            if 0 < short <= min(cap, o * cfg["charges_cap_pct"] / 100.0):
                hits.add(i)
        return next(iter(hits)) if len(hits) == 1 else None

    # ------------------------------------------------------------------ overpayments (POLICIES §3)
    def _overpayment(self, target, cur, side, entities, parties, strong_refs, ledger):
        """Single open invoice of the payer (or the single strongly referenced one) paid with a little more than its open
        amount: 0 < cash - open <= over_max_pct % of open, and the excess is a whole number of currency units or exactly
        that percentage (a round top-up: 1, 2 or 5 times a power of ten in currency units). Unique, otherwise abstain. Amount = open amount (the excess stays on account)."""
        s, cfg = self.s, self.cfg
        pool = strong_refs or [i for p in parties[: cfg["top_parties"]] for i in s.by_party.get(p, ())]
        hits = set()
        for i in pool:
            o = ledger.open(i)
            if s.side[i] != side or s.currency[i] != cur or s.entity[i] not in entities or s.sign[i] < 0 or o <= 0:
                continue
            ex = target - o
            if 0 < ex <= o * cfg["over_max_pct"] / 100.0 and (_round_units(ex) or ex == (o * int(cfg["over_max_pct"]) + 50) // 100):
                hits.add(i)
        return next(iter(hits)) if len(hits) == 1 else None

    # ------------------------------------------------------------------ duplicate payments (POLICIES §3)
    def record(self, proposal, cash_cents, booking_ord):
        """Called by the caller after it accepted a proposal: remember single-item full settlements by the exact cash."""
        al = proposal.allocations
        if len(al) == 1 and al[0]["exception"] == "" and int(round(al[0]["amount"] * 100)) == cash_cents:
            i = self.s.pos[al[0]["item_id"]]
            self.closed[self.s.party[i]].append((cash_cents, i, booking_ord))

    def _duplicate(self, target, parties, ledger, bord):
        """The same payer paid exactly this cash for one item it had already settled in full within `duplicate_days`,
        and none of its open items has that amount: unique such item -> DUPLICATE_PAYMENT."""
        hits = set()
        for p in parties[: self.cfg["top_parties"]]:
            for c, i, d in self.closed.get(p, ()):
                if c == target and 0 < bord - d <= self.cfg["duplicate_days"] and ledger.open(i) == 0:
                    hits.add(i)
        return next(iter(hits)) if len(hits) == 1 else None

    def _open_with_amount(self, target, cur, side, entities, parties, ledger):
        s = self.s
        for p in parties[: self.cfg["top_parties"]]:
            for i in s.by_party.get(p, ()):
                if s.side[i] == side and s.currency[i] == cur and s.entity[i] in entities and s.sign[i] * ledger.open(i) == target:
                    return True
        return False

    # ------------------------------------------------------------------ main
    def match(self, line, ledger) -> MatchProposal:
        s, cfg = self.s, self.cfg
        tid = line.bank_txn_id
        amt = int(round(float(line.amount) * 100))
        side = "AR" if amt > 0 else "AP"
        target = abs(amt)
        cur = line.currency
        fx = False
        if cfg["fx"] and line.orig_currency and line.orig_currency != line.currency and line.orig_amount:
            target = int(round(abs(float(line.orig_amount)) * 100))
            cur = line.orig_currency
            fx = True
        entities = self.scope.get(line.entity, {line.entity})
        blob = " ".join((line.text, line.ref1, line.ref2, line.remittance_info, line.end_to_end_id))
        ev = {"side": side, "target": target, "currency": cur}
        bord = pd.Timestamp(line.booking_date).toordinal() if line.booking_date else 0
        modes = self._modes(bord) if side == "AR" else [("full", None)]

        refs = self.reference_hits(blob, entities, side) if cfg["use_refs"] else {}
        refs = {i: v for i, v in refs.items() if ledger.open(i) != 0 and s.currency[i] == cur}
        strong_refs = [i for i, v in refs.items() if v >= 1.0]

        def alloc(idxs, method, conf, mode="full", changed=()):
            out = []
            mexc = self._mode_exc(mode)
            for i in idxs:
                o = ledger.open(i)
                e = mexc if i in changed else ""
                if fx and s.sign[i] > 0 and e in ("", "SHORT_DISCOUNT", "SHORT_WITHHOLDING"):
                    e = "FX_DIFFERENCE"
                out.append({"item_id": s.item_id[i], "amount": abs(o) / 100.0, "exception": e,
                            "source": method, "confidence": conf})
            return out

        # 1. referenced items alone (each value mode)
        if strong_refs:
            for mode, fn in modes:
                vals, excs = self._values(strong_refs, fn, ledger)
                if mode != "full" and "*" not in excs:
                    continue
                if sum(vals) == target:
                    changed = {i for i, e in zip(strong_refs, excs) if e == "*"}
                    conf = 0.97 if mode == "full" else 0.93
                    return MatchProposal(tid, s.party[strong_refs[0]], "ref_exact" + ("" if mode == "full" else "_" + mode),
                                         conf, [], alloc(strong_refs, "ref_exact", conf, mode, changed), False, ev)

        # 2. parties
        parties = []
        for i in sorted(refs, key=lambda i: -refs[i]):
            if s.party[i] not in parties:
                parties.append(s.party[i])
        if cfg["use_iban"] or cfg["use_names"]:
            iban = line.counterparty_account if cfg["use_iban"] else ""
            text = blob if cfg["use_names"] else ""
            name = line.counterparty_name if cfg["use_names"] else ""
            pc = self.r.resolve(text, name, iban, entities, side, top_k=cfg["top_parties"])
            ev["parties"] = [(c.party_id, round(c.score, 3), c.match_method) for c in pc]
            for c in pc:
                if c.score >= cfg["min_party_score"] and c.party_id not in parties:
                    parties.append(c.party_id)
        month = service_month(blob) if cfg["use_month"] else None

        # 2b. a repeat of an earlier full payment of one invoice, with nothing else open at that amount
        if cfg["use_duplicates"] and side == "AR" and not fx and parties and not strong_refs \
                and not self._open_with_amount(target, cur, side, entities, parties, ledger):
            i = self._duplicate(target, parties, ledger, bord)
            if i is not None:
                a = {"item_id": s.item_id[i], "amount": target / 100.0, "exception": "DUPLICATE_PAYMENT",
                     "source": "duplicate", "confidence": 0.8}
                return MatchProposal(tid, s.party[i], "duplicate", 0.8, [], [a], False, ev)

        if cfg["use_subset"]:
            for mode, fn in modes:
                for pid in parties[: cfg["top_parties"] + 1]:
                    res = self._search([pid], target, cur, side, entities, refs, month, ledger, fn, mode)
                    if res:
                        idxs, conf, ambiguous, changed = res
                        if ambiguous and cfg["ambiguity_abstain"]:
                            continue
                        meth = "party_subset" + ("" if mode == "full" else "_" + mode)
                        return MatchProposal(tid, pid, meth, conf, [], alloc(idxs, meth, conf, mode, changed), False, ev)
            if cfg["use_group"]:
                for pid in parties[: cfg["top_parties"]]:
                    root = self.r.parent.get(pid, pid)
                    members = self.r.group_members(root)
                    if len(members) > 1:
                        res = self._search(members, target, cur, side, entities, refs, month, ledger, None, "full")
                        if res:
                            idxs, conf, ambiguous, changed = res
                            return MatchProposal(tid, root, "group_subset", conf * 0.9, [], alloc(idxs, "group_subset", conf * 0.9), False, ev)

        # 3b. cross-border transfer short by an intermediary bank fee -> SHORT_BANK_CHARGES, item fully cleared
        if cfg["use_bank_charges"] and side == "AR" and not fx and self._cross_border(line):
            i = self._bank_charges(target, cur, side, entities, parties, strong_refs, ledger, str(line.booking_date)[:10])
            if i is not None:
                a = {"item_id": s.item_id[i], "amount": abs(ledger.open(i)) / 100.0, "exception": "SHORT_BANK_CHARGES",
                     "source": "bank_charges", "confidence": 0.85}
                return MatchProposal(tid, s.party[i], "bank_charges", 0.85, [], [a], False, ev)

        # 3c. a round top-up over one invoice -> OVERPAYMENT, item fully cleared at its open amount
        if cfg["use_overpayment"] and side == "AR" and not fx:
            i = self._overpayment(target, cur, side, entities, parties, strong_refs, ledger)
            if i is not None:
                a = {"item_id": s.item_id[i], "amount": abs(ledger.open(i)) / 100.0, "exception": "OVERPAYMENT",
                     "source": "overpayment", "confidence": 0.8}
                return MatchProposal(tid, s.party[i], "overpayment", 0.8, [], [a], False, ev)

        # 4. partial hint on a single strongly referenced item
        if cfg["partial_hint"] and len(strong_refs) == 1:
            i = strong_refs[0]
            o = ledger.open(i)
            if s.sign[i] > 0 and 0 < target < o:
                a = {"item_id": s.item_id[i], "amount": target / 100.0, "exception": "PARTIAL",
                     "source": "partial_hint", "confidence": 0.5}
                return MatchProposal(tid, s.party[i], "partial_hint", 0.5, [], [a], False, ev)
        return MatchProposal(tid, parties[0] if parties else None, "abstain", 0.0, [], [], True, ev)

    def _search(self, members, target, cur, side, entities, refs, month, ledger, fn, mode):
        """Exact search on the open items of `members`. Returns (idxs, confidence, ambiguous, changed) or None.
        Non-'full' modes only accept solutions where at least one item uses the mode's value."""
        s, cfg = self.s, self.cfg
        pool = []
        for m in members:
            for i in s.by_party.get(m, ()):
                if s.side[i] == side and s.currency[i] == cur and s.entity[i] in entities and ledger.open(i) != 0:
                    pool.append(i)
        if not pool:
            return None
        forced = [i for i in pool if refs.get(i, 0) >= 1.0]
        # abbreviated citations ("FA250112467 112647 112745"): when two or more of this payer's own open items are hit by
        # digit / tail references, they anchor the search and only the residual is searched
        weak = [i for i in pool if 0 < refs.get(i, 0) < 1.0]
        if cfg.get("weak_anchors", True) and len(forced) + len(weak) >= 2 and weak:
            forced = forced + weak
        # preference order: referenced first, then service-month match, then oldest (FIFO)
        if month:
            y, m = month
            ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
            mkey = f"{ny:04d}-{nm:02d}"
        def key(i):
            mm = 0 if (month and s.issue[i][:7] == mkey) else 1
            return (-refs.get(i, 0), mm, s.issue[i], s.item_id[i])
        pool.sort(key=key)
        vals, excs = self._values(pool, fn, ledger)
        if mode != "full" and "*" not in excs:
            return None
        def ok(idx_list):
            return mode == "full" or any(excs[j] == "*" for j in idx_list)
        def res(js, conf, amb):
            return [pool[j] for j in js], conf, amb, {pool[j] for j in js if excs[j] == "*"}
        # fast path over the FULL pool: single item of exactly the amount (first in preference order)
        hits = [j for j, v in enumerate(vals) if v == target and ok([j]) and (not forced or pool[j] in forced)]
        if hits:
            return res([hits[0]], 0.9 if len(hits) == 1 else 0.75, len(hits) > 1)
        # oldest-first prefix (customer clears its oldest open items), over the full pool
        if not forced and cfg.get("use_prefix", True):
            acc = 0
            for j, v in enumerate(vals):
                acc += v
                if acc == target and j > 0 and ok(range(j + 1)):
                    return res(list(range(j + 1)), 0.85, False)
                if acc > target and all(x > 0 for x in vals[: j + 1]):
                    break
        sel = list(range(len(pool)))
        if len(pool) > cfg["max_pool"]:
            fset = set(forced)
            sel = [j for j in sel if j < cfg["max_pool"] or pool[j] in fset]
        svals = [vals[j] for j in sel]
        req = [k for k, j in enumerate(sel) if pool[j] in forced]
        budget = cfg["node_budget"] if mode == "full" else cfg["mode_node_budget"]
        sols, exhausted = find_subsets(svals, target, node_budget=budget, required=req)
        sols = [[sel[k] for k in sol] for sol in sols if ok([sel[k] for k in sol])]
        if not sols:
            return None
        ambiguous = len(sols) > 1
        return res(sols[0], 0.85 if not ambiguous else 0.6, ambiguous)
