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
                 terms: pd.DataFrame | None = None):
        self.s = store
        # payment_terms code -> (discount basis points, discount days); POLICIES §3 SHORT_DISCOUNT
        self.disc = {}
        if terms is not None:
            for r in terms.itertuples(index=False):
                pct = str(r.early_payment_discount_pct or "").strip()
                if pct:
                    self.disc[r.code] = (int(round(float(pct) * 100)), int(r.discount_days or 0))
        self.r = resolver
        self.scope = entity_scope        # line entity -> set of item entities it may settle
        self.cfg = {"use_iban": True, "use_names": True, "use_refs": True, "use_digit_refs": True,
                    "use_subset": True, "use_group": True, "use_month": True, "partial_hint": True,
                    "fx": True, "max_pool": 60, "node_budget": 20000, "min_party_score": 0.55,
                    "top_parties": 8, "use_discount": True, "use_withholding": True, "grace_days": 3,
                    "withholding_bp": [125, 150, 1000], "withholding_countries": ["MX", "BR", "MA"], "mode_node_budget": 500, "ambiguity_abstain": False}
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
