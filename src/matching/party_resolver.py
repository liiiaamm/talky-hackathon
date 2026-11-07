"""Payer / beneficiary identification from observable bank-line evidence.

Evidence, strongest first:
  exact_iban   camt counterparty IBAN equals a master IBAN (customers.iban / suppliers.iban)
  name         idf-weighted coverage of the party's distinctive name tokens in the bank text,
               allowing a truncated last token (bank texts cut at 16/35/38/65 chars)
  trade_name   same, on customers.trade_name
Group expansion (parent -> subsidiaries) is exposed separately: group membership alone
never decides an allocation, the amount search must confirm it.

No golden labels are used. Blocking: an inverted index token -> parties, per entity;
only parties sharing a reasonably rare token with the text are scored.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

import pandas as pd

from .normalize import ascii_upper, name_tokens, norm_text, compact


@dataclass
class PartyCandidate:
    party_id: str
    score: float
    match_method: str
    evidence: dict = field(default_factory=dict)


class PartyResolver:
    MAX_DF = 400          # tokens shared by more parties than this are not used for blocking
    TIE_COVERAGE = 0.85   # above this coverage, the name explaining more of the text ranks first

    def __init__(self, customers: pd.DataFrame, suppliers: pd.DataFrame):
        self.side = {}                     # party_id -> 'AR' | 'AP'
        self.entity = {}                   # party_id -> entity
        self.country = {}                  # party_id -> country
        self.parent = {}                   # child -> parent
        self.children = defaultdict(list)  # parent -> children
        self.iban = defaultdict(set)       # compact IBAN -> party ids
        self.tokens = {}                   # party_id -> list[(token, idf)]
        self.index = defaultdict(lambda: defaultdict(list))  # entity -> token -> party ids
        rows = []
        for df, side, name_cols in ((customers, "AR", ("name", "trade_name")), (suppliers, "AP", ("name",))):
            for r in df.itertuples(index=False):
                pid = r.customer_id if side == "AR" else r.supplier_id
                self.side[pid] = side
                self.entity[pid] = r.entity
                self.country[pid] = getattr(r, "country", "") or ""
                if r.iban:
                    self.iban[compact(r.iban)].add(pid)
                if side == "AR" and r.parent_customer_id and r.parent_customer_id != pid:
                    self.parent[pid] = r.parent_customer_id
                    self.children[r.parent_customer_id].append(pid)
                for col in name_cols:
                    nm = getattr(r, col)
                    if nm:
                        rows.append((pid, r.entity, col, name_tokens(nm)))
        df_count = defaultdict(lambda: defaultdict(int))
        for pid, ent, col, toks in rows:
            for t in set(toks):
                df_count[ent][t] += 1
        n_ent = defaultdict(int)
        for pid, ent in self.entity.items():
            n_ent[ent] += 1
        self.names = defaultdict(list)     # party_id -> list[(method, [(tok, idf)])]
        for pid, ent, col, toks in rows:
            n = n_ent[ent]
            wt = [(t, math.log(1 + n / df_count[ent][t])) for t in toks]
            self.names[pid].append(("name" if col == "name" else "trade_name", wt))
            for t in set(toks):
                if df_count[ent][t] <= self.MAX_DF and len(t) >= 3:
                    self.index[ent][t].append(pid)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _coverage(wt, text_tokens, text_prefixes):
        """(coverage share, matched idf weight). Truncated tokens (>=3 chars, a prefix of the name
        token) count 0.8 - bank texts cut names at fixed widths."""
        tot = sum(w for _, w in wt)
        if tot <= 0:
            return 0.0, 0.0
        hit = 0.0
        for t, w in wt:
            if t in text_tokens:
                hit += w
            elif len(t) >= 4 and any(t.startswith(x) for x in text_prefixes.get(t[:3], ())):
                hit += 0.8 * w
        return hit / tot, hit

    def resolve(self, text: str, name: str, iban: str, entities, side: str | None = None, top_k: int = 5):
        """Return up to top_k PartyCandidate sorted by score (desc). `entities` = allowed owner entities."""
        out = {}
        if iban:
            for pid in self.iban.get(compact(iban), ()):
                if self.entity.get(pid) in entities and (side is None or self.side[pid] == side):
                    out[pid] = PartyCandidate(pid, 1.0, "exact_iban", {"iban": iban})
        blob = norm_text(f"{name} {text}")
        toks = blob.split()
        tset = set(toks)
        pref = defaultdict(list)          # possible truncated tokens, keyed by first 3 chars
        for t in tset:
            if len(t) >= 3:
                pref[t[:3]].append(t)
        cands = set()
        for ent in entities:
            idx = self.index.get(ent)
            if not idx:
                continue
            for t in tset:
                for pid in idx.get(t, ()):
                    cands.add(pid)
        for pid in cands:
            if side is not None and self.side[pid] != side:
                continue
            best, bw, meth = 0.0, 0.0, "name"
            for m, wt in self.names[pid]:
                c, w = self._coverage(wt, tset, pref)
                if (c, w) > (best, bw):
                    best, bw, meth = c, w, m
            if best <= 0:
                continue
            prev = out.get(pid)
            if prev is not None and prev.match_method == "exact_iban":
                prev.evidence["coverage"] = round(best, 3)
                prev.evidence["weight"] = bw
                continue
            out[pid] = PartyCandidate(pid, 0.95 * best, meth, {"coverage": round(best, 3), "weight": bw})
        # ties on coverage are common (name A is a sub-name of name B): prefer the name that explains
        # more of the text (higher matched weight), then deterministic id order
        def rank(c):
            w = c.evidence.get("weight", 9e9)
            strong = c.match_method == "exact_iban" or c.score >= 0.95 * self.TIE_COVERAGE
            return (not strong, -w if strong else -c.score, -c.score, c.party_id)
        res = sorted(out.values(), key=rank)
        return res[:top_k]

    def group_members(self, pid: str) -> list[str]:
        """Party plus its subsidiaries (if it is a parent) - used only to widen the item pool."""
        return [pid] + self.children.get(pid, [])
