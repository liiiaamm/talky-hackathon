"""Unreferenced partial payments (Persona C): a learned ranker over the customer's open invoices.

A receipt from an evidenced customer that is smaller than every open invoice and cites none of them (POLICIES §3 "A CUENTA")
has to be booked as PARTIAL against ONE invoice. No written rule picks the invoice (measured: oldest 23 %, newest 40 %,
largest 33 %), so the pick is learned from the labelled history: per-candidate features (amount ratios, ages, ranks,
citations) plus the customer's own past behaviour (how often it pays oldest / newest / largest, its usual fraction).
Nothing customer-specific is hard-coded; the model is retrained from history at run time and a confidence threshold keeps
precision first. The remainder of a partially paid invoice is then recognised chronologically (cash == what is left).

    python -m bankrec.c_partial --train --to 2024-12-31        # -> models/c_partial_2024-12-31.joblib
Resolver `partial_payments` (place after ar_multi); model path from BANKREC_PARTIAL_MODEL or the --to date = window start.
"""
from __future__ import annotations

import argparse
import os
import re
from collections import defaultdict

import numpy as np
import polars as pl

from .data import PQ, ROOT, Ctx, norm_expr, _pq
from .deterministic import _acc_entity, empty_rows

# ----------------------------------------------------------------------------------------------------------- CONFIG
AMT_TOL = 0.011
P_MIN = float(os.environ.get("BANKREC_PARTIAL_PMIN", "0.0"))     # 0 = use the threshold stored with the model
TARGET_PRECISION = 0.90       # threshold is chosen on held-out months to reach this precision on claimed lines
MIN_RATIO, MAX_RATIO = 0.02, 0.98   # a "partial" pays between 2 % and 98 % of the invoice
FEATURES = ["ratio", "rem_ratio", "round5", "round_amt", "days_issue", "days_due", "overdue", "rank_old", "rank_amt", "n_cand",
            "rel_amt", "cited6", "h_n", "h_oldest", "h_newest", "h_largest", "h_ratio_mean", "h_round5", "ratio_dev"]


def _hist_stats(part: pl.DataFrame) -> pl.DataFrame:
    """Per-customer cumulative stats of past first partials, as of each line (strictly before it)."""
    p = part.sort(["party_id", "booking_date", "bank_txn_id"])
    cum = p.with_columns(
        pl.col("is_oldest").cast(pl.Float64).cum_sum().over("party_id").alias("c_old"),
        pl.col("is_newest").cast(pl.Float64).cum_sum().over("party_id").alias("c_new"),
        pl.col("is_largest").cast(pl.Float64).cum_sum().over("party_id").alias("c_lar"),
        pl.col("g_round5").cast(pl.Float64).cum_sum().over("party_id").alias("c_r5"),
        pl.col("g_ratio").cum_sum().over("party_id").alias("c_ratio"),
        pl.int_range(1, pl.len() + 1).over("party_id").cast(pl.Float64).alias("c_n"))
    # shift by one so the line's own label is not in its history
    sh = lambda c: pl.col(c).shift(1).over("party_id").fill_null(0.0)  # noqa: E731
    return cum.select("bank_txn_id", sh("c_n").alias("h_n"), (sh("c_old") / pl.max_horizontal(sh("c_n"), 1)).alias("h_oldest"),
                      (sh("c_new") / pl.max_horizontal(sh("c_n"), 1)).alias("h_newest"), (sh("c_lar") / pl.max_horizontal(sh("c_n"), 1)).alias("h_largest"),
                      (sh("c_ratio") / pl.max_horizontal(sh("c_n"), 1)).alias("h_ratio_mean"), (sh("c_r5") / pl.max_horizontal(sh("c_n"), 1)).alias("h_round5"))


def _final_stats(part: pl.DataFrame, cutoff: str) -> pl.DataFrame:
    p = part.filter(pl.col("booking_date") < cutoff)
    return p.group_by("party_id").agg(pl.len().cast(pl.Float64).alias("h_n"), pl.col("is_oldest").mean().alias("h_oldest"),
                                      pl.col("is_newest").mean().alias("h_newest"), pl.col("is_largest").mean().alias("h_largest"),
                                      pl.col("g_ratio").mean().alias("h_ratio_mean"), pl.col("g_round5").mean().alias("h_round5"))


def _cand_features(c: pl.DataFrame) -> pl.DataFrame:
    """c: one row per (line, candidate invoice) with cash, total, issue_date, due_date, booking_date, tnorm, docnorm."""
    c = c.with_columns((pl.col("cash") / pl.col("total")).alias("ratio"))
    c = c.with_columns(
        (1 - pl.col("ratio")).alias("rem_ratio"),
        ((pl.col("ratio") * 20 - (pl.col("ratio") * 20).round()).abs() < 0.002).cast(pl.Float64).alias("round5"),
        ((pl.col("cash") - (pl.col("cash") / 100).round() * 100).abs() < 0.011).cast(pl.Float64).alias("round_amt"),
        (pl.col("booking_date").str.to_date() - pl.col("issue_date").str.to_date()).dt.total_days().cast(pl.Float64).alias("days_issue"),
        (pl.col("booking_date").str.to_date() - pl.col("due_date").str.to_date()).dt.total_days().cast(pl.Float64).fill_null(0.0).alias("days_due"),
    )
    c = c.with_columns((pl.col("days_due") > 0).cast(pl.Float64).alias("overdue"),
                       pl.col("issue_date").rank("ordinal").over("bank_txn_id").cast(pl.Float64).alias("rank_old"),
                       pl.col("total").rank("ordinal", descending=True).over("bank_txn_id").cast(pl.Float64).alias("rank_amt"),
                       pl.len().over("bank_txn_id").cast(pl.Float64).alias("n_cand"),
                       (pl.col("total") / pl.col("total").median().over("bank_txn_id")).alias("rel_amt"),
                       pl.col("tnorm").str.contains(pl.col("docnorm").str.replace_all(r"\D", "").str.slice(-6), literal=True)
                       .fill_null(False).cast(pl.Float64).alias("cited6"))
    return c.with_columns((pl.col("rank_old") - 1).alias("rank_old"), (pl.col("rank_amt") - 1).alias("rank_amt"))


# ------------------------------------------------------------------------------------------------------- training
def build_training(cutoff: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    g = _pq("train", "golden_allocations")
    bank = _pq("train", "bank_transactions").select("bank_txn_id", "booking_date", "currency", pl.col("amount").cast(pl.Float64).alias("cash"), "text")
    ar = _pq("train", "ar_invoices").select("item_id", pl.col("customer_id").alias("party_id"), "entity", "issue_date", "due_date", "currency",
                                           pl.col("total_amount").cast(pl.Float64).alias("total"), norm_expr("document_number").alias("docnorm"))
    gg = g.join(bank, on="bank_txn_id")
    # first partials: single-row PARTIAL lines on an invoice, not explained by PDF clauses (the segment label is only used to
    # build the training set; it is never a feature)
    part = gg.filter((pl.col("segment") == "ar_partial") & (pl.col("exception") == "PARTIAL")).select(
        "bank_txn_id", pl.col("item_id").alias("gold"), "booking_date", "currency", "cash", norm_expr("text").alias("tnorm"))
    part = part.join(ar.select(pl.col("item_id").alias("gold"), "party_id"), on="gold")
    cleared = gg.filter(~pl.col("exception").fill_null("").is_in(["PARTIAL", "SHORT_DISPUTE", "RETURNED", "DUPLICATE_PAYMENT"])).group_by("item_id").agg(
        pl.col("booking_date").min().alias("cleared_on"))
    cand = (part.join(ar, on="party_id", suffix="_i").filter((pl.col("issue_date") <= pl.col("booking_date")) & (pl.col("currency_i") == pl.col("currency"))
                                                           & (pl.col("total") > pl.col("cash") + AMT_TOL))
            .join(cleared, on="item_id", how="left").filter(pl.col("cleared_on").is_null() | (pl.col("cleared_on") > pl.col("booking_date"))))
    cand = _cand_features(cand).with_columns((pl.col("item_id") == pl.col("gold")).cast(pl.Int8).alias("y"))
    # gold rank flags per line -> customer history
    gl = cand.filter(pl.col("y") == 1).select("bank_txn_id", "party_id", "booking_date", (pl.col("rank_old") == 0).alias("is_oldest"),
                                               (pl.col("rank_old") == pl.col("n_cand") - 1).alias("is_newest"), (pl.col("rank_amt") == 0).alias("is_largest"),
                                               pl.col("ratio").alias("g_ratio"), pl.col("round5").alias("g_round5"))
    hist = _hist_stats(gl)
    cand = cand.join(hist, on="bank_txn_id", how="left")
    # only lines whose gold is among the candidates
    ok = cand.group_by("bank_txn_id").agg(pl.col("y").max().alias("has")).filter(pl.col("has") == 1).select("bank_txn_id")
    cand = cand.join(ok, on="bank_txn_id", how="semi")
    # NEGATIVE lines: receipts of the same evidenced customers that are NOT partials (1:N, noisy references, credit-note
    # nettings, discounts...). At run time the resolver sees every unclaimed receipt, so the model must also learn to say
    # "this line is not a partial at all": every candidate of these lines is labelled 0.
    neg_segs = ["ar_ref_noise", "ar_1to1", "ar_1toN_text", "ar_n_to_m", "ar_1toN_credit_note", "ar_same_amount_fifo", "ar_same_amount_month",
                "ar_short_discount", "ar_discount_late", "ar_short_withholding", "ar_dispute_settlement", "ar_third_party_parent", "ar_new_iban"]
    nl = (gg.filter(pl.col("segment").is_in(neg_segs) & pl.col("item_id").str.starts_with("ARI"))
          .group_by("bank_txn_id").agg(pl.first("booking_date"), pl.first("currency"), pl.first("cash"), pl.first("text"), pl.first("item_id"))
          .join(ar.select("item_id", "party_id"), on="item_id").select("bank_txn_id", "booking_date", "currency", "cash", norm_expr("text").alias("tnorm"), "party_id"))
    nl = nl.sample(n=min(nl.height, 3 * part.height), seed=7)
    ncand = (nl.join(ar, on="party_id", suffix="_i").filter((pl.col("issue_date") <= pl.col("booking_date")) & (pl.col("currency_i") == pl.col("currency"))
                                                           & (pl.col("total") > pl.col("cash") + AMT_TOL))
             .join(cleared, on="item_id", how="left").filter(pl.col("cleared_on").is_null() | (pl.col("cleared_on") > pl.col("booking_date"))))
    ncand = ncand.filter((pl.col("cash") / pl.col("total")).is_between(MIN_RATIO, MAX_RATIO))
    ncand = _cand_features(ncand).with_columns(pl.lit(0, pl.Int8).alias("y"))
    # history as of the line for negatives: as-of join on the cumulative partial stats (strictly earlier lines)
    cum = (gl.sort(["party_id", "booking_date"]).with_columns(
        pl.int_range(1, pl.len() + 1).over("party_id").cast(pl.Float64).alias("h_n"),
        pl.col("is_oldest").cast(pl.Float64).cum_sum().over("party_id").alias("so"), pl.col("is_newest").cast(pl.Float64).cum_sum().over("party_id").alias("sn"),
        pl.col("is_largest").cast(pl.Float64).cum_sum().over("party_id").alias("sl"), pl.col("g_ratio").cum_sum().over("party_id").alias("sr"),
        pl.col("g_round5").cast(pl.Float64).cum_sum().over("party_id").alias("s5"))
        .select("party_id", pl.col("booking_date").str.to_date().alias("d"), "h_n", (pl.col("so") / pl.col("h_n")).alias("h_oldest"),
                (pl.col("sn") / pl.col("h_n")).alias("h_newest"), (pl.col("sl") / pl.col("h_n")).alias("h_largest"),
                (pl.col("sr") / pl.col("h_n")).alias("h_ratio_mean"), (pl.col("s5") / pl.col("h_n")).alias("h_round5")).sort("d"))
    ncand = ncand.with_columns((pl.col("booking_date").str.to_date() - pl.duration(days=1)).alias("d")).sort("d")
    ncand = ncand.join_asof(cum, on="d", by="party_id", strategy="backward", check_sortedness=False).drop("d")
    cols = [c for c in cand.columns if c in ncand.columns]
    cand = pl.concat([cand.select(cols), ncand.select(cols)], how="vertical_relaxed")
    cand = cand.with_columns([pl.col(c).fill_null(0.0) for c in ["h_n", "h_oldest", "h_newest", "h_largest", "h_ratio_mean", "h_round5"]])
    cand = cand.with_columns((pl.col("ratio") - pl.col("h_ratio_mean")).abs().alias("ratio_dev")).filter(pl.col("booking_date") < cutoff)
    return cand, _final_stats(gl, cutoff)


def train(cutoff: str, out: str | None = None):
    import joblib
    import lightgbm as lgb
    cand, stats = build_training(cutoff)
    dates = cand.select("bank_txn_id", "booking_date").unique().sort("booking_date")
    split = dates["booking_date"][int(dates.height * 0.8)]
    tr, va = cand.filter(pl.col("booking_date") < split), cand.filter(pl.col("booking_date") >= split)
    X, y = tr.select(FEATURES).to_numpy(), tr["y"].to_numpy()
    clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=20, subsample=0.8, colsample_bytree=0.8, verbose=-1)
    clf.fit(X, y)
    # per-line argmax on validation -> choose P_MIN for the target precision
    va = va.with_columns(pl.Series("p", clf.predict_proba(va.select(FEATURES).to_numpy())[:, 1]))
    top = va.sort("p", descending=True).group_by("bank_txn_id").agg(pl.col("p").first().alias("p1"), pl.col("y").first().alias("hit"), pl.col("y").max().alias("is_partial"))
    n_pos = int(top["is_partial"].sum())
    best = (0.5, 0.0, 0.0)
    for thr in np.arange(0.3, 0.99, 0.02):
        s = top.filter(pl.col("p1") >= thr)
        if s.height < 30:
            break
        prec, rec = s["hit"].mean(), s["hit"].sum() / max(n_pos, 1)
        if prec >= TARGET_PRECISION:
            best = (float(thr), float(prec), float(rec)); break
        best = (float(thr), float(prec), float(rec))
    print(f"cutoff {cutoff}: train lines {tr['bank_txn_id'].n_unique():,} (rows {tr.height:,}), val lines {top.height:,} of which partials {n_pos:,} | "
          f"argmax precision {top['hit'].mean():.3f} | threshold {best[0]:.2f} -> precision {best[1]:.3f} recall {best[2]:.3f}")
    imp = sorted(zip(FEATURES, clf.feature_importances_), key=lambda t: -t[1])[:8]
    print("  top features:", imp)
    out = out or os.path.join(ROOT, "models", f"c_partial_{cutoff}.joblib")
    joblib.dump(dict(model=clf, threshold=best[0], stats=stats.to_pandas(), cutoff=cutoff, features=FEATURES), out)
    print("  saved", out)


# ------------------------------------------------------------------------------------------------------- resolver
def _load_model(ctx: Ctx):
    import joblib
    path = os.environ.get("BANKREC_PARTIAL_MODEL")
    if not path:
        cut = ctx.date_from or "9999-99-99"
        cands = sorted(f for f in os.listdir(os.path.join(ROOT, "models")) if f.startswith("c_partial_") and f.endswith(".joblib"))
        cands = [f for f in cands if f[len("c_partial_"):-len(".joblib")] <= cut]
        if not cands:
            return None
        path = os.path.join(ROOT, "models", cands[-1])
    return joblib.load(path) if os.path.exists(path) else None


PARTY_MIN_SCORE = 0.6     # PartyResolver score of the best customer candidate (1.0 = IBAN; names scored by text coverage)


def _parties(ctx: Ctx, lines: pl.DataFrame, items: pl.DataFrame) -> list[tuple[str, str]]:
    """(bank_txn_id, customer_id) for lines whose payer is identified: exact name keys / IBAN first (ar_multi's automata);
    where they find nobody, Person B's PartyResolver (fuzzy / truncated names, trade names) as a fallback when importable."""
    from .ar_multi import _automata
    autos = _automata(ctx, items)
    iban2p = defaultdict(set)
    for ent, p, ib in ctx.customers.select("entity", "customer_id", "iban").iter_rows():
        if ib:
            iban2p[(ent, ib)].add(p)
    out, missing = [], []
    for txn, ent, t, ib, raw, nm in lines.select("bank_txn_id", "entity", "tnorm", "iban", "raw", "cp_name").iter_rows():
        best, bl = None, 0
        if ib and len(iban2p.get((ent, ib), ())) == 1:
            best, bl = next(iter(iban2p[(ent, ib)])), 99
        A = autos.get(ent)
        if A is not None and bl < 99:
            for _end, (L, ps) in A.iter(t):
                if L > bl and len(ps) == 1:
                    best, bl = next(iter(ps)), L
                elif L == bl and best not in ps:
                    best = None
        if best:
            out.append((txn, best))
        elif bl == 0:
            missing.append((txn, ent, raw, nm, ib))
    if missing:
        try:
            import sys
            src = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            if src not in sys.path:
                sys.path.insert(0, src)
            from matching.party_resolver import PartyResolver
            pr = PartyResolver(ctx.customers.to_pandas().fillna(""), ctx.suppliers.to_pandas().fillna(""))
            for txn, ent, raw, nm, ib in missing:
                cands = [c for c in pr.resolve(raw, nm, ib or "", {ent}, top_k=3) if str(c.party_id).startswith("C")]
                if cands and cands[0].score >= PARTY_MIN_SCORE and (len(cands) == 1 or cands[0].score > cands[1].score):
                    out.append((txn, cands[0].party_id))
        except Exception:
            pass
    return out


def partial_payments(ctx: Ctx, pending: pl.DataFrame) -> pl.DataFrame:
    m = _load_model(ctx)
    if m is None:
        return empty_rows()
    items = ctx.items.filter((pl.col("item_type") == "ARI") & (pl.col("open_amount") > 0)).select(
        "item_id", "entity", "party_id", "party_name", "currency", "docnorm", "issue_date", "due_date", "total", "open_amount")
    lines = (pending.filter(pl.col("is_credit") & pl.col("orig_amt").is_null()).join(_acc_entity(ctx), on="account_id")
             .select("bank_txn_id", "booking_date", "currency", pl.col("acc_entity").alias("entity"), pl.col("abs_amt").round(2).alias("cash"),
                     norm_expr("alltext").alias("tnorm"), pl.col("counterparty_account").alias("iban"),
                     pl.col("alltext").fill_null("").alias("raw"), pl.col("counterparty_name").fill_null("").alias("cp_name")).sort("booking_date"))
    if not lines.height:
        return empty_rows()
    party = _parties(ctx, lines, items)
    if not party:
        return empty_rows()
    lp = lines.join(pl.DataFrame(party, schema={"bank_txn_id": pl.Utf8, "party_id": pl.Utf8}, orient="row"), on="bank_txn_id")
    c = lp.join(items, on=["entity", "party_id", "currency"]).filter(pl.col("issue_date") <= pl.col("booking_date"))
    # an open invoice equal to the cash is the matchers' business; a partial pays 2-98 % of an invoice bigger than the cash
    eq = c.filter((pl.col("open_amount") - pl.col("cash")).abs() <= AMT_TOL).select("bank_txn_id").unique()
    c = c.join(eq, on="bank_txn_id", how="anti").filter((pl.col("total") > pl.col("cash") + AMT_TOL)
                                                       & (pl.col("cash") / pl.col("total")).is_between(MIN_RATIO, MAX_RATIO))
    if not c.height:
        return empty_rows()
    c = _cand_features(c)
    stats = pl.from_pandas(m["stats"]) if len(m["stats"]) else None
    if stats is not None:
        c = c.join(stats, on="party_id", how="left")
    for k in ["h_n", "h_oldest", "h_newest", "h_largest", "h_ratio_mean", "h_round5"]:
        c = c.with_columns(pl.col(k).fill_null(0.0) if k in c.columns else pl.lit(0.0).alias(k))
    c = c.with_columns((pl.col("ratio") - pl.col("h_ratio_mean")).abs().alias("ratio_dev"))
    c = c.with_columns(pl.Series("p", m["model"].predict_proba(c.select(m["features"]).to_numpy())[:, 1]))
    thr = P_MIN or m["threshold"]
    top = c.sort(["bank_txn_id", "p"], descending=[False, True]).unique("bank_txn_id", keep="first", maintain_order=True).filter(pl.col("p") >= thr)
    # chronological local ledger: a later receipt equal to what is left of a partially paid invoice is its remainder (no exception)
    out, left = [], {}
    for txn, iid, cash, open_amt in top.sort("booking_date").select("bank_txn_id", "item_id", "cash", "open_amount").iter_rows():
        if iid in left:
            if abs(left[iid] - cash) <= AMT_TOL:
                out.append((txn, iid, cash, "")); left.pop(iid)
            continue                                   # second partial on the same invoice: abstain
        out.append((txn, iid, cash, "PARTIAL")); left[iid] = round(open_amt - cash, 2)
    # remainders paid by lines the ranker did not score (cash == what is left): pick them up from the same party's lines
    if left:
        rem = lp.join(pl.DataFrame([(i, v) for i, v in left.items()], schema={"item_id": pl.Utf8, "left": pl.Float64}, orient="row")
                      .join(items.select("item_id", "party_id", "entity", "currency"), on="item_id"), on=["party_id", "entity", "currency"])
        rem = rem.filter((pl.col("cash") - pl.col("left")).abs() <= AMT_TOL).unique("bank_txn_id").unique("item_id")
        done = {t for t, *_ in out}
        for txn, iid, cash in rem.select("bank_txn_id", "item_id", "cash").iter_rows():
            if txn not in done:
                out.append((txn, iid, cash, ""))
    df = pl.DataFrame(out, schema={"bank_txn_id": pl.Utf8, "item_id": pl.Utf8, "amount": pl.Float64, "exception": pl.Utf8}, orient="row")
    return df.select("bank_txn_id", "item_id", "amount", pl.lit(None, pl.Utf8).alias("category"), "exception", pl.lit("c_partial").alias("module"))


RESOLVERS = [partial_payments]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--to", default="2024-12-31", help="use history strictly before this date (window start)")
    a = ap.parse_args()
    if a.train:
        train(a.to)
