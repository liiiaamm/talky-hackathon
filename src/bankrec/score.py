"""Local scorer: same maths as data/starter/bankrec_score.py (official), vectorised, with per-segment and per-module breakdown."""
from __future__ import annotations

import polars as pl

from .data import EXCEPTIONS


def _clean(df: pl.DataFrame) -> pl.DataFrame:
    return df.select(
        "bank_txn_id",
        pl.col("item_id").fill_null(""),
        pl.col("amount").cast(pl.Float64, strict=False),
        pl.col("category").fill_null("").str.to_uppercase(),
        pl.col("exception").fill_null("").str.to_uppercase(),
        *([pl.col("module")] if "module" in df.columns else []),
    )


def score(sub: pl.DataFrame, golden: pl.DataFrame, verbose: bool = True) -> dict:
    """sub: submission rows (+ optional 'module' column). golden: golden rows of the same bank lines (with segment)."""
    g = _clean(golden).with_columns(golden["segment"].alias("segment"))
    s = _clean(sub)
    gi = g.filter(pl.col("item_id") != "")
    si = s.filter(pl.col("item_id") != "")
    # ---- allocation pairs
    pairs = si.join(gi.select("bank_txn_id", "item_id", pl.col("amount").alias("gamt"), pl.col("exception").alias("gexc")),
                    on=["bank_txn_id", "item_id"], how="left")
    tol = pl.max_horizontal(pl.lit(0.01), 0.005 * pl.col("gamt").abs()) + 1e-9
    pairs = pairs.with_columns(
        (pl.col("gamt").is_not_null() & ((pl.col("amount") - pl.col("gamt")).abs() <= tol)).alias("tp"))
    pairs = pairs.with_columns((pl.col("tp") & (pl.col("exception") == pl.col("gexc"))).alias("ok_row"))
    tp = int(pairs["tp"].sum())
    n_sub, n_gold = si.height, gi.height
    prec = tp / n_sub if n_sub else 0.0
    rec = tp / n_gold if n_gold else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    # ---- transaction accuracy
    gstat = g.group_by("bank_txn_id").agg(
        (pl.col("item_id") != "").sum().alias("g_n"), pl.col("category").filter(pl.col("item_id") == "").first().alias("g_cat"),
        pl.col("exception").filter(pl.col("item_id") == "").first().alias("g_exc"), pl.col("segment").first())
    sstat_items = pairs.group_by("bank_txn_id").agg(pl.len().alias("s_n"), pl.col("ok_row").sum().alias("s_ok"))
    sstat_cat = s.filter(pl.col("item_id") == "").group_by("bank_txn_id").agg(
        pl.col("category").first().alias("s_cat"), pl.col("exception").first().alias("s_exc"))
    t = gstat.join(sstat_items, on="bank_txn_id", how="left").join(sstat_cat, on="bank_txn_id", how="left").with_columns(
        pl.col("s_n").fill_null(0), pl.col("s_ok").fill_null(0))
    t = t.with_columns(pl.when(pl.col("g_n") > 0)
                       .then(pl.col("s_cat").is_null() & (pl.col("s_n") == pl.col("g_n")) & (pl.col("s_ok") == pl.col("g_n")))
                       .otherwise(pl.col("s_cat").is_not_null() & (pl.col("s_n") == 0) & (pl.col("s_cat") == pl.col("g_cat"))
                                  & (pl.col("s_exc") == pl.col("g_exc").fill_null("")))
                       .alias("ok"))
    acc = float(t["ok"].mean())
    # ---- exception macro F1 (row level; no-item rows keyed by bank line)
    exc_rows_sub = pl.concat([
        pairs.select("bank_txn_id", "item_id", "exception", pl.col("gexc").fill_null("__none__").alias("gexc")),
        s.filter(pl.col("item_id") == "").join(gstat.select("bank_txn_id", "g_n", "g_exc"), on="bank_txn_id", how="left").select(
            "bank_txn_id", "item_id", "exception",
            pl.when(pl.col("g_n") == 0).then(pl.col("g_exc").fill_null("")).otherwise(pl.lit("__none__")).alias("gexc")),
    ])
    gold_exc = pl.concat([gi.select("exception"), g.filter(pl.col("item_id") == "").select("exception")])
    f1s, per_exc = [], {}
    for c in EXCEPTIONS:
        gc = int((gold_exc["exception"] == c).sum())
        if not gc:
            continue
        sub_c = exc_rows_sub.filter(pl.col("exception") == c)
        tpc = int((sub_c["gexc"] == c).sum())
        fpc = sub_c.height - tpc
        p_ = tpc / (tpc + fpc) if tpc + fpc else 0.0
        r_ = tpc / gc
        f = 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0.0
        f1s.append(f)
        per_exc[c] = (round(p_, 3), round(r_, 3), round(f, 3), gc)
    macro = sum(f1s) / len(f1s) if f1s else 1.0
    total = round(100 * (0.5 * f1 + 0.3 * acc + 0.2 * macro), 2)
    seg = (t.group_by("segment").agg(pl.len().alias("n"), pl.col("ok").mean().alias("acc"))
           .with_columns((pl.col("n") * (1 - pl.col("acc")) / t.height * 100).alias("lost_acc_pts_x0.3"))
           .sort("lost_acc_pts_x0.3", descending=True))
    res = dict(score=total, f1=round(f1, 4), precision=round(prec, 4), recall=round(rec, 4), txn_acc=round(acc, 4),
               exc_macro=round(macro, 4), per_exception=per_exc, segments=seg, txn_table=t)
    if verbose:
        print(f"SCORE {total}  | F1 {f1:.4f} (P {prec:.4f} R {rec:.4f})  TxnAcc {acc:.4f}  ExcMacroF1 {macro:.4f}"
              f"  | lines {t.height:,} sub_pairs {n_sub:,} gold_pairs {n_gold:,}")
        print("exceptions (P, R, F1, n_gold):", {k: v for k, v in per_exc.items()})
        with pl.Config(tbl_rows=40, tbl_width_chars=160, fmt_str_lengths=40):
            print(seg.head(30))
        if "module" in s.columns:
            m = (s.group_by("bank_txn_id").agg(pl.col("module").first()).join(t.select("bank_txn_id", "ok", "segment"), on="bank_txn_id")
                 .group_by("module").agg(pl.len().alias("lines"), pl.col("ok").mean().alias("acc")).sort("lines", descending=True))
            with pl.Config(tbl_rows=40):
                print(m)
    return res
