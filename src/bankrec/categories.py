"""Non-item lines (Persona A): text classifier over bank lines -> one of the 21 categories or ITEM (= settles items).

    python -m bankrec.categories --train            # trains on TRAIN up to 2024-12-31, saves models/categories.joblib
    python -m bankrec.categories --train --all      # trains on all TRAIN (use this for LIVE/FINAL)
    python -m bankrec.categories --train --until 2024-06-30  # clean validation on both later windows

Resolvers:
  strong_categories: lines whose predicted category (not ITEM / UNIDENTIFIED) has prob >= STRONG -> category row (runs early)
  all_categories:    remaining lines: predicted category if not ITEM; else credits -> UNIDENTIFIED, debits -> best non-item class
"""
from __future__ import annotations

import argparse
import os
import time

import joblib
import numpy as np
import polars as pl

from .data import ROOT, Ctx

MODEL = os.environ.get("BANKREC_CAT_MODEL", os.path.join(ROOT, "models", "categories.joblib"))
STRONG = 0.97
_cache = {}


def features(bank: pl.DataFrame) -> list[str]:
    """One string per line: sign, format, bank codes and the normalised text (digits collapsed)."""
    df = bank.select(
        pl.when(pl.col("amt") > 0).then(pl.lit("SGN_CR")).otherwise(pl.lit("SGN_DB")).alias("a"),
        pl.concat_str([pl.lit("FMT_"), pl.col("statement_format").fill_null("")]).alias("b"),
        pl.concat_str([pl.lit("BC_"), pl.col("bank_code").fill_null("").str.replace_all(r"[^A-Za-z0-9]", "_")]).alias("c"),
        pl.concat_str([pl.lit("BS_"), pl.col("bank_subcode").fill_null("")]).alias("d"),
        pl.concat_str([pl.lit("ENT_"), pl.col("entity").fill_null("")]).alias("e"),
        pl.col("alltext").str.replace_all(r"\d", "0").str.replace_all(r"0{2,}", "00").alias("f"),
        # amount magnitude bucket (payroll / loans are big, fees small)
        pl.concat_str([pl.lit("MAG_"), pl.col("abs_amt").add(1).log10().floor().cast(pl.Int32).cast(pl.Utf8)]).alias("g"),
    )
    return [" ".join(r) for r in df.iter_rows()]


def _labels(ctx: Ctx) -> pl.DataFrame:
    g = ctx.golden.group_by("bank_txn_id").agg(
        pl.col("item_id").is_not_null().any().alias("has_item"), pl.col("category").drop_nulls().first().alias("cat"))
    return g.select("bank_txn_id", pl.when(pl.col("has_item")).then(pl.lit("ITEM")).otherwise(pl.col("cat")).alias("label"))


def train(all_years: bool = False, n_item: int = 400_000, until: str | None = None):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    from .data import load
    t0 = time.time()
    if all_years and until:
        raise ValueError("--all and --until are mutually exclusive")
    ctx = load("train", None, until or (None if all_years else "2024-12-31"))
    lab = _labels(ctx)
    df = ctx.bank.join(lab, on="bank_txn_id")
    item = df.filter(pl.col("label") == "ITEM").sample(min(n_item, df.filter(pl.col("label") == "ITEM").height), seed=0)
    df = pl.concat([df.filter(pl.col("label") != "ITEM"), item])
    X, y = features(df), df["label"].to_list()
    print(f"training on {len(y):,} lines ({time.time() - t0:.0f}s)", flush=True)
    clf = make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3, max_features=300_000, sublinear_tf=True, dtype=np.float32),
        LogisticRegression(max_iter=300, C=4.0, n_jobs=-1))
    clf.fit(X, y)
    os.makedirs(os.path.dirname(MODEL), exist_ok=True)
    joblib.dump(clf, MODEL)
    import json
    from pathlib import Path
    Path(MODEL + ".metadata.json").write_text(json.dumps({
        "training_until": until or (None if all_years else "2024-12-31"),
        "all_years": all_years, "training_lines": len(y), "item_sample_seed": 0,
    }, indent=2) + "\n")
    print(f"saved {MODEL} ({time.time() - t0:.0f}s)")


def _predict(ctx: Ctx, pending: pl.DataFrame) -> pl.DataFrame:
    if "clf" not in _cache:
        _cache["clf"] = joblib.load(MODEL)
    clf = _cache["clf"]
    P = clf.predict_proba(features(pending))
    classes = np.array(clf.classes_)
    best = P.argmax(1)
    nonitem = [i for i, c in enumerate(classes) if c not in ("ITEM", "UNIDENTIFIED")]
    best_ni = np.array(nonitem)[P[:, nonitem].argmax(1)]
    return pending.select("bank_txn_id", "is_credit").with_columns(
        pl.Series("pred", classes[best]), pl.Series("p", P.max(1)), pl.Series("pred_ni", classes[best_ni]),
        pl.Series("p_ni", P[np.arange(len(P)), best_ni]),
        pl.Series("p_unid", P[:, list(classes).index("UNIDENTIFIED")] if "UNIDENTIFIED" in classes else np.zeros(len(P))))


def _rows(df: pl.DataFrame, cat: pl.Expr, module: str, exc: pl.Expr | None = None) -> pl.DataFrame:
    exc = pl.when(cat == "UNIDENTIFIED").then(pl.lit("UNIDENTIFIED")).otherwise(pl.lit("")) if exc is None else exc
    return df.select("bank_txn_id", pl.lit(None, pl.Utf8).alias("item_id"), pl.lit(None, pl.Float64).alias("amount"), cat.alias("category"),
                     exc.alias("exception"), pl.lit(module).alias("module"))


def strong_categories(ctx: Ctx, pending: pl.DataFrame) -> pl.DataFrame:
    if not os.path.exists(MODEL) or pending.height == 0:
        return pending.head(0).select("bank_txn_id")
    pr = _predict(ctx, pending)
    _cache["last_pred"] = pr
    sel = pr.filter(~pl.col("pred").is_in(["ITEM", "UNIDENTIFIED"]) & (pl.col("p") >= STRONG))
    return _rows(sel, pl.col("pred"), "cat_strong")


UNID_P = float(os.environ.get("BANKREC_UNID_P", "0.10"))   # UNIDENTIFIED exception threshold on cat_rest credits


def all_categories(ctx: Ctx, pending: pl.DataFrame) -> pl.DataFrame:
    if not os.path.exists(MODEL) or pending.height == 0:
        return pending.head(0).select("bank_txn_id")
    pr = _predict(ctx, pending)
    cat = (pl.when(~pl.col("pred").is_in(["ITEM"])).then(pl.col("pred"))
           .when(pl.col("is_credit")).then(pl.lit("UNIDENTIFIED"))
           .otherwise(pl.col("pred_ni")))
    # UNIDENTIFIED exception only when the model believes it (otherwise the line is a missed item anyway: an UNIDENTIFIED
    # exception there is a false positive in the exception macro-F1)
    # ... or when it gives UNIDENTIFIED a fair chance: P >= UNID_P recovers ~640 true UNIDENTIFIED per half for ~170 false
    # flags on lines that are wrong anyway (missed items). Replay: H1 93.58 -> 93.80, out-of-sample H2 92.92 -> 93.14.
    exc = pl.when((cat == "UNIDENTIFIED") & ((pl.col("pred") == "UNIDENTIFIED") | (pl.col("p_unid") >= UNID_P)
                                             | (os.environ.get("BANKREC_UNID_ALL") == "1")))\
        .then(pl.lit("UNIDENTIFIED")).otherwise(pl.lit(""))
    return _rows(pr, cat, "cat_rest", exc)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    span = ap.add_mutually_exclusive_group()
    span.add_argument("--all", action="store_true")
    from datetime import date
    span.add_argument("--until", type=lambda value: date.fromisoformat(value).isoformat(),
                      help="last training booking date (inclusive), e.g. 2024-06-30 for clean H2 validation")
    a = ap.parse_args()
    if a.train:
        train(a.all, until=a.until)
