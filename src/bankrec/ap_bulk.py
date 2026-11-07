"""Aggregated AP payment runs whose bank debit cannot equal the run total (mixed-currency runs), and C34 statements that
split the run id across text fields ("TRANSF. EMITIDAS FICHERO C34 PR0200250 | 11085 | (357 ORD.)" = run PR020025011085).

`deterministic.payment_runs` links an aggregated debit to its run only when the run total equals the bank amount. A run
that pays documents in several currencies is debited once in the account currency, so that check never holds. Here a
pending debit is linked to the whole run when all the structural evidence agrees:
  - the run id is in the text (after re-joining the C34 split) and the run belongs to the debited account;
  - the run mixes currencies (otherwise payment_runs would already have matched it on amount);
  - the bank text states how many transfers it carries ("N TRANSFERS" / "(N ORD.)") and the run has exactly that many
    end-to-end ids;
  - the part of the run in the account currency does not exceed the debit.
Rows and exceptions (FX_DIFFERENCE / PARTIAL / credit notes) come from the same helpers as payment_runs.
"""
from __future__ import annotations

import polars as pl

from .data import Ctx
from .deterministic import _acc_entity, _f, _pr_cumulative, _pr_exception, _pr_rows_to_items, empty_rows, mk_rows, norm_expr

C34_SPLIT = r"((?:PR|CF)\d{7})\s*\|\s*(\d{5})\b"
COUNT = r"(\d+)\s+TRANSFERS|\((\d+)\s+ORD"


def bulk_mixed_runs(ctx: Ctx, pending: pl.DataFrame) -> pl.DataFrame:
    if ctx.payruns is None or pending.height == 0:
        return empty_rows()
    deb = pending.filter(~pl.col("is_credit")).with_columns(
        pl.col("alltext").str.replace_all(C34_SPLIT, "${1}${2}").alias("_t")).with_columns(
        pl.col("_t").str.extract(r"((?:PR|CF)\d{12})").alias("run_id"),
        pl.coalesce(pl.col("_t").str.extract(COUNT, 1), pl.col("_t").str.extract(COUNT, 2)).cast(pl.Int64, strict=False).alias("n_txt"),
        pl.col("account_id").alias("acc")).filter(pl.col("run_id").is_not_null() & pl.col("n_txt").is_not_null())
    if deb.height == 0:
        return empty_rows()
    pr = ctx.payruns.join(_acc_entity(ctx), left_on="debtor_account_id", right_on="account_id").with_columns(
        _f("transfer_amount").alias("tamt"), _f("document_amount").alias("damt"), pl.col("debtor_account_id").alias("acc"))
    acc_ccy = ctx.bank.select("account_id", pl.col("currency").alias("acc_ccy")).unique("account_id")
    runs = pr.unique(["acc", "run_id", "end_to_end_id"]).join(acc_ccy, left_on="acc", right_on="account_id", how="left").group_by(
        "acc", "run_id").agg(
        pl.len().alias("n_e2e"), pl.col("currency").n_unique().alias("n_ccy"),
        pl.col("tamt").filter(pl.col("currency") == pl.col("acc_ccy")).sum().alias("same_ccy_tot"))
    link = deb.join(runs, on=["acc", "run_id"]).filter(
        (pl.col("n_ccy") > 1) & (pl.col("n_e2e") == pl.col("n_txt")) & (pl.col("same_ccy_tot") <= pl.col("abs_amt") + 0.01))
    link = link.unique("bank_txn_id").unique(["acc", "run_id"], keep="none")      # one debit per run, else abstain
    if link.height == 0:
        return empty_rows()
    rows = pr.join(link.select("bank_txn_id", "acc", "run_id", pl.col("currency").alias("bank_ccy"),
                               pl.lit(False).alias("dup")), on=["acc", "run_id"])
    rows = _pr_rows_to_items(ctx, rows).rename({"currency": "item_ccy"}).unique(["bank_txn_id", "item_id"])
    involved = pr.join(rows.select(pl.col("dnorm").alias("_d")).unique(), left_on=norm_expr("document_number"), right_on="_d", how="semi")
    rows = rows.join(_pr_cumulative(ctx, involved), on=["acc", "end_to_end_id", "item_id"], how="left")
    return mk_rows(rows.with_columns(pl.col("doc_amt").alias("amount")), "ap_bulk_mixed", _pr_exception())


RESOLVERS = [bulk_mixed_runs]
