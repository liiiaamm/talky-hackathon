"""Deterministic resolvers (Persona A): structured files say exactly which items a bank line clears.

Every resolver: f(ctx, pending_bank_df) -> rows DataFrame [bank_txn_id, item_id, amount, category, exception, module]
"""
from __future__ import annotations

import polars as pl

from .data import Ctx, norm_expr

ROWS_SCHEMA = {"bank_txn_id": pl.Utf8, "item_id": pl.Utf8, "amount": pl.Float64, "category": pl.Utf8, "exception": pl.Utf8, "module": pl.Utf8}


def empty_rows() -> pl.DataFrame:
    return pl.DataFrame(schema=ROWS_SCHEMA)


def mk_rows(df: pl.DataFrame, module: str, exception: pl.Expr | str = "", amount: str = "amount") -> pl.DataFrame:
    exc = exception if isinstance(exception, pl.Expr) else pl.lit(exception)
    return df.select(
        pl.col("bank_txn_id"), pl.col("item_id"), pl.col(amount).cast(pl.Float64).round(2).alias("amount"),
        pl.lit(None, pl.Utf8).alias("category"), exc.cast(pl.Utf8).alias("exception"), pl.lit(module).alias("module"))


def _f(c: str) -> pl.Expr:
    return pl.col(c).cast(pl.Float64, strict=False)


def _acc_entity(ctx: Ctx) -> pl.DataFrame:
    """account_id -> entity. Not every account is in the master (e.g. IT01-VALD-EUR-2): the entity is the id prefix."""
    ids = [ctx.bank.select("account_id"), ctx.accounts.select("account_id")]
    for df, col in ((ctx.payruns, "debtor_account_id"), (ctx.dd, "account_id"), (ctx.cards, "account_id"), (ctx.cheques, "account_id")):
        if df is not None:
            ids.append(df.select(pl.col(col).alias("account_id")))
    if "acc_entity" not in ctx.__dict__.get("_cache", {}):
        ctx.__dict__.setdefault("_cache", {})["acc_entity"] = pl.concat(ids).unique().with_columns(
            pl.col("account_id").str.slice(0, 4).alias("acc_entity"))
    return ctx.__dict__["_cache"]["acc_entity"]


def _tokens(bank: pl.DataFrame, pattern: str, name: str = "tok") -> pl.DataFrame:
    return (bank.select("bank_txn_id", "account_id", "amt", "abs_amt", "booking_date", "currency", "orig_amt", "orig_currency",
                        pl.col("alltext").str.extract_all(pattern).alias(name))
            .explode(name).filter(pl.col(name).is_not_null()).unique(["bank_txn_id", name]))


# --------------------------------------------------------------------------------------------- direct debits (AR)
def _dd_rows_to_items(ctx: Ctx, ddrows: pl.DataFrame) -> pl.DataFrame:
    """ddrows: rows of direct_debit_batches (+ entity). Adds item_id: exact doc number, else 'RECIBO <digits>' via mandate/customer + amount."""
    ari = ctx.items.filter(pl.col("item_type") == "ARI").select("item_id", "entity", "docnorm", "party_id", "total", "issue_date")
    ddrows = ddrows.with_row_index("_r").with_columns(norm_expr("remittance_info").alias("rnorm"), _f("amount").alias("dd_amt"))
    exact = ddrows.join(ari, left_on=["entity", "rnorm"], right_on=["entity", "docnorm"], how="inner").unique("_r")
    rest = ddrows.join(exact.select("_r"), on="_r", how="anti").with_columns(
        pl.col("remittance_info").str.extract(r"(\d{4,})$").str.strip_chars_start("0").alias("rdig"))
    mand = ctx.customers.filter(pl.col("sepa_mandate_id").is_not_null()).select("sepa_mandate_id", pl.col("customer_id").alias("party_id"))
    rest = rest.join(mand, left_on="mandate_id", right_on="sepa_mandate_id", how="left")
    cand = rest.join(ari.rename({"entity": "ientity"}), on="party_id", how="inner").filter(
        pl.col("docnorm").str.ends_with(pl.col("rdig")) & ((pl.col("total") - pl.col("dd_amt")).abs() < 0.01))
    by_mandate = cand.sort("issue_date").unique("_r", keep="first")
    rest2 = rest.join(by_mandate.select("_r"), on="_r", how="anti").drop("party_id")
    cand2 = rest2.join(ari, on="entity", how="inner").filter(
        pl.col("docnorm").str.ends_with(pl.col("rdig")) & ((pl.col("total") - pl.col("dd_amt")).abs() < 0.01))
    by_suffix = cand2.sort("issue_date").unique("_r", keep="first")
    cols = ddrows.columns + ["item_id"]
    return pl.concat([exact.select(cols), by_mandate.select(cols), by_suffix.select(cols)])


def dd_batches(ctx: Ctx, pending: pl.DataFrame) -> pl.DataFrame:
    if ctx.dd is None:
        return empty_rows()
    batches = ctx.dd.group_by("batch_id", "account_id").agg(_f("batch_total").first().alias("btot"))
    cand = _tokens(pending.filter(pl.col("is_credit")), r"\d{8,14}").join(
        batches, left_on=["account_id", "tok"], right_on=["account_id", "batch_id"]).filter((pl.col("amt") - pl.col("btot")).abs() < 0.005)
    link = cand.unique("bank_txn_id", keep="first").unique("tok", keep="first").select("bank_txn_id", pl.col("tok").alias("batch_id"), "account_id")
    rows = ctx.dd.join(link, on=["batch_id", "account_id"]).join(_acc_entity(ctx), on="account_id").rename({"acc_entity": "entity"})
    rows = _dd_rows_to_items(ctx, rows)
    return mk_rows(rows, "dd_batch", amount="dd_amt")


def dd_returns(ctx: Ctx, pending: pl.DataFrame) -> pl.DataFrame:
    if ctx.dd is None:
        return empty_rows()
    deb = pending.filter(~pl.col("is_credit")).with_columns(pl.col("alltext").str.extract(r"(VTM-[A-Z]{2}-\d{5,8})").alias("mandate"))
    deb = deb.filter(pl.col("mandate").is_not_null())
    if deb.height == 0:
        return empty_rows()
    ddr = ctx.dd.join(_acc_entity(ctx), on="account_id").rename({"acc_entity": "entity"})
    cand = deb.select("bank_txn_id", "mandate", "abs_amt", "booking_date", pl.col("account_id").alias("bacc")).join(
        ddr, left_on="mandate", right_on="mandate_id").filter(
        ((_f("amount") - pl.col("abs_amt")).abs() < 0.01) & (pl.col("collection_date") <= pl.col("booking_date")))
    cand = cand.sort(["bank_txn_id", "collection_date"], descending=[False, True]).unique("bank_txn_id", keep="first")
    cand = cand.with_columns(pl.col("mandate").alias("mandate_id"))
    rows = _dd_rows_to_items(ctx, cand.drop("mandate"))
    return mk_rows(rows.with_columns(pl.col("abs_amt").alias("amount")), "dd_return", "RETURNED")


# --------------------------------------------------------------------------------------------- cards
def card_settlements(ctx: Ctx, pending: pl.DataFrame) -> pl.DataFrame:
    if ctx.cards is None:
        return empty_rows()
    st = ctx.cards.group_by("settlement_id", "account_id", "merchant_id", "settlement_date").agg(
        _f("net").sum().round(2).alias("snet"))
    cand = _tokens(pending.filter(pl.col("is_credit")), r"\d{9}").join(
        st, left_on=["account_id", "tok"], right_on=["account_id", "merchant_id"]).filter(
        ((pl.col("amt") - pl.col("snet")).abs() < 0.015)
        & (pl.col("booking_date").str.to_date() >= pl.col("settlement_date").str.to_date())
        & (pl.col("booking_date").str.to_date() <= pl.col("settlement_date").str.to_date().dt.offset_by("6d")))
    cand = cand.with_columns((pl.col("booking_date").str.to_date() - pl.col("settlement_date").str.to_date()).dt.total_days().alias("lag"))
    link = cand.sort("lag").unique("bank_txn_id", keep="first").unique("settlement_id", keep="first").select("bank_txn_id", "settlement_id")
    rows = ctx.cards.join(link, on="settlement_id").join(_acc_entity(ctx), on="account_id")
    ari = ctx.items.filter(pl.col("item_type") == "ARI").select("item_id", "entity", "docnorm")
    rows = rows.with_columns(norm_expr("order_number").alias("onorm")).join(
        ari, left_on=["acc_entity", "onorm"], right_on=["entity", "docnorm"]).unique(["bank_txn_id", "item_id"])
    return mk_rows(rows.with_columns(_f("gross").alias("amount")), "card_settlement")


def card_chargebacks(ctx: Ctx, pending: pl.DataFrame) -> pl.DataFrame:
    if ctx.cards is None:
        return empty_rows()
    deb = pending.filter(~pl.col("is_credit") & pl.col("alltext").str.contains("CHARGEBACK|CONTRACARGO")).with_columns(
        pl.col("alltext").str.extract(r"(\*{4}\d{4})").alias("card"),
        pl.col("alltext").str.extract(r"(?:MID|COMERC)\s*(\d{9})").alias("mid"),
        pl.col("alltext").str.extract(r"OP\s+([A-Z0-9-]{6,})").alias("op"))
    if deb.height == 0:
        return empty_rows()
    cs = ctx.cards.join(_acc_entity(ctx), on="account_id").with_columns(_f("gross").alias("gross_f"))
    by_op = deb.filter(pl.col("op").is_not_null()).select("bank_txn_id", "abs_amt", "op").join(
        cs, left_on="op", right_on="order_number").filter((pl.col("gross_f") - pl.col("abs_amt")).abs() < 0.01).with_columns(pl.col("op").alias("order_number"))
    rest = deb.join(by_op.select("bank_txn_id"), on="bank_txn_id", how="anti")
    by_card = rest.select("bank_txn_id", "abs_amt", "card", "mid", "booking_date").join(
        cs, left_on=["card", "mid"], right_on=["card", "merchant_id"]).filter(
        ((pl.col("gross_f") - pl.col("abs_amt")).abs() < 0.01) & (pl.col("operation_date") <= pl.col("booking_date")))
    by_card = by_card.sort("operation_date", descending=True).unique("bank_txn_id", keep="first")
    cols = ["bank_txn_id", "abs_amt", "order_number", "acc_entity"]
    m = pl.concat([by_op.select(cols), by_card.select(cols)]).with_columns(norm_expr("order_number").alias("onorm"))
    ari = ctx.items.filter(pl.col("item_type") == "ARI").select("item_id", "entity", "docnorm")
    rows = m.join(ari, left_on=["acc_entity", "onorm"], right_on=["entity", "docnorm"]).unique("bank_txn_id")
    return mk_rows(rows.with_columns(pl.col("abs_amt").alias("amount")), "card_chargeback", "RETURNED")


# --------------------------------------------------------------------------------------------- cheques
def _cheques_to_items(ctx: Ctx, cq: pl.DataFrame) -> pl.DataFrame:
    """cq: cheque rows (+ acc_entity, bank_txn_id). Adds item_id via stub reference, else drawer name + amount."""
    ari = ctx.items.filter(pl.col("item_type") == "ARI").select("item_id", "entity", "docnorm", "party_id", "party_name", "total", "issue_date")
    cq = cq.with_row_index("_r").with_columns(norm_expr("stub_reference").alias("snorm"), _f("amount").alias("cq_amt"),
                                              norm_expr("drawer").alias("dnorm"))
    exact = cq.filter(pl.col("snorm") != "").join(ari, left_on=["acc_entity", "snorm"], right_on=["entity", "docnorm"]).unique("_r")
    rest = cq.join(exact.select("_r"), on="_r", how="anti")
    cust = ctx.customers.select(pl.col("customer_id").alias("party_id"), "entity", norm_expr("name").alias("dnorm"))
    rest = rest.join(cust, left_on=["acc_entity", "dnorm"], right_on=["entity", "dnorm"], how="left")
    by_name = rest.filter(pl.col("party_id").is_not_null()).join(ari.drop("entity"), on="party_id").filter(
        (pl.col("total") - pl.col("cq_amt")).abs() < 0.01).sort("issue_date").unique("_r", keep="first")
    rest2 = rest.join(by_name.select("_r"), on="_r", how="anti").drop("party_id")
    by_amt = rest2.join(ari.with_columns(norm_expr("party_name").alias("pn")), left_on=["acc_entity", "dnorm"], right_on=["entity", "pn"]).filter(
        (pl.col("total") - pl.col("cq_amt")).abs() < 0.01).sort("issue_date").unique("_r", keep="first")
    cols = cq.columns + ["item_id"]
    return pl.concat([exact.select(cols), by_name.select(cols), by_amt.select(cols)])


def cheque_deposits(ctx: Ctx, pending: pl.DataFrame) -> pl.DataFrame:
    if ctx.cheques is None:
        return empty_rows()
    dep = ctx.cheques.group_by("deposit_id", "account_id").agg(_f("deposit_total").first().alias("dtot"))
    cand = _tokens(pending.filter(pl.col("is_credit")), r"[A-Z]{2,4}\d{8,16}").join(
        dep, left_on=["account_id", "tok"], right_on=["account_id", "deposit_id"]).filter((pl.col("amt") - pl.col("dtot")).abs() < 0.015)
    link = cand.unique("bank_txn_id", keep="first").unique("tok", keep="first").select("bank_txn_id", pl.col("tok").alias("deposit_id"))
    rows = ctx.cheques.join(link, on="deposit_id").join(_acc_entity(ctx), on="account_id")
    rows = _cheques_to_items(ctx, rows).unique(["bank_txn_id", "item_id"])
    return mk_rows(rows.with_columns(pl.col("cq_amt").alias("amount")), "cheque_deposit")


def cheque_returns(ctx: Ctx, pending: pl.DataFrame) -> pl.DataFrame:
    if ctx.cheques is None:
        return empty_rows()
    deb = pending.filter(~pl.col("is_credit")).with_columns(
        pl.col("alltext").str.extract(r"(?:CHK|CHEQUE IMPAYE|CHEQUE DEVUELTO|CHQ)\s*N?\D{0,3}(\d{6,8})").alias("chk")).filter(pl.col("chk").is_not_null())
    if deb.height == 0:
        return empty_rows()
    cq = ctx.cheques.join(_acc_entity(ctx), on="account_id")
    cand = deb.select("bank_txn_id", "abs_amt", "chk", "booking_date", pl.col("account_id").alias("bacc")).join(
        cq, left_on=["chk", "bacc"], right_on=["cheque_number", "account_id"]).filter(
        ((_f("amount") - pl.col("abs_amt")).abs() < 0.01) & (pl.col("deposit_date") <= pl.col("booking_date")))
    cand = cand.sort("deposit_date", descending=True).unique("bank_txn_id", keep="first").with_columns(pl.col("chk").alias("cheque_number"))
    rows = _cheques_to_items(ctx, cand.drop("chk"))
    return mk_rows(rows.with_columns(pl.col("abs_amt").alias("amount")), "cheque_return", "RETURNED")


# --------------------------------------------------------------------------------------------- payment runs (AP)
PAYFACTORY = {"ES00": ["ES00", "ES01", "ES02", "PT01"]}


def _pr_rows_to_items(ctx: Ctx, prr: pl.DataFrame) -> pl.DataFrame:
    """prr: payment_runs rows (+ debtor entity 'acc_entity'). Adds item_id, item currency, open amount."""
    items = ctx.items.filter(pl.col("item_type").is_in(["API", "APC"])).select(
        "item_id", "item_type", "entity", "docnorm", "party_name", "currency", "open_amount", "total")
    prr = prr.with_row_index("_r").with_columns(
        norm_expr("document_number").alias("dnorm"),
        pl.when(pl.col("document_type") == "CREDIT_NOTE").then(pl.lit("APC")).otherwise(pl.lit("API")).alias("want_type"),
        _f("document_amount").abs().alias("doc_amt"))
    # expand each row to the entities whose items it may pay (payment factory), then join on (doc, type, entity): no blow-up
    prr = prr.with_columns(pl.when(pl.col("acc_entity") == "ES00").then(pl.lit(PAYFACTORY["ES00"]))
                           .otherwise(pl.concat_list([pl.col("acc_entity")])).alias("pay_entity")).explode("pay_entity")
    cand = prr.join(items, left_on=["dnorm", "want_type", "pay_entity"], right_on=["docnorm", "item_type", "entity"], suffix="_i")
    cand = cand.with_columns(pl.col("pay_entity").alias("entity")).with_columns(
        (pl.col("party_name").str.to_uppercase() == pl.col("beneficiary").str.to_uppercase()).alias("same_name"),
        (pl.col("entity") == pl.col("acc_entity")).alias("same_ent"),
        (pl.col("total") - pl.col("doc_amt")).abs().alias("dist"))
    best = cand.sort(["same_name", "same_ent", "dist"], descending=[True, True, False]).unique("_r", keep="first")
    return best


def _pr_cumulative(ctx: Ctx, pr: pl.DataFrame) -> pl.DataFrame:
    """Cumulative amount paid per item through payment runs, in execution order -> (acc, end_to_end_id, item_id, cum_paid)."""
    m = _pr_rows_to_items(ctx, pr).sort(["execution_date", "end_to_end_id"])
    return m.with_columns(pl.col("doc_amt").cum_sum().over("item_id").alias("cum_paid")).select(
        "acc", "end_to_end_id", "item_id", "cum_paid").unique(["acc", "end_to_end_id", "item_id"])


def _pr_exception(bank_ccy: str = "bank_ccy") -> pl.Expr:
    paid = pl.col("cum_paid").fill_null(pl.col("doc_amt"))
    partial = paid < pl.col("open_amount") - pl.max_horizontal(pl.lit(0.01), 0.005 * pl.col("open_amount"))
    return (pl.when(pl.col("dup")).then(pl.lit("DUPLICATE_PAYMENT"))
            .when(pl.col("want_type") == "APC").then(pl.lit(""))
            .when(partial).then(pl.lit("PARTIAL"))
            .when(pl.col(bank_ccy) != pl.col("item_ccy")).then(pl.lit("FX_DIFFERENCE"))
            .otherwise(pl.lit("")))


def payment_runs(ctx: Ctx, pending: pl.DataFrame) -> pl.DataFrame:
    if ctx.payruns is None:
        return empty_rows()
    # NB: run ids / e2e ids are NOT unique across debtor accounts -> every join is (id, acc)
    pr = ctx.payruns.join(_acc_entity(ctx), left_on="debtor_account_id", right_on="account_id").with_columns(
        _f("transfer_amount").alias("tamt"), _f("document_amount").alias("damt"), pl.col("debtor_account_id").alias("acc"))
    deb = pending.filter(~pl.col("is_credit")).drop("end_to_end_id").with_columns(pl.col("account_id").alias("acc")).with_columns(
        pl.col("alltext").str.extract(r"((?:PR|CF)\d{12})").alias("run_id"),
        pl.col("alltext").str.extract(r"((?:PR|CF)\d{12}-\d{4})").alias("e2e"),
        pl.col("alltext").str.contains(r"(?:PR|CF)\d{12}-\d{4}D").alias("dup"),
        pl.col("alltext").str.extract(r"(\d+)\s+PROVEEDORES").cast(pl.Int32, strict=False).alias("n_benef"),
    ).filter(pl.col("run_id").is_not_null())
    if deb.height == 0:
        return empty_rows()
    deb = deb.with_columns(
        pl.when(pl.col("orig_amt").is_not_null()).then(pl.col("orig_amt").abs()).otherwise(pl.col("abs_amt")).alias("cmp_amt"))
    e2e = pr.group_by("acc", "run_id", "run_type", "end_to_end_id").agg(pl.col("tamt").first(), pl.col("execution_date").first())
    K = ["bank_txn_id", "acc", "end_to_end_id"]
    # 1) full e2e in the bank text (transfers, all formats except N43 truncation)
    l1 = deb.filter(pl.col("e2e").is_not_null()).join(e2e.select("acc", "end_to_end_id", "tamt"), left_on=["acc", "e2e"],
                                                      right_on=["acc", "end_to_end_id"]).select("bank_txn_id", "acc", pl.col("e2e").alias("end_to_end_id"))
    rest = deb.join(l1, on="bank_txn_id", how="anti")
    # 2) run id + amount == one transfer of the run (N43 truncated e2e)
    l2 = rest.join(e2e.filter(pl.col("run_type") == "transfers"), on=["acc", "run_id"]).filter(
        (pl.col("tamt") - pl.col("cmp_amt")).abs() < 0.015)
    l2 = l2.sort("end_to_end_id").unique("bank_txn_id", keep="first").unique(["acc", "end_to_end_id"], keep="first").select(K)
    rest = rest.join(l2, on="bank_txn_id", how="anti")
    # 3) aggregated: one debit for the whole run
    run_tot = e2e.group_by("acc", "run_id").agg(pl.col("tamt").sum().round(2).alias("rtot"), pl.col("end_to_end_id").alias("e2es"))
    l3 = rest.join(run_tot, on=["acc", "run_id"]).filter((pl.col("rtot") - pl.col("cmp_amt")).abs() < 0.015).explode("e2es").select(
        "bank_txn_id", "acc", pl.col("e2es").alias("end_to_end_id"))
    rest = rest.join(l3, on="bank_txn_id", how="anti")
    # 4) confirming: one debit per maturity date = the beneficiaries (e2e) whose invoices fall due that day
    links = [l1, l2, l3]
    conf = rest.filter(pl.col("run_id").str.starts_with("CF"))
    if conf.height:
        due = _pr_rows_to_items(ctx, pr.filter(pl.col("run_type") == "confirming")).join(
            ctx.items.select("item_id", "due_date"), on="item_id", how="left")
        e2e_due = due.group_by("acc", "run_id", "end_to_end_id").agg(
            pl.col("due_date").max().alias("dmax"), pl.col("due_date").min().alias("dmin"), pl.col("tamt").first())
        groups = []
        for key in ("dmax", "dmin"):
            groups.append(e2e_due.group_by("acc", "run_id", pl.col(key).alias("due_date")).agg(
                pl.col("tamt").sum().round(2).alias("dtot"), pl.col("end_to_end_id").alias("e2e_list"), pl.len().alias("nb")))
        # single beneficiary maturing alone ("1 PROVEEDORES")
        groups.append(e2e_due.select("acc", "run_id", pl.col("dmax").alias("due_date"), pl.col("tamt").round(2).alias("dtot"),
                                     pl.concat_list([pl.col("end_to_end_id")]).alias("e2e_list"), pl.lit(1, pl.UInt32).alias("nb")))
        by_day = pl.concat(groups)
        c = conf.join(by_day, on=["acc", "run_id"]).filter(
            ((pl.col("dtot") - pl.col("cmp_amt")).abs() < 0.015) & (pl.col("n_benef").is_null() | (pl.col("n_benef") == pl.col("nb")))
        ).with_columns((pl.col("booking_date").str.to_date() - pl.col("due_date").str.to_date()).dt.total_days().abs().fill_null(999).alias("lag"))
        c = c.sort("lag").unique("bank_txn_id", keep="first")
        links.append(c.explode("e2e_list").select("bank_txn_id", "acc", pl.col("e2e_list").alias("end_to_end_id")))
    link = pl.concat(links).unique(K)
    rows = pr.join(link, on=["acc", "end_to_end_id"]).join(
        deb.select("bank_txn_id", pl.col("currency").alias("bank_ccy"), "dup"), on="bank_txn_id")
    rows = _pr_rows_to_items(ctx, rows).rename({"currency": "item_ccy"})
    rows = rows.unique(["bank_txn_id", "item_id"])
    involved = pr.join(rows.select(pl.col("dnorm").alias("_d")).unique(), left_on=norm_expr("document_number"), right_on="_d", how="semi")
    rows = rows.join(_pr_cumulative(ctx, involved), on=["acc", "end_to_end_id", "item_id"], how="left")
    return mk_rows(rows.with_columns(pl.col("doc_amt").alias("amount")), "payment_run", _pr_exception())


def payment_returns(ctx: Ctx, pending: pl.DataFrame) -> pl.DataFrame:
    """Our transfer to a supplier came back: credit line citing the full e2e id."""
    if ctx.payruns is None:
        return empty_rows()
    cr = pending.filter(pl.col("is_credit")).with_columns(pl.col("alltext").str.extract(r"((?:PR|CF)\d{12}-\d{4})").alias("e2e")).filter(
        pl.col("e2e").is_not_null())
    if cr.height == 0:
        return empty_rows()
    pr = ctx.payruns.join(_acc_entity(ctx), left_on="debtor_account_id", right_on="account_id")
    rows = pr.join(cr.select("bank_txn_id", pl.col("e2e").alias("end_to_end_id"), pl.col("account_id").alias("debtor_account_id")),
                   on=["end_to_end_id", "debtor_account_id"])
    rows = _pr_rows_to_items(ctx, rows).unique(["bank_txn_id", "item_id"])
    return mk_rows(rows.with_columns(pl.col("doc_amt").alias("amount")), "payment_return", "RETURNED")


# --------------------------------------------------------------------------------------------- AP: party + exact amount
def _party_amount_match(ctx: Ctx, lines: pl.DataFrame, items: pl.DataFrame, module: str) -> pl.DataFrame:
    """Lines whose amount equals an open item of a party named (or whose doc number appears) in the bank text."""
    if lines.height == 0 or items.height == 0:
        return empty_rows()
    lines = lines.join(_acc_entity(ctx), on="account_id").with_columns(
        norm_expr("alltext").alias("tnorm"), pl.col("abs_amt").round(2).alias("amt2"),
        pl.when(pl.col("acc_entity") == "ES00").then(pl.lit(PAYFACTORY["ES00"])).otherwise(pl.concat_list([pl.col("acc_entity")])).alias("ents")
    ).explode("ents")
    items = items.with_columns(pl.col("open_amount").round(2).alias("amt2"), norm_expr("party_name").str.slice(0, 10).alias("pn10"))
    c = lines.select("bank_txn_id", "booking_date", "tnorm", "amt2", "ents").join(items, left_on=["ents", "amt2"], right_on=["entity", "amt2"])
    c = c.with_columns(
        ((pl.col("docnorm").str.len_chars() >= 4) & pl.col("tnorm").str.contains(pl.col("docnorm"), literal=True)).alias("doc_in"),
        ((pl.col("pn10").str.len_chars() >= 5) & pl.col("tnorm").str.contains(pl.col("pn10"), literal=True)).alias("name_in"),
        (pl.col("booking_date").str.to_date() - pl.col("due_date").fill_null(pl.col("issue_date")).str.to_date()).dt.total_days().abs().alias("lag"),
    ).filter(pl.col("doc_in") | pl.col("name_in"))
    c = c.sort(["doc_in", "lag"], descending=[True, False]).unique("bank_txn_id", keep="first").unique("item_id", keep="first")
    return mk_rows(c.with_columns(pl.col("open_amount").alias("amount")), module)


def ap_direct_debits(ctx: Ctx, pending: pl.DataFrame) -> pl.DataFrame:
    """Suppliers collecting from us by SEPA direct debit (MREF/RUM … RECIBO …): one API item, exact amount, supplier in text."""
    items = ctx.items.filter((pl.col("item_type") == "API") & (pl.col("payment_method") == "dd"))
    return _party_amount_match(ctx, pending.filter(~pl.col("is_credit")), items, "ap_direct_debit")


def supplier_refunds(ctx: Ctx, pending: pl.DataFrame) -> pl.DataFrame:
    """A supplier refunds one of its credit notes by transfer: credit line = APC amount, supplier in text."""
    items = ctx.items.filter(pl.col("item_type") == "APC")
    return _party_amount_match(ctx, pending.filter(pl.col("is_credit")), items, "supplier_refund")


ALL = [dd_batches, dd_returns, card_settlements, card_chargebacks, cheque_deposits, cheque_returns, payment_runs, payment_returns]
LATE_AP = [ap_direct_debits, supplier_refunds]   # run after strong categories (leasing/insurance DDs look alike)
