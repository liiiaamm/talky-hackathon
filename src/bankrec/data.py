"""Load one phase into a context of polars DataFrames.

ctx.bank     bank lines of the phase (+ helper columns: amt, abs_amt, is_credit, alltext)
ctx.items    every item that can be settled in the phase, unified schema:
             item_id, item_type (ARI/ARC/API/APC), entity, party_id, party_name, docnum, docnorm, issue_date, due_date,
             currency, total, open_amount, payment_terms, po, payment_method, related_document
ctx.dd / ctx.cards / ctx.cheques / ctx.payruns   structured files of the phase
ctx.customers / ctx.suppliers / ctx.accounts / ctx.terms / ctx.fx / ctx.categories   masters
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import polars as pl

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(ROOT, "data")
PQ = os.path.join(DATA, "parquet")
PHASES = ["train", "live", "final"]
SUBMIT_COLS = ["bank_txn_id", "item_id", "amount", "category", "exception"]
EXCEPTIONS = ["PARTIAL", "OVERPAYMENT", "SHORT_DISCOUNT", "SHORT_BANK_CHARGES", "SHORT_WITHHOLDING", "SHORT_DISPUTE",
              "DUPLICATE_PAYMENT", "UNIDENTIFIED", "RETURNED", "FX_DIFFERENCE"]


def norm_expr(col: str | pl.Expr) -> pl.Expr:
    e = pl.col(col) if isinstance(col, str) else col
    return e.fill_null("").str.to_uppercase().str.replace_all(r"[^A-Z0-9]", "")


def _pq(phase: str, name: str) -> pl.DataFrame | None:
    p = os.path.join(PQ, phase, f"{name}.parquet")
    return pl.read_parquet(p) if os.path.exists(p) else None


def _f(e: str) -> pl.Expr:
    return pl.col(e).cast(pl.Float64, strict=False)


def _items_from_tables(phase: str) -> pl.DataFrame | None:
    ar, ap, cn = _pq(phase, "ar_invoices"), _pq(phase, "ap_invoices"), _pq(phase, "credit_notes")
    parts = []
    if ar is not None:
        parts.append(ar.select(
            "item_id", pl.lit("ARI").alias("item_type"), "entity", pl.col("customer_id").alias("party_id"),
            pl.col("customer_name").alias("party_name"), pl.col("document_number").alias("docnum"), "issue_date", "due_date",
            "currency", _f("total_amount").alias("total"), "payment_terms", pl.col("customer_po").alias("po"), "payment_method",
            pl.lit(None, pl.Utf8).alias("related_document")))
    if ap is not None:
        parts.append(ap.select(
            "item_id", pl.lit("API").alias("item_type"), "entity", pl.col("supplier_id").alias("party_id"),
            pl.col("supplier_name").alias("party_name"), pl.col("supplier_invoice_number").alias("docnum"), "issue_date", "due_date",
            "currency", _f("total_amount").alias("total"), "payment_terms", pl.col("our_po").alias("po"), "payment_method",
            pl.lit(None, pl.Utf8).alias("related_document")))
    if cn is not None:
        parts.append(cn.select(
            "item_id", pl.when(pl.col("side") == "AR").then(pl.lit("ARC")).otherwise(pl.lit("APC")).alias("item_type"), "entity",
            "party_id", "party_name", pl.col("document_number").alias("docnum"), "issue_date", pl.lit(None, pl.Utf8).alias("due_date"),
            "currency", _f("total_amount").alias("total"), pl.lit(None, pl.Utf8).alias("payment_terms"), pl.lit(None, pl.Utf8).alias("po"),
            pl.lit(None, pl.Utf8).alias("payment_method"), "related_document"))
    return pl.concat(parts) if parts else None


def _read_master(name: str, phase: str) -> pl.DataFrame:
    p = os.path.join(DATA, phase, "masters_final", name) if phase == "final" else None
    if not (p and os.path.exists(p)):
        p = os.path.join(DATA, "masters", name)
    return pl.read_csv(p, infer_schema_length=0, quote_char='"')


@dataclass
class Ctx:
    phase: str
    bank: pl.DataFrame
    items: pl.DataFrame
    dd: pl.DataFrame | None = None
    cards: pl.DataFrame | None = None
    cheques: pl.DataFrame | None = None
    payruns: pl.DataFrame | None = None
    customers: pl.DataFrame | None = None
    suppliers: pl.DataFrame | None = None
    accounts: pl.DataFrame | None = None
    terms: pl.DataFrame | None = None
    fx: pl.DataFrame | None = None
    categories: list = field(default_factory=list)
    golden: pl.DataFrame | None = None


def load(phase: str, date_from: str | None = None, date_to: str | None = None) -> Ctx:
    """phase in train/live/final/sample. date_from/date_to (YYYY-MM-DD, inclusive) restrict the bank lines (dev windows on TRAIN)."""
    bank = _pq(phase, "bank_transactions")
    if date_from:
        bank = bank.filter(pl.col("booking_date") >= date_from)
    if date_to:
        bank = bank.filter(pl.col("booking_date") <= date_to)
    bank = bank.with_columns(
        _f("amount").alias("amt"), _f("orig_amount").alias("orig_amt"),
    ).with_columns(
        pl.col("amt").abs().alias("abs_amt"), (pl.col("amt") > 0).alias("is_credit"),
        pl.concat_str([pl.col(c).fill_null("") for c in ["text", "ref1", "ref2", "end_to_end_id", "remittance_info", "counterparty_name"]],
                      separator=" ").str.to_uppercase().alias("alltext"),
    )

    # ---- items: everything settle-able in the phase
    if phase in ("train", "sample"):
        items = _items_from_tables(phase).with_columns(pl.col("total").alias("open_amount"))
    else:
        cur = _items_from_tables(phase)
        oi = _pq(phase, "open_items_at_start")
        prior = [x for x in (_items_from_tables(p) for p in PHASES[:PHASES.index(phase)]) if x is not None]
        prior = pl.concat(prior).unique("item_id", keep="last") if prior else None
        oi = oi.select("item_id", _f("open_amount").alias("open_amount"))
        if prior is not None:
            carried = prior.join(oi, on="item_id", how="inner")
            missing = oi.join(prior.select("item_id"), on="item_id", how="anti")
            if missing.height:   # open items whose original document is not in our tables: rebuild from open_items_at_start
                raw = _pq(phase, "open_items_at_start").join(missing.select("item_id"), on="item_id")
                carried = pl.concat([carried, raw.select(
                    "item_id", "item_type", "entity", "party_id", pl.lit(None, pl.Utf8).alias("party_name"),
                    pl.col("document_number").alias("docnum"), "issue_date", "due_date", "currency", _f("total_amount").alias("total"),
                    pl.lit(None, pl.Utf8).alias("payment_terms"), pl.lit(None, pl.Utf8).alias("po"), pl.lit(None, pl.Utf8).alias("payment_method"),
                    pl.lit(None, pl.Utf8).alias("related_document"), _f("open_amount").alias("open_amount"))], how="diagonal_relaxed")
        else:
            raise RuntimeError("prior phase item tables are needed to enrich open items")
        items = pl.concat([carried, cur.with_columns(pl.col("total").alias("open_amount"))], how="diagonal_relaxed").unique("item_id", keep="last")
    items = items.with_columns(norm_expr("docnum").alias("docnorm"))

    ctx = Ctx(phase=phase, bank=bank, items=items)
    ctx.dd = _pq(phase, "direct_debit_batches")
    ctx.cards = _pq(phase, "card_settlements")
    ctx.cheques = _pq(phase, "cheque_deposits")
    ctx.payruns = _pq(phase, "payment_runs")
    ctx.customers = _read_master("customers.csv", phase)
    ctx.suppliers = _read_master("suppliers.csv", phase)
    ctx.accounts = _read_master("bank_accounts.csv", phase)
    ctx.terms = _read_master("payment_terms.csv", phase)
    ctx.fx = _read_master("fx_rates.csv", phase)
    ctx.categories = _read_master("categories.csv", phase)["category"].to_list()
    g = _pq(phase, "golden_allocations")
    if g is not None:
        ctx.golden = g.join(bank.select("bank_txn_id"), on="bank_txn_id", how="semi")
    return ctx
