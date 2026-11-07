"""Invoice PDF clauses (POLICIES §3, §5.7-5.8): instalments, guarantee retention, advance deduction, cancel-and-replace.

parse_phase(phase): one row per invoice PDF with a clause -> data/parquet/<phase>/pdf_clauses.parquet
  file, entity, document_number, kind (INST/RET/ADV/REPL), n, days, amounts, ret_amount, payable, replaces
Clause templates are EN/ES only (checked on LIVE), wrapped lines are joined before matching.

Resolver `pdf_clauses` (runs before ar_exact): a pending receipt whose amount (to the cent) equals an amount the clause says
will be paid, for an open invoice of a customer evidenced in the text (name / IBAN / doc number):
  instalment k of n  -> PARTIAL (k < n) or no exception (last); k = instalment whose due date is closest to the booking date
  retention: payable part -> PARTIAL;  later release of the retained amount -> no exception
  advance: net amount due -> PARTIAL
  payment citing a cancelled invoice X -> the replacement invoice (no exception)
amount = cash received (= the expected amount).
"""
from __future__ import annotations

import os
import re
import time
from concurrent.futures import ProcessPoolExecutor

import polars as pl

from .data import DATA, PQ, Ctx, norm_expr
from .deterministic import _acc_entity, empty_rows, mk_rows

NUMS = r"\d[\d.,\s]*\d"
INST = re.compile(r"(?:Payment terms:\s*(\d+)\s+instalments at\s+([\d/]+)\s+days|Forma de pago:\s*(\d+)\s+vencimientos a\s+([\d/]+)\s+d[ií]as)\s*\(([^)]*)\)")
RET = re.compile(r"(?:Retention\s+(\d+)\s*%|Retenci[oó]n en garant[ií]a del\s+(\d+)\s*%)\s*\((" + NUMS + r")\s*[A-Z]{3}\).*?"
                 r"(?:Amount payable at due date|Importe a pagar al vencimiento):\s*(" + NUMS + r")\s*[A-Z]{3}")
ADV = re.compile(r"(?:Less advance received against proforma|A deducir anticipo recibido seg[uú]n proforma)\s+(PF[\w-]+)\s*(?::|de)\s*(" + NUMS +
                 r")\s*[A-Z]{3}\.?\s*(?:Net amount due|Importe neto a pagar)\s*(" + NUMS + r")")
REPL = re.compile(r"(?:cancels and replaces invoice|anula y sustituye a la factura)\s+([A-Z0-9][A-Z0-9\-/]+)")


def _num(s: str) -> float | None:
    s = s.strip().replace(" ", "").replace(" ", "")
    if re.search(r",\d{2}$", s):
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def _parse_one(args):
    path, f, ent, docnum = args
    try:
        from pypdf import PdfReader
        t = " ".join((p.extract_text() or "") for p in PdfReader(path).pages)
    except Exception:
        return []
    t = re.sub(r"\s+", " ", t)
    out = []
    m = INST.search(t)
    if m:
        n = int(m.group(1) or m.group(3))
        days = [int(x) for x in (m.group(2) or m.group(4)).split("/") if x]
        amts = [_num(re.sub(r"[A-Z]{3}", "", a)) for a in m.group(5).split(" / ")]
        out.append(dict(kind="INST", n=n, days=days, amounts=[a for a in amts if a is not None]))
    m = RET.search(t)
    if m:
        out.append(dict(kind="RET", n=None, days=None, amounts=[_num(m.group(4)), _num(m.group(3))]))   # [payable, retained]
    m = ADV.search(t)
    if m:
        out.append(dict(kind="ADV", n=None, days=None, amounts=[_num(m.group(3))]))
    m = REPL.search(t)
    if m:
        out.append(dict(kind="REPL", n=None, days=None, amounts=[], replaces=m.group(1)))
    return [dict(file=f, entity=ent, document_number=docnum, replaces=o.pop("replaces", None), **o) for o in out]


SCHEMA = {"file": pl.Utf8, "entity": pl.Utf8, "document_number": pl.Utf8, "replaces": pl.Utf8, "kind": pl.Utf8, "n": pl.Int32,
          "days": pl.List(pl.Int32), "amounts": pl.List(pl.Float64)}


def parse_phase(phase: str, force: bool = False, workers: int = 6) -> pl.DataFrame | None:
    cache = os.path.join(PQ, phase, "pdf_clauses.parquet")
    if os.path.exists(cache) and not force:
        return pl.read_parquet(cache)
    base = os.path.join(DATA, phase, "pdfs")
    idx = os.path.join(base, "index.csv")
    if not os.path.exists(idx):
        return None
    index = pl.read_csv(idx, infer_schema_length=0).filter(pl.col("document_type") == "invoice")
    jobs = [(os.path.join(base, f), f, e, d) for f, e, d in index.select("file", "entity", "document_number").iter_rows()
            if os.path.exists(os.path.join(base, f))]
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        rows = [r for rs in ex.map(_parse_one, jobs, chunksize=64) for r in rs]
    df = pl.DataFrame(rows, schema=SCHEMA) if rows else pl.DataFrame(schema=SCHEMA)
    df.write_parquet(cache)
    print(f"  {phase}: {len(jobs):,} invoice PDFs -> {df.height:,} clauses {df.group_by('kind').len().rows()} ({time.time() - t0:.0f}s)")
    return df


def _expected(ctx: Ctx, cl: pl.DataFrame) -> pl.DataFrame:
    """clauses -> expected receipts: item_id, exp (2 dec), kind, k, n, due (date of instalment k)."""
    items = ctx.all_items.filter(pl.col("item_type") == "ARI").select("item_id", "entity", "docnorm", "issue_date")
    cl = cl.with_columns(norm_expr("document_number").alias("docnorm")).join(items, on=["entity", "docnorm"])
    inst = cl.filter((pl.col("kind") == "INST") & (pl.col("amounts").list.len() == pl.col("days").list.len())).with_columns(
        pl.int_ranges(1, pl.col("amounts").list.len() + 1).alias("k"), pl.col("amounts").list.len().cast(pl.Int32).alias("n"),
        pl.col("amounts").list.eval(pl.element().cum_sum()).alias("cum")).explode(
        ["amounts", "days", "k", "cum"]).select("item_id", pl.col("amounts").round(2).alias("exp"), pl.lit("INST").alias("kind"), "k", "n",
                                                (pl.col("issue_date").str.to_date() + pl.duration(days=pl.col("days"))).alias("due"),
                                                (pl.col("cum") - pl.col("amounts")).round(2).alias("cum_before"))   # paid by instalments 1..k-1
    ret = cl.filter(pl.col("kind") == "RET")
    ret = pl.concat([
        ret.select("item_id", pl.col("amounts").list.get(0).round(2).alias("exp"), pl.lit("RET1").alias("kind")),
        ret.select("item_id", pl.col("amounts").list.get(1).round(2).alias("exp"), pl.lit("RET2").alias("kind"))]).with_columns(
        pl.lit(None, pl.Int64).alias("k"), pl.lit(None, pl.Int32).alias("n"), pl.lit(None, pl.Date).alias("due"), pl.lit(None, pl.Float64).alias("cum_before"))
    adv = cl.filter(pl.col("kind") == "ADV").select("item_id", pl.col("amounts").list.get(0).round(2).alias("exp"), pl.lit("ADV").alias("kind"),
                                                    pl.lit(None, pl.Int64).alias("k"), pl.lit(None, pl.Int32).alias("n"), pl.lit(None, pl.Date).alias("due"),
                                                    pl.lit(None, pl.Float64).alias("cum_before"))
    return pl.concat([inst.with_columns(pl.col("k").cast(pl.Int64), pl.col("n").cast(pl.Int32)), ret, adv], how="vertical_relaxed")


def pdf_clauses(ctx: Ctx, pending: pl.DataFrame) -> pl.DataFrame:
    from .ar_exact import _names
    cl = parse_phase(ctx.phase)
    if cl is None or cl.height == 0:
        return empty_rows()
    exp = _expected(ctx, cl)
    open_ari = ctx.items.filter(pl.col("item_type") == "ARI").select("item_id", "entity", "party_id", "currency", "docnorm", "issue_date",
                                                                      "total", "open_amount", norm_expr("party_name").str.slice(0, 10).alias("k0"))
    exp = exp.join(open_ari, on="item_id")
    lines = pending.filter(pl.col("is_credit") & pl.col("orig_amt").is_null()).join(_acc_entity(ctx), on="account_id").select(
        "bank_txn_id", "booking_date", "currency", pl.col("acc_entity").alias("entity"), pl.col("abs_amt").round(2).alias("cash"),
        (pl.col("abs_amt") * 100).round(0).cast(pl.Int64).alias("ck"),
        norm_expr("alltext").alias("tnorm"), pl.col("counterparty_account").alias("cp_iban"))
    # amounts printed on a PDF are rounded on their own (payable = total - retention, instalment splits): allow one cent of
    # difference between the clause amount and the cash (POLICIES §2 tolerance is max(0.01, 0.5 %)); the row keeps the cash.
    exp = exp.with_columns((pl.col("exp") * 100).round(0).cast(pl.Int64).alias("ck"))
    exp = pl.concat([exp.with_columns((pl.col("ck") + d).alias("ck")) for d in (-1, 0, 1)])
    c = lines.join(exp, on=["entity", "currency", "ck"]).filter(pl.col("issue_date") <= pl.col("booking_date")).join(
        _names(ctx), on="party_id", how="left")

    def has(k):
        return ((pl.col(k).str.len_chars() >= 5) & pl.col("tnorm").str.contains(pl.col(k), literal=True)).fill_null(False)

    digits = pl.col("docnorm").str.replace_all(r"\D", "")
    c = c.with_columns(
        ((pl.col("docnorm").str.len_chars() >= 6) & pl.col("tnorm").str.contains(pl.col("docnorm"), literal=True)).alias("doc"),
        (has("k0") | has("k1") | has("k2") | (pl.col("cp_iban").is_not_null() & (pl.col("cp_iban") == pl.col("iban"))).fill_null(False)
         | ((digits.str.len_chars() >= 6) & pl.col("tnorm").str.contains(digits.str.slice(-6), literal=True))).alias("ev"))
    c = c.filter(pl.col("doc") | pl.col("ev")).with_columns(
        (pl.col("booking_date").str.to_date() - pl.col("due")).dt.total_days().abs().fill_null(0).alias("lag"),
        # an instalment the balance says is already paid (its predecessors' sum + its own amount <= what was paid before this
        # resolver ran) cannot be paid again. The balance only EXCLUDES; it never picks: this resolver sees the whole window
        # at once, so several instalments of one invoice inside the window share the same opening balance and must be told
        # apart by their due dates below.
        ((pl.col("kind") != "INST") | (pl.col("cum_before") + pl.col("exp") > pl.col("total") - pl.col("open_amount") + 0.011)).alias("unpaid"))
    c = c.filter(pl.col("unpaid"))
    # instalment number: cited doc first, then the instalment whose due date is closest to the booking date (per line & item)
    c = c.sort(["doc", "lag"], descending=[True, False]).unique(["bank_txn_id", "item_id"], keep="first")
    c = c.unique("bank_txn_id", keep="first", maintain_order=True).unique(["item_id", "kind", "k"], keep="first", maintain_order=True)
    exc = (pl.when(pl.col("kind") == "INST").then(pl.when(pl.col("k") < pl.col("n")).then(pl.lit("PARTIAL")).otherwise(pl.lit("")))
           .when(pl.col("kind").is_in(["RET1", "ADV"])).then(pl.lit("PARTIAL")).otherwise(pl.lit("")))
    return mk_rows(c.with_columns(pl.col("cash").alias("amount")), "pdf_clause", exc)


def replaced_invoices(ctx: Ctx, pending: pl.DataFrame) -> pl.DataFrame:
    """A receipt citing a cancelled invoice X settles the replacement invoice (exact amount)."""
    cl = parse_phase(ctx.phase)
    if cl is None or cl.height == 0:
        return empty_rows()
    rp = cl.filter(pl.col("kind") == "REPL").select("entity", norm_expr("document_number").alias("docnorm"), norm_expr("replaces").alias("old"))
    items = ctx.items.filter(pl.col("item_type") == "ARI").select("item_id", "entity", "docnorm", "currency", pl.col("open_amount").round(2).alias("amt"))
    rp = rp.join(items, on=["entity", "docnorm"])
    toks = pending.filter(pl.col("is_credit") & pl.col("orig_amt").is_null()).join(_acc_entity(ctx), on="account_id").select(
        "bank_txn_id", "currency", pl.col("acc_entity").alias("entity"), pl.col("abs_amt").round(2).alias("amt"),
        pl.col("alltext").str.extract_all(r"[A-Z0-9][A-Z0-9\-/]{5,}").alias("t")).explode("t").filter(pl.col("t").is_not_null()).with_columns(
        norm_expr("t").alias("old"))
    c = toks.join(rp, on=["entity", "old", "currency", "amt"]).unique("bank_txn_id").unique("item_id")
    return mk_rows(c.with_columns(pl.col("amt").alias("amount")), "pdf_replaced")


RESOLVERS = [replaced_invoices, pdf_clauses]

if __name__ == "__main__":
    import sys
    for ph in sys.argv[1:] or ["live"]:
        parse_phase(ph, force=True)
