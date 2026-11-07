"""Remittance advices (csv / xlsx / txt / pdf) -> allocation rows for the bank receipt they describe.

parse_phase(phase) parses every advice of the phase once and caches data/parquet/<phase>/ra_parsed.parquet with one row per
document line: file, entity, payer, received, ref, total, doc, gross, deduction, reason, paid.

Resolver `remittance_advices`: link advice -> pending credit line (payment ref token in the bank text, same entity, received
within -5..+20 days of the booking, and sum(paid) == bank amount when available), then one row per document:
  deduction with a discount reason -> SHORT_DISCOUNT, amount = open amount (fully cleared)
  other deduction (stated reason)   -> SHORT_DISPUTE,  amount = paid
  paid < gross, no reason           -> PARTIAL,        amount = paid
  credit note                        -> no exception,   amount = |paid|
  bank currency != item currency on fully-cleared invoice rows -> FX_DIFFERENCE
"""
from __future__ import annotations

import csv
import io
import os
import re
import time

import polars as pl

from .data import DATA, PQ, Ctx, norm_expr
from .deterministic import _acc_entity, empty_rows, mk_rows

DISCOUNT_RE = re.compile(r"DISC|DESC|SKONTO|ESCOMPTE|SCONTO|PRONTO|EARLY", re.I)
NUM = r"-?\(?[\d.,\s]*\d[.,]\d{2}\)?"


def _num(s) -> float | None:
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).strip().replace(" ", "").replace(" ", "")
    if not s:
        return None
    neg = s.startswith("-") or (s.startswith("(") and s.endswith(")"))
    s = s.strip("-()")
    if re.search(r",\d{2}$", s):          # 1.234,56
        s = s.replace(".", "").replace(",", ".")
    else:                                  # 1,234.56
        s = s.replace(",", "")
    try:
        v = float(s)
    except ValueError:
        return None
    return -v if neg else v


def _row(doc, gross, ded, reason, paid):
    gross, ded, paid = _num(gross), _num(ded), _num(paid)
    if paid is None and gross is not None:
        paid = gross + (ded or 0.0) if (ded or 0) < 0 else gross - (ded or 0.0)
    if ded is not None and ded > 0 and gross is not None and paid is not None and paid < gross:
        ded = -ded
    return dict(doc=str(doc).strip(), gross=gross, deduction=ded, reason=(reason or "").strip() or None, paid=paid)


def _parse_csv(text: str):
    lines = text.splitlines()
    delim = ";" if lines and lines[0].count(";") > lines[0].count(",") else ","
    rows = list(csv.reader(io.StringIO(text), delimiter=delim))
    head = rows[0] if rows else []
    meta = {head[i].strip().lower(): head[i + 1] for i in range(0, len(head) - 1, 2)}
    out = []
    hdr = None
    for r in rows[1:]:
        if not r:
            continue
        if hdr is None:
            hdr = [h.strip().lower() for h in r]
            continue
        d = dict(zip(hdr, r))
        doc = d.get("document") or d.get("doc") or r[0]
        if not doc or doc.upper().startswith("TOTAL"):
            continue
        out.append(_row(doc, d.get("gross"), d.get("discount_or_deduction") or d.get("deduction"), d.get("reason"), d.get("paid")))
    return meta.get("payment_ref"), _num(None), out


def _parse_xlsx(path: str):
    import openpyxl
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.worksheets[0]
    ref, hdr, out = None, None, []
    for r in ws.iter_rows(values_only=True):
        cells = [c for c in r]
        if hdr is None:
            if cells and isinstance(cells[0], str) and cells[0].strip().lower() in ("payment advice", "payment_ref", "payment ref.", "payment ref"):
                ref = str(cells[1]) if len(cells) > 1 else None
            if cells and isinstance(cells[0], str) and cells[0].strip().lower() in ("document", "doc", "voucher"):
                hdr = [str(c).strip().lower() if c else "" for c in cells]
            continue
        if not cells or cells[0] is None:
            continue
        d = dict(zip(hdr, cells))
        doc = d.get("invoice") or d.get("document") or cells[0]     # Sept-2025 layout: Voucher | Invoice (no prefix) | PO | … | Net paid
        if str(doc).upper().startswith("TOTAL"):
            continue
        out.append(_row(doc, d.get("gross", d.get("amount")), d.get("deduction") or d.get("discount_or_deduction"),
                        d.get("reason"), d.get("paid", d.get("net paid"))))
    return ref, None, out


TXT_LINE = re.compile(rf"^\s+(\S+)\s+(\d{{2}}/\d{{2}}/\d{{2,4}})\s+({NUM})(?:\s+({NUM})\s*(.*))?\s*$")


def _parse_txt(text: str):
    m = re.search(r"(?:Payment advice|Subject:.*?)\s+([A-Z0-9][A-Z0-9\-/]{5,})\s*$", text, re.M)
    ref = m.group(1) if m else None
    m = re.search(r"payment of\s+(" + NUM + r")\s+[A-Z]{3}", text)
    total = _num(m.group(1)) if m else None
    out = []
    for line in text.splitlines():
        mm = TXT_LINE.match(line)
        if mm:
            doc, _d, amt, ded, reason = mm.groups()
            g = _num(amt)
            if ded:
                dd = _num(ded)
                out.append(dict(doc=doc, gross=g, deduction=dd, reason=(reason or "").strip() or None, paid=round(g + dd, 2)))
            else:
                out.append(dict(doc=doc, gross=g, deduction=None, reason=None, paid=g))
    return ref, total, out


DATE_RE = re.compile(r"^\d{2}[/.-]\d{2}[/.-]\d{2,4}$|^\d{4}-\d{2}-\d{2}$")
NUM_RE = re.compile(rf"^{NUM}$")


DED_INLINE = re.compile(rf"^({NUM})\s+(\D.*)$")
DOC_RE = re.compile(r"^[A-Z0-9][A-Z0-9\-/.]{3,}$")


def _parse_pdf(path: str):
    from pypdf import PdfReader
    text = "\n".join((p.extract_text() or "") for p in PdfReader(path).pages)
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    ref = total = None
    for i, x in enumerate(lines[:-1]):
        xl = x.lower().rstrip(".:")
        if ref is None and xl in ("payment ref", "ref", "referencia", "réf", "ref. pagamento", "zahlungsreferenz", "riferimento"):
            ref = lines[i + 1]
        if total is None and xl in ("amount", "importe", "montant", "betrag", "importo", "valor", "kwota"):
            total = _num(lines[i + 1].split()[0])
    # the table starts at the first document-like line followed by a date
    start = next((i for i in range(len(lines) - 1) if DOC_RE.match(lines[i]) and DATE_RE.match(lines[i + 1])
                  and not DATE_RE.match(lines[i])), None)
    if start is None:
        return ref, total, []
    out, cur = [], None
    for x in lines[start:]:
        if x.startswith("Documento sint") or x.lower().startswith("total"):
            break
        if NUM_RE.match(x):
            if cur is not None:
                cur["nums"].append(x)
        elif DATE_RE.match(x):
            continue
        elif (mm := DED_INLINE.match(x)) and cur is not None and cur["nums"]:
            cur["ded"] = mm.group(1)
            cur["reason"].append(mm.group(2))
        elif DOC_RE.match(x) and (cur is None or len(cur["nums"]) >= 1):
            if cur is not None:
                out.append(cur)
            cur = {"doc": x, "nums": [], "reason": [], "ded": None}
        elif cur is not None:
            cur["reason"].append(x)
    if cur is not None:
        out.append(cur)
    rows = []
    for c in out:
        n = [_num(v) for v in c["nums"]]
        reason = " ".join(c["reason"]) or None
        ded = _num(c["ded"]) if c["ded"] else (n[1] if len(n) >= 3 else None)
        if not n:
            continue
        rows.append(dict(doc=c["doc"], gross=n[0], deduction=-abs(ded) if ded else None, reason=reason, paid=n[-1]))
    return ref, total, rows


def parse_phase(phase: str, force: bool = False) -> pl.DataFrame | None:
    cache = os.path.join(PQ, phase, "ra_parsed.parquet")
    if os.path.exists(cache) and not force:
        return pl.read_parquet(cache)
    base = os.path.join(DATA, phase, "remittance_advices")
    idx = os.path.join(base, "index.csv")
    if not os.path.exists(idx):
        return None
    index = pl.read_csv(idx, infer_schema_length=0)
    recs, t0, nerr = [], time.time(), 0
    for f, ent, payer, rdate, fmt in index.select("file", "entity", "payer_name", "received_date", "format").iter_rows():
        p = os.path.join(base, f)
        if not os.path.exists(p):
            continue
        try:
            if fmt == "csv":
                ref, total, rows = _parse_csv(open(p, encoding="utf-8-sig", errors="replace").read())
            elif fmt == "txt":
                ref, total, rows = _parse_txt(open(p, encoding="utf-8", errors="replace").read())
            elif fmt == "xlsx":
                ref, total, rows = _parse_xlsx(p)
            else:
                ref, total, rows = _parse_pdf(p)
        except Exception:
            nerr += 1
            continue
        for r in rows:
            recs.append(dict(file=f, entity=ent, payer=payer, received=rdate, fmt=fmt, ref=str(ref) if ref else None, total=total, **r))
    df = pl.DataFrame(recs, schema={"file": pl.Utf8, "entity": pl.Utf8, "payer": pl.Utf8, "received": pl.Utf8, "fmt": pl.Utf8,
                                    "ref": pl.Utf8, "total": pl.Float64, "doc": pl.Utf8, "gross": pl.Float64, "deduction": pl.Float64,
                                    "reason": pl.Utf8, "paid": pl.Float64})
    df.write_parquet(cache)
    print(f"  parsed {df['file'].n_unique():,} advices / {df.height:,} doc lines for {phase} ({time.time() - t0:.0f}s, {nerr} errors)")
    return df


def remittance_advices(ctx: Ctx, pending: pl.DataFrame) -> pl.DataFrame:
    ra = parse_phase(ctx.phase)
    if ra is None or ra.height == 0:
        return empty_rows()
    adv = ra.group_by("file").agg(pl.col("entity").first(), pl.col("received").first(), pl.col("ref").first(),
                                  pl.col("paid").sum().round(2).alias("sum_paid"), pl.col("total").first())
    adv = adv.with_columns(norm_expr("ref").alias("rnorm")).filter(pl.col("rnorm").str.len_chars() >= 6)
    lines = pending.filter(pl.col("is_credit")).join(_acc_entity(ctx), on="account_id").select(
        "bank_txn_id", "booking_date", "currency", "abs_amt", "orig_amt", pl.col("acc_entity").alias("entity"),
        pl.col("alltext").str.extract_all(r"[A-Z0-9][A-Z0-9\-/]{5,}").alias("tok")).explode("tok").filter(pl.col("tok").is_not_null()).with_columns(
        norm_expr("tok").alias("rnorm"))
    cand = lines.join(adv, on=["entity", "rnorm"]).with_columns(
        (pl.col("received").str.to_date() - pl.col("booking_date").str.to_date()).dt.total_days().alias("lag"),
        pl.when(pl.col("orig_amt").is_not_null()).then(pl.col("orig_amt").abs()).otherwise(pl.col("abs_amt")).alias("cmp"))
    cand = cand.filter((pl.col("lag") >= -5) & (pl.col("lag") <= 30)).with_columns(
        ((pl.col("sum_paid") - pl.col("cmp")).abs() < 0.015).alias("sum_ok"))
    link = cand.sort(["sum_ok", "lag"], descending=[True, False]).unique("bank_txn_id", keep="first").unique("file", keep="first")
    link = link.filter(pl.col("sum_ok") | (pl.col("orig_amt").is_not_null()))
    docs = ra.join(link.select("file", "bank_txn_id", "currency"), on="file").with_columns(norm_expr("doc").alias("dn"),
                                                                                         pl.col("doc").str.replace_all(r"\D", "").str.strip_chars_start("0").alias("dd"))
    items = ctx.items.filter(pl.col("item_type").is_in(["ARI", "ARC"])).select(
        "item_id", "item_type", "entity", "docnorm", "open_amount", pl.col("currency").alias("item_ccy"),
        pl.col("docnorm").str.replace_all(r"\D", "").str.strip_chars_start("0").alias("idig"))
    exact = docs.join(items, left_on=["entity", "dn"], right_on=["entity", "docnorm"])
    rest = docs.join(exact.select("file", "doc"), on=["file", "doc"], how="anti").filter(pl.col("dd").str.len_chars() >= 5)
    # documents without our prefix / leading zeros: match the digit tail within the entity, amount-checked
    tail = rest.join(items.with_columns(pl.col("idig").str.slice(-7).alias("k")), left_on=["entity", pl.col("dd").str.slice(-7)],
                     right_on=["entity", "k"]).filter(pl.col("idig").str.ends_with(pl.col("dd")) & (
                         (pl.col("gross").abs() - pl.col("open_amount")).abs() < 0.015)).unique(["file", "doc"])
    cols = ["bank_txn_id", "currency", "item_id", "item_type", "open_amount", "item_ccy", "gross", "deduction", "reason", "paid"]
    m = pl.concat([exact.select(cols), tail.select(cols)]).unique(["bank_txn_id", "item_id"])
    has_ded = pl.col("deduction").is_not_null() & (pl.col("deduction").abs() > 0.005)
    disc = has_ded & pl.col("reason").fill_null("").str.contains(DISCOUNT_RE.pattern, literal=False)
    fx = pl.col("currency") != pl.col("item_ccy")
    short_paid = (pl.col("paid").abs() < pl.col("open_amount") - 0.01)
    exc = (pl.when(pl.col("item_type") == "ARC").then(pl.lit(""))
           .when(disc).then(pl.when(fx).then(pl.lit("FX_DIFFERENCE")).otherwise(pl.lit("SHORT_DISCOUNT")))
           .when(has_ded).then(pl.lit("SHORT_DISPUTE"))
           .when(short_paid).then(pl.lit("PARTIAL"))
           .when(fx).then(pl.lit("FX_DIFFERENCE"))
           .otherwise(pl.lit("")))
    m = m.with_columns(exc.alias("exception"))
    m = m.with_columns(pl.when(pl.col("exception").is_in(["SHORT_DISPUTE", "PARTIAL"])).then(pl.col("paid").abs())
                       .when(pl.col("item_type") == "ARC").then(pl.col("paid").abs())
                       .otherwise(pl.col("open_amount")).alias("amount"))
    return mk_rows(m, "ra_advice", pl.col("exception"))


RESOLVERS = [remittance_advices]


if __name__ == "__main__":
    import sys
    for ph in sys.argv[1:] or ["live"]:
        d = parse_phase(ph, force=True)
        print(d.group_by("fmt").agg(pl.col("file").n_unique(), pl.len(), pl.col("deduction").is_not_null().sum().alias("ded")))
