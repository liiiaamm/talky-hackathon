"""Parse customer remittance advices (csv / xlsx / txt / pdf) into one flat table.

    cd src && python -m bankrec.remittance_parse --phase train      # -> data/parquet/<phase>/remittance_lines.parquet

One row per document line of an advice:
    file, entity, payer_name, received_date, fmt, ref, pay_date, currency, total,
    doc, doc_date, gross, deduction, reason, paid, voucher, po
`deduction` is positive (the amount the customer kept). Header-only fields repeat on every row.
Layouts are detected from content, not from file names, so new customers / the Sept-2025 layout change parse the same way.
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import re
from concurrent.futures import ProcessPoolExecutor

import polars as pl

from .data import DATA, PQ

NUM = r"-?\(?\d{1,3}(?:[.,\s]\d{3})*(?:[.,]\d{1,2})?\)?|-?\d+(?:[.,]\d{1,2})?"
DATE = r"\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2}"


def num(s) -> float | None:
    """Parse 1.234,56 / 1,234.56 / 1234.56 / (12.00) / -12,5 into a float."""
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).strip().replace(" ", "").replace(" ", "")
    if not s:
        return None
    neg = s.startswith("-") or (s.startswith("(") and s.endswith(")"))
    s = s.strip("-()+")
    if "," in s and "." in s:
        dec = "," if s.rfind(",") > s.rfind(".") else "."
    elif "," in s:
        dec = "," if re.search(r",\d{1,2}$", s) else None
    elif "." in s:
        dec = "." if re.search(r"\.\d{1,2}$", s) else None
    else:
        dec = None
    if dec:
        ip, fp = s.rsplit(dec, 1)
        s = re.sub(r"[.,]", "", ip) + "." + fp
    else:
        s = re.sub(r"[.,]", "", s)
    try:
        v = float(s)
    except ValueError:
        return None
    return -v if neg else v


def iso(d) -> str | None:
    if d is None:
        return None
    if hasattr(d, "strftime"):
        return d.strftime("%Y-%m-%d")
    d = str(d).strip()
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})", d)
    if m:
        return m.group(0)
    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{2,4})$", d)
    if m:
        dd, mm, yy = (int(x) for x in m.groups())
        yy = yy + (2000 if yy < 100 else 0)
        if mm > 12 and dd <= 12:   # US order (MM/DD/YYYY) in English advices
            dd, mm = mm, dd
        if not (1 <= mm <= 12 and 1 <= dd <= 31):
            return None
        return f"{yy:04d}-{mm:02d}-{dd:02d}"
    return None


def _row(doc=None, doc_date=None, gross=None, deduction=None, reason=None, paid=None, voucher=None, po=None):
    ded = num(deduction)
    return dict(doc=(str(doc).strip() if doc not in (None, "") else None), doc_date=iso(doc_date), gross=num(gross),
                deduction=abs(ded) if ded else None, reason=(str(reason).strip().upper() or None) if reason else None,
                paid=num(paid), voucher=(str(voucher).strip() if voucher else None), po=(str(po).strip() if po else None))


# ---------------------------------------------------------------- per format
def parse_csv(text: str) -> tuple[dict, list[dict]]:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    delim = ";" if lines and lines[0].count(";") > lines[0].count(",") else ","
    rows = list(csv.reader(io.StringIO("\n".join(lines)), delimiter=delim))
    head, out, cols = {}, [], None
    for r in rows:
        low = [c.strip().lower() for c in r]
        if low and low[0] == "payment_ref":
            kv = dict(zip(low[0::2], r[1::2]))
            head = dict(ref=kv.get("payment_ref"), pay_date=iso(kv.get("payment_date")), currency=kv.get("currency"))
        elif "document" in low or "invoice" in low:
            cols = low
        elif cols and r and r[0].strip().upper() not in ("TOTAL", ""):
            d = dict(zip(cols, r))
            out.append(_row(d.get("document") or d.get("invoice"), d.get("doc_date") or d.get("date"), d.get("gross") or d.get("amount"),
                            d.get("discount_or_deduction") or d.get("deduction"), d.get("reason"), d.get("paid") or d.get("net paid"),
                            d.get("voucher"), d.get("your_order") or d.get("po")))
    return head, out


def parse_xlsx(path: str) -> tuple[dict, list[dict]]:
    import openpyxl
    ws = openpyxl.load_workbook(path, read_only=True, data_only=True).active
    rows = [list(r) for r in ws.iter_rows(values_only=True)]
    head, out, cols = {}, [], None
    for r in rows:
        cells = [c for c in r if c is not None]
        if not cells:
            continue
        first = str(cells[0]).strip().lower()
        if first.startswith("payment advice") and len(cells) >= 2:
            head = dict(ref=str(cells[1]), pay_date=iso(cells[2]) if len(cells) > 2 else None)
        elif first in ("document", "voucher", "invoice"):
            cols = [str(c).strip().lower() if c is not None else "" for c in r]
        elif cols:
            d = dict(zip(cols, r))
            if str(d.get(cols[0]) or "").strip().upper() == "TOTAL":
                continue
            out.append(_row(d.get("document") or d.get("invoice"), d.get("date") or d.get("inv. date"), d.get("gross") or d.get("amount"),
                            d.get("deduction"), d.get("reason"), d.get("paid") or d.get("net paid"), d.get("voucher"), d.get("po")))
    return head, out


TXT_TOTAL = re.compile(r"(?:payment of|importe de|montant de|betrag von|pagamento di)\s+(" + NUM + r")\s*([A-Z]{3})?(?:.*?(?:on|con fecha|le|am|del)\s+(" + DATE + "))?", re.I)
TXT_ROW = re.compile(r"^\s+(\S+)\s+(" + DATE + r")\s+(" + NUM + r")(?:\s+(" + NUM + r")\s+([A-Za-z][A-Za-z .\-/]*?))?(?:\s+(" + NUM + r"))?\s*$")


def parse_txt(text: str) -> tuple[dict, list[dict]]:
    head, out = {}, []
    m = re.search(r"^Subject:.*?(?:advice|pago|paiement|Zahlungsavis|pagamento)\s+(\S+)", text, re.M | re.I)
    if m:
        head["ref"] = m.group(1)
    m = TXT_TOTAL.search(text.replace("\n", " "))
    if m:
        head.update(total=num(m.group(1)), currency=m.group(2), pay_date=iso(m.group(3)) if m.group(3) else None)
    for ln in text.splitlines():
        r = TXT_ROW.match(ln)
        if r:
            doc, d, g, ded, reason, paid = r.groups()
            out.append(_row(doc, d, g, ded, reason, paid if paid else (num(g) - abs(num(ded)) if ded else g)))
    return head, out


PDF_LABELS = {"payment ref.": "ref", "ref.": "ref", "date": "pay_date", "fecha": "pay_date", "amount": "total", "importe": "total"}
PDF_TABLE_END = {"paid", "pagado", "payé", "bezahlt"}
DED_TOKEN = re.compile(r"^(" + NUM + r")\s+([A-Za-zÁÉÍÓÚÑÜ][\w .\-/]*)$")


def parse_pdf_text(text: str) -> tuple[dict, list[dict]]:
    toks = [t.strip() for t in text.splitlines() if t.strip()]
    head, out = {}, []
    i = 0
    while i < len(toks) and toks[i].lower() not in PDF_TABLE_END:   # header block: label line, then value line
        key = PDF_LABELS.get(toks[i].lower())
        if key and i + 1 < len(toks) and key not in head:
            v = toks[i + 1]
            if key == "total":
                m = re.match(r"^(" + NUM + r")\s*([A-Z]{3})?$", v)
                if m:
                    head["total"], head["currency"] = num(m.group(1)), m.group(2)
            elif key == "pay_date":
                head[key] = iso(v)
            else:
                head[key] = v
            i += 2
            continue
        i += 1
    i += 1
    is_num = lambda t: re.fullmatch(NUM, t) is not None  # noqa: E731
    while i + 3 < len(toks) + 1:
        if i + 2 >= len(toks):
            break
        doc, d = toks[i], toks[i + 1]
        if not re.fullmatch(DATE, d):
            i += 1
            continue
        gross = toks[i + 2]
        nxt = toks[i + 3] if i + 3 < len(toks) else ""
        m = DED_TOKEN.match(nxt)
        if m and i + 4 < len(toks) and is_num(toks[i + 4]):
            out.append(_row(doc, d, gross, m.group(1), m.group(2), toks[i + 4]))
            i += 5
        elif is_num(nxt):
            out.append(_row(doc, d, gross, None, None, nxt))
            i += 4
        else:
            i += 1
    return head, out


def parse_file(args) -> list[dict]:
    path, meta = args
    fmt = meta["format"]
    try:
        if fmt == "csv":
            head, rows = parse_csv(open(path, encoding="utf-8-sig", errors="replace").read())
        elif fmt == "txt":
            head, rows = parse_txt(open(path, encoding="utf-8", errors="replace").read())
        elif fmt == "xlsx":
            head, rows = parse_xlsx(path)
        elif fmt == "pdf":
            import pypdf
            head, rows = parse_pdf_text("\n".join((p.extract_text() or "") for p in pypdf.PdfReader(path).pages))
        else:
            return []
    except Exception as e:  # a broken advice must never stop the pipeline
        return [dict(meta, fmt=fmt, error=f"{type(e).__name__}: {e}")]
    if head.get("total") is None and rows:
        head["total"] = round(sum((r["paid"] if r["paid"] is not None else (r["gross"] or 0)) for r in rows), 2)
    base = dict(file=meta["file"], entity=meta["entity"], payer_name=meta["payer_name"], received_date=meta["received_date"], fmt=fmt,
                ref=head.get("ref"), pay_date=head.get("pay_date"), currency=head.get("currency"), total=head.get("total"))
    return [dict(base, **r) for r in rows] or [dict(base, error="no rows")]


SCHEMA = dict(file=pl.Utf8, entity=pl.Utf8, payer_name=pl.Utf8, received_date=pl.Utf8, fmt=pl.Utf8, ref=pl.Utf8, pay_date=pl.Utf8,
              currency=pl.Utf8, total=pl.Float64, doc=pl.Utf8, doc_date=pl.Utf8, gross=pl.Float64, deduction=pl.Float64,
              reason=pl.Utf8, paid=pl.Float64, voucher=pl.Utf8, po=pl.Utf8, error=pl.Utf8)


def parse_phase(phase: str, workers: int | None = None) -> pl.DataFrame:
    root = os.path.join(DATA, phase, "remittance_advices")
    idx = list(csv.DictReader(open(os.path.join(root, "index.csv"), encoding="utf-8")))
    jobs = [(os.path.join(root, r["file"]), r) for r in idx]
    with ProcessPoolExecutor(max_workers=workers or os.cpu_count()) as ex:
        rows = [r for rs in ex.map(parse_file, jobs, chunksize=64) for r in rs]
    return pl.DataFrame([{k: r.get(k) for k in SCHEMA} for r in rows], schema=SCHEMA)


def load_lines(phase: str) -> pl.DataFrame | None:
    """Cached parsed advices for a phase (parses and caches on first use)."""
    p = os.path.join(PQ, phase, "remittance_lines.parquet")
    if os.path.exists(p):
        return pl.read_parquet(p)
    if not os.path.isdir(os.path.join(DATA, phase, "remittance_advices")):
        return None
    df = parse_phase(phase)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    df.write_parquet(p)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="train")
    a = ap.parse_args()
    p = os.path.join(PQ, a.phase, "remittance_lines.parquet")
    if os.path.exists(p):
        os.remove(p)
    df = load_lines(a.phase)
    ok = df.filter(pl.col("error").is_null())
    print(f"{a.phase}: {df['file'].n_unique():,} advices, {ok.height:,} document lines, "
          f"{df.filter(pl.col('error').is_not_null())['file'].n_unique():,} files with errors")
    print(ok.group_by("fmt").agg(pl.col("file").n_unique().alias("files"), pl.len().alias("lines"),
                                 pl.col("deduction").is_not_null().sum().alias("with_deduction")).sort("fmt"))
    print(ok.filter(pl.col("reason").is_not_null()).group_by("reason").len().sort("len", descending=True).head(25))
    err = df.filter(pl.col("error").is_not_null())
    if err.height:
        print(err.group_by("fmt", "error").len().sort("len", descending=True).head(10))


if __name__ == "__main__":
    main()
