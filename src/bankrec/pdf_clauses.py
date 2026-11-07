"""Invoice / credit-note PDF clauses (POLICIES §3, §5.6-5.9): what the documents say will be paid, and how.

parse_phase(phase): one row per (PDF, clause) -> data/parquet/<phase>/pdf_clauses_v2.parquet
  file, entity, document_number, doc_type, kind, n, days, amounts, replaces, period
  kinds (invoice PDFs)      INST instalment plan (n, days, amounts) | RET retention [payable, retained] | ADV advance [net due]
                            REPL cancels-and-replaces <replaces> | MONTH service month <period YYYY-MM>
        (credit-note PDFs)  CN_PAYOUT  the credit note is paid out / refunded by bank transfer (not offset in a payment)
Clause templates INST/RET/ADV/REPL are EN/ES (checked on LIVE); wrapped lines are joined before matching. MONTH and CN_PAYOUT
read stated words only (month names, "refund"/"paid to your account"), see CONFIG.

Resolvers (run after remittance, before ar_text):
  replaced_invoices  payment citing a cancelled invoice X (exact amount) -> the replacement invoice, no exception (§5.7).
  pdf_clauses        planned payments (§5.8). A receipt whose cash *in the item currency* equals an amount the PDF says will be
                     paid (to ROUND_TOL: printed amounts and the cash can differ by a rounding cent), for an open invoice of a
                     customer evidenced in the text (doc number / name / IBAN). Instalment identity: the instalments still
                     open are the trailing ones adding up to the open amount at the window start; lines are assigned to them
                     minimising |booking - due date|, a stated "k/n" is binding. §3: instalment k < n, retention payable part,
                     advance-net payment -> PARTIAL; last instalment / retention release -> '' (FX_DIFFERENCE when the line
                     is converted from another currency). amount = cash in item currency; never more than the open amount.
  service_month      §5.6: a receipt naming a month ("SERVICE MAY 2025", "MANT. ENE 25") pays the open invoice whose PDF names that
                     service month, same customer, exact open amount. Abstains when ambiguous.
  credit_note_payouts §5.9: an ARC whose PDF says it is paid by transfer is settled by our outgoing transfer of exactly its amount
                     to that customer; an APC the supplier refunds by transfer is settled by its incoming transfer.
"""
from __future__ import annotations

import os
import re
import time
import unicodedata
from concurrent.futures import ProcessPoolExecutor

import polars as pl

from .data import DATA, PQ, Ctx, norm_expr
from .deterministic import _acc_entity, empty_rows, mk_rows

# ---------------------------------------------------------------- CONFIG (every constant used below)
ROUND_TOL = 0.011        # cash vs printed planned amount: one rounding cent (PDF prints total - round(total*pct); the bank gets round(total*(1-pct)))
KN_RX = re.compile(r"\b(\d{1,2})\s*/\s*(\d{1,2})\b")   # a stated instalment "k/n" in the bank text (counted only when n = the plan's n)
INST_ALONE = os.environ.get("BANKREC_PDF_INST_ALONE", "0") == "1"   # off: an instalment amount alone (even unique) hits ~140 wrong lines/window for ~4 right
MIN_KEY = 5            # name keys shorter than this are not evidence
CACHE = "pdf_clauses_v2.parquet"
# month vocabulary (stated words only; accents stripped, upper-case). Full names may stand alone; 3-letter forms need a year.
MONTHS = {1: "JANUARY ENERO JANVIER JANUAR GENNAIO JANEIRO JANUARI STYCZEN|JAN ENE",
          2: "FEBRUARY FEBRERO FEVRIER FEBRUAR FEBBRAIO FEVEREIRO FEBRUARI LUTY|FEB FEV",
          3: "MARCH MARZO MARS MARZ MAERZ MARCO MAART MARZEC|MAR",
          4: "APRIL ABRIL AVRIL APRILE KWIECIEN|APR ABR AVR",
          5: "MAY MAYO MAI MAGGIO MAIO MEI MAJ|",
          6: "JUNE JUNIO JUIN JUNI GIUGNO JUNHO CZERWIEC|JUN GIU",
          7: "JULY JULIO JUILLET JULI LUGLIO JULHO LIPIEC|JUL LUG",
          8: "AUGUST AGOSTO AOUT AUGUSTUS SIERPIEN|AUG AGO",
          9: "SEPTEMBER SEPTIEMBRE SETIEMBRE SEPTEMBRE SETTEMBRE SETEMBRO WRZESIEN|SEP SEPT SET",
          10: "OCTOBER OCTUBRE OCTOBRE OKTOBER OTTOBRE OUTUBRO PAZDZIERNIK|OCT OKT OTT OUT",
          11: "NOVEMBER NOVIEMBRE NOVEMBRE NOVEMBRO LISTOPAD|NOV",
          12: "DECEMBER DICIEMBRE DECEMBRE DEZEMBER DICEMBRE DEZEMBRO GRUDZIEN|DEC DIC DEZ"}
FULL = {w: m for m, s in MONTHS.items() for w in s.split("|")[0].split()}
ABBR = {w: m for m, s in MONTHS.items() for w in s.split("|")[1].split()}
# credit-note disposition (stated words): paid out / refunded by transfer vs offset in a later payment
CN_PAYOUT_RX = re.compile(r"paid to your bank|refunded|reembolsad|por transferencia|by bank transfer|virement|uberweisung|bonifico", re.I)
CN_OFFSET_RX = re.compile(r"offset|compensar|deducted|descontar|next payment|pr[oó]xim", re.I)

NUMS = r"\d[\d.,\s]*\d"
INST = re.compile(r"(?:Payment terms:\s*(\d+)\s+instalments at\s+([\d/]+)\s+days|Forma de pago:\s*(\d+)\s+vencimientos a\s+([\d/]+)\s+d[ií]as)\s*\(([^)]*)\)")
RET = re.compile(r"(?:Retention\s+(\d+)\s*%|Retenci[oó]n en garant[ií]a del\s+(\d+)\s*%)\s*\((" + NUMS + r")\s*[A-Z]{3}\).*?"
                 r"(?:Amount payable at due date|Importe a pagar al vencimiento):\s*(" + NUMS + r")\s*[A-Z]{3}")
ADV = re.compile(r"(?:Less advance received against proforma|A deducir anticipo recibido seg[uú]n proforma)\s+(PF[\w-]+)\s*(?::|de)\s*(" + NUMS +
                 r")\s*[A-Z]{3}\.?\s*(?:Net amount due|Importe neto a pagar)\s*(" + NUMS + r")")
REPL = re.compile(r"(?:cancels and replaces invoice|anula y sustituye a la factura)\s+([A-Z0-9][A-Z0-9\-/]+)")
PDF_MONTH = re.compile(r"\b(" + "|".join(sorted(FULL, key=len, reverse=True)) + r")\s+(\d{4})\b")


def _num(s: str) -> float | None:
    s = s.strip().replace(" ", "").replace(" ", "")
    if re.search(r",\d{2}$", s):
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def _ascii_upper(t: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", t) if not unicodedata.combining(c)).upper()


def _parse_one(args):
    path, f, ent, docnum, dtype = args
    try:
        from pypdf import PdfReader
        t = " ".join((p.extract_text() or "") for p in PdfReader(path).pages)
    except Exception:
        return []
    t = re.sub(r"\s+", " ", t)
    out = []
    if dtype == "invoice":
        m = INST.search(t)
        if m:
            n = int(m.group(1) or m.group(3))
            days = [int(x) for x in (m.group(2) or m.group(4)).split("/") if x]
            amts = [_num(re.sub(r"[A-Z]{3}", "", a)) for a in m.group(5).split(" / ")]
            out.append(dict(kind="INST", n=n, days=days, amounts=[a for a in amts if a is not None]))
        m = RET.search(t)
        if m:
            out.append(dict(kind="RET", amounts=[_num(m.group(4)), _num(m.group(3))]))   # [payable, retained]
        m = ADV.search(t)
        if m:
            out.append(dict(kind="ADV", amounts=[_num(m.group(3))]))
        m = REPL.search(t)
        if m:
            out.append(dict(kind="REPL", replaces=m.group(1)))
        periods = {(int(y), FULL[w]) for w, y in PDF_MONTH.findall(_ascii_upper(t))}
        if len(periods) == 1:            # abstain when the document names several months
            y, mo = periods.pop()
            out.append(dict(kind="MONTH", period=f"{y:04d}-{mo:02d}"))
    elif dtype in ("credit_note", "supplier_credit_note"):
        if CN_PAYOUT_RX.search(t) and not CN_OFFSET_RX.search(t):
            out.append(dict(kind="CN_PAYOUT"))
    return [dict(file=f, entity=ent, document_number=docnum, doc_type=dtype, kind=o["kind"], n=o.get("n"), days=o.get("days"),
                 amounts=o.get("amounts"), replaces=o.get("replaces"), period=o.get("period")) for o in out]


SCHEMA = {"file": pl.Utf8, "entity": pl.Utf8, "document_number": pl.Utf8, "doc_type": pl.Utf8, "kind": pl.Utf8, "n": pl.Int32,
          "days": pl.List(pl.Int32), "amounts": pl.List(pl.Float64), "replaces": pl.Utf8, "period": pl.Utf8}
DOC_TYPES = ("invoice", "credit_note", "supplier_credit_note")


def parse_phase(phase: str, force: bool = False, workers: int = 6) -> pl.DataFrame | None:
    cache = os.path.join(PQ, phase, CACHE)
    if os.path.exists(cache) and not force:
        return pl.read_parquet(cache)
    base = os.path.join(DATA, phase, "pdfs")
    idx = os.path.join(base, "index.csv")
    if not os.path.exists(idx):
        return None
    index = pl.read_csv(idx, infer_schema_length=0).filter(pl.col("document_type").is_in(DOC_TYPES))
    jobs = [(os.path.join(base, f), f, e, d, t) for f, e, d, t in index.select("file", "entity", "document_number", "document_type").iter_rows()
            if os.path.exists(os.path.join(base, f))]
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        rows = [r for rs in ex.map(_parse_one, jobs, chunksize=64) for r in rs]
    df = pl.DataFrame(rows, schema=SCHEMA) if rows else pl.DataFrame(schema=SCHEMA)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    df.write_parquet(cache)
    print(f"  {phase}: {len(jobs):,} PDFs -> {df.height:,} clauses {sorted(df.group_by('kind').len().rows())} ({time.time() - t0:.0f}s)")
    return df


# ---------------------------------------------------------------- shared helpers
PHASE_ORDER = ("sample", "train", "live", "final")


def _clauses_upto(phase: str) -> pl.DataFrame | None:
    """Clauses of this phase's PDFs and of every EARLIER phase (port of INT6 3722bd8): an invoice still open at the phase start
    keeps the clause printed on its PDF, which sits in the folder of the phase it was issued in. Never reads a later phase."""
    if phase not in PHASE_ORDER:
        return parse_phase(phase)
    parts = [x for x in (parse_phase(ph) for ph in PHASE_ORDER[1:PHASE_ORDER.index(phase) + 1] if ph != "sample" or phase == "sample")
             if x is not None and x.height]
    if not parts:
        return None
    return pl.concat(parts, how="vertical_relaxed").unique(["entity", "document_number", "kind"], keep="last", maintain_order=True)


def _clauses(ctx: Ctx) -> pl.DataFrame | None:
    c = ctx.__dict__.setdefault("_cache", {})
    if "pdf_clauses" not in c:
        cl = _clauses_upto(ctx.phase)
        c["pdf_clauses"] = None if cl is None or cl.height == 0 else cl.with_columns(norm_expr("document_number").alias("docnorm"))
    return c["pdf_clauses"]


def _credit_lines(ctx: Ctx, pending: pl.DataFrame) -> pl.DataFrame:
    """Pending receipts, once per currency view: the bank currency (abs_amt) and, when the bank shows an original amount in
    another currency, that currency (orig_amt). cash/cur = what was paid in that currency; fx = the line is converted."""
    base = pending.filter(pl.col("is_credit")).join(_acc_entity(ctx), on="account_id").select(
        "bank_txn_id", "booking_date", pl.col("acc_entity").alias("entity"), norm_expr("alltext").alias("tnorm"), "alltext",
        pl.col("counterparty_account").alias("cp_iban"), "currency", "abs_amt", "orig_amt", "orig_currency")
    same = base.filter(pl.col("orig_amt").is_null()).with_columns(pl.col("currency").alias("cur"), pl.col("abs_amt").alias("cash"),
                                                                 pl.lit(False).alias("fx"))
    conv = base.filter(pl.col("orig_amt").is_not_null() & pl.col("orig_currency").is_not_null()
                       & (pl.col("orig_currency") != pl.col("currency"))).with_columns(
        pl.col("orig_currency").alias("cur"), pl.col("orig_amt").abs().alias("cash"), pl.lit(True).alias("fx"))
    return pl.concat([same, conv]).drop("currency", "abs_amt", "orig_amt", "orig_currency").with_columns(
        (pl.col("cash") * 100).round(0).cast(pl.Int64).alias("cents"))


def _near(df: pl.DataFrame, col: str = "cents") -> pl.DataFrame:
    """explode a cents key to cents-1..cents+1 (ROUND_TOL)."""
    return df.with_columns(pl.concat_list(pl.col(col) - 1, pl.col(col), pl.col(col) + 1).alias(col)).explode(col)


def _evidence(c: pl.DataFrame) -> pl.DataFrame:
    """doc: our document number in the text; ev: doc, customer name/trade name/IBAN, or the last 6 digits of the number."""
    def has(k):
        return ((pl.col(k).str.len_chars() >= MIN_KEY) & pl.col("tnorm").str.contains(pl.col(k), literal=True)).fill_null(False)
    digits = pl.col("docnorm").str.replace_all(r"\D", "")
    return c.with_columns(
        ((pl.col("docnorm").str.len_chars() >= 6) & pl.col("tnorm").str.contains(pl.col("docnorm"), literal=True)).fill_null(False).alias("doc"),
        (has("k0") | has("k1") | has("k2") | (pl.col("cp_iban").is_not_null() & (pl.col("cp_iban") == pl.col("iban"))).fill_null(False)
         | ((digits.str.len_chars() >= 6) & pl.col("tnorm").str.contains(digits.str.slice(-6), literal=True)).fill_null(False)).alias("ev"))


def _open_ari(ctx: Ctx) -> pl.DataFrame:
    return ctx.items.filter((pl.col("item_type") == "ARI") & (pl.col("open_amount") > 0)).select(
        "item_id", "entity", "party_id", "currency", "docnorm", "issue_date", "total", "open_amount",
        norm_expr("party_name").str.slice(0, 10).alias("k0"))


# ---------------------------------------------------------------- planned payments (instalments, retention, advance)
def _expected(ctx: Ctx, cl: pl.DataFrame) -> pl.DataFrame:
    """clauses -> planned receipts: item_id, exp, kind, n, due (due date of instalment k; null otherwise)."""
    items = ctx.all_items.filter(pl.col("item_type") == "ARI").select("item_id", "entity", "docnorm", "issue_date")
    cl = cl.filter(pl.col("doc_type") == "invoice").join(items, on=["entity", "docnorm"])
    inst = cl.filter((pl.col("kind") == "INST") & (pl.col("amounts").list.len() == pl.col("days").list.len())).with_columns(
        pl.col("amounts").list.len().cast(pl.Int32).alias("n"), pl.int_ranges(1, pl.col("amounts").list.len() + 1).cast(pl.List(pl.Int32)).alias("k"),
        pl.col("amounts").alias("plan")).explode(["amounts", "days", "k"]).select(
        "item_id", pl.col("amounts").alias("exp"), pl.lit("INST").alias("kind"), "k", "n", "plan",
        (pl.col("issue_date").str.to_date() + pl.duration(days=pl.col("days"))).alias("due"))
    # retention [payable, retained] -> k = 1, 2 of 2; advance [net due] -> k = 1 of 2 (the advance part is never paid by bank)
    other = cl.filter(pl.col("kind").is_in(["RET", "ADV"])).with_columns(
        pl.int_ranges(1, pl.col("amounts").list.len() + 1).cast(pl.List(pl.Int32)).alias("k")).explode(["amounts", "k"]).select(
        "item_id", pl.col("amounts").alias("exp"), "kind", "k", pl.lit(2, pl.Int32).alias("n"), pl.lit(None, pl.List(pl.Float64)).alias("plan"),
        pl.lit(None, pl.Date).alias("due"))
    return pl.concat([inst, other]).filter(pl.col("exp").is_not_null()).with_columns(
        (pl.col("exp") * 100).round(0).cast(pl.Int64).alias("cents"))


def _assign_instalments(c: pl.DataFrame) -> pl.DataFrame:
    """Which instalment each line pays (k), per item: instalments still open at the window start are the trailing ones whose
    amounts add up to the open amount; lines are matched to them minimising |booking date - due date| (a stated "k/n" in the
    text is binding). 99.3 % right on fully observed TRAIN plans vs 95.2 % for chronological order."""
    import datetime as dt
    import itertools
    out = []
    for grp in c.partition_by("item_id", maintain_order=True):
        r0 = grp.row(0, named=True)
        plan, n, open_amt = r0["plan"], r0["n"], r0["open_amount"]
        issue = dt.date.fromisoformat(r0["issue_date"])
        dues = sorted(set(zip(grp["k"].to_list(), grp["due"].to_list())))
        due = dict(dues) if len(dues) else {}
        tail = [k for k in range(1, n + 1) if abs(sum(plan[k - 1:]) - open_amt) <= ROUND_TOL * n]
        allk = list(range(1, n + 1))
        best = None
        # first the trailing instalments still open; if the lines contradict that (stated k, amounts), all instalments
        for rem in ([list(range(tail[0], n + 1))] if tail else []) + [allk]:
            lines = grp.unique("bank_txn_id", keep="first").sort(["booking_date", "bank_txn_id"]).head(len(rem))
            bd = [dt.date.fromisoformat(x) for x in lines["booking_date"].to_list()]
            cash, stated = lines["cash"].to_list(), lines["stated"].to_list()
            for perm in itertools.permutations(rem, len(bd)):
                bad = sum(abs(a - plan[k - 1]) > ROUND_TOL or bool(s and k not in s) for k, a, s in zip(perm, cash, stated))
                cost = (bad, sum(abs((x - due.get(k, issue)).days) for x, k in zip(bd, perm)))
                if best is None or cost < best[0]:
                    best = (cost, perm, lines)
            if best is not None and best[0][0] == 0:
                break
        if best is None:
            continue
        perm, lines = best[1], best[2]
        ok = [abs(a - plan[k - 1]) <= ROUND_TOL and not (s and k not in s) for k, a, s in zip(perm, lines["cash"].to_list(), lines["stated"].to_list())]
        out.append(lines.with_columns(pl.Series("k", perm, dtype=pl.Int32)).filter(pl.Series(ok)))
    return pl.concat(out) if out else c.head(0)


def pdf_clauses(ctx: Ctx, pending: pl.DataFrame) -> pl.DataFrame:
    from .ar_exact import _names
    cl = _clauses(ctx)
    if cl is None:
        return empty_rows()
    exp = _expected(ctx, cl).join(_open_ari(ctx), on="item_id")
    exp = exp.filter(pl.col("exp") <= pl.col("open_amount") + ROUND_TOL)          # only what can still be paid
    lines = _near(_credit_lines(ctx, pending))
    c = lines.join(exp, left_on=["entity", "cur", "cents"], right_on=["entity", "currency", "cents"]).filter(
        pl.col("issue_date") <= pl.col("booking_date")).join(_names(ctx), on="party_id", how="left")
    # evidence: our number / customer name / IBAN; an instalment amount to the cent is specific enough on its own when it is
    # the line's only candidate invoice (never for retention / advance amounts: a release alone hits unrelated receipts)
    c = _evidence(c).with_columns(pl.col("item_id").n_unique().over("bank_txn_id").alias("_cands"))
    c = c.filter(pl.col("doc") | pl.col("ev") | ((pl.col("kind") == "INST") & (pl.col("_cands") == 1) & INST_ALONE))
    if c.height == 0:
        return empty_rows()
    # stated instalment "k/n" (e.g. "PLAZO 2/3", "RATA 1/3", "ECHEANCE 3/4") where n = the plan's number of instalments
    def stated(r):
        if r["kind"] != "INST":
            return None
        ks = sorted({int(a) for a, b in KN_RX.findall(r["alltext"] or "") if int(b) == r["n"] and 1 <= int(a) <= r["n"]})
        return ks or None
    c = c.with_columns(
        pl.struct("alltext", "kind", "n").map_elements(stated, return_dtype=pl.List(pl.Int64)).alias("stated"),
        (pl.col("booking_date").str.to_date() - pl.col("due")).dt.total_days().abs().fill_null(0).alias("lag"),
        (pl.col("cash") - pl.col("exp")).abs().alias("dev"))
    c = c.with_columns(pl.col("stated").is_not_null().alias("kn"))
    # one item per line: our number > stated k/n > exact cent > due date closest to the booking date > oldest invoice
    c = c.sort(["doc", "kn", "dev", "lag", "issue_date", "item_id"], descending=[True, True, False, False, False, False])
    item_of_line = c.unique("bank_txn_id", keep="first", maintain_order=True).select("bank_txn_id", "item_id", "kind")
    c = c.join(item_of_line, on=["bank_txn_id", "item_id", "kind"], how="semi")
    # instalment plans: which instalment each line pays; retention / advance: the matched plan entry (dev, then lag)
    inst = _assign_instalments(c.filter(pl.col("kind") == "INST").sort(["item_id", "booking_date", "bank_txn_id", "k"]))
    if inst.height:
        inst = c.join(inst.select("bank_txn_id", "item_id", "k"), on=["bank_txn_id", "item_id", "k"], how="semi")
    rest = c.filter(pl.col("kind") != "INST").sort(["dev", "lag"]).unique("bank_txn_id", keep="first", maintain_order=True)
    c = pl.concat([inst, rest]).unique("bank_txn_id", keep="first")
    # never apply more than the open amount at the window start (booking order)
    c = c.sort(["item_id", "booking_date", "bank_txn_id"]).with_columns(pl.col("cash").cum_sum().over("item_id").alias("cum"))
    c = c.filter(pl.col("cum") <= pl.col("open_amount") + ROUND_TOL * pl.int_range(1, pl.len() + 1).over("item_id"))
    # §3: every instalment but the last (by identity, even when an earlier one is paid late), the payable part of a retention
    # and the net payment of an invoice with an advance are PARTIAL; the last instalment / retention release clear the item.
    exc = (pl.when((pl.col("k") < pl.col("n")) | (pl.col("kind") == "ADV")).then(pl.lit("PARTIAL"))
           .when(pl.col("fx")).then(pl.lit("FX_DIFFERENCE")).otherwise(pl.lit("")))
    return mk_rows(c.with_columns(pl.col("cash").alias("amount")), "pdf_clause", exc)


# ---------------------------------------------------------------- service month (§5.6)
def _month_rx() -> str:
    full = "|".join(sorted(FULL, key=len, reverse=True))
    abbr = "|".join(sorted(ABBR, key=len, reverse=True))
    return rf"\b(?:({full})\b\.?\s*[-/]?\s*(\d{{1,4}})?|({abbr})\.?\s*[-/]?\s*(\d{{2}}|\d{{4}}))\b"


def _line_period(lines: pl.DataFrame) -> pl.DataFrame:
    """bank text naming exactly one month -> smonth (1-12) and syear (null when absent / truncated)."""
    rx = _month_rx()
    t = lines.with_columns(pl.col("alltext").str.normalize("NFKD").str.replace_all(r"\p{Mn}", "").str.to_uppercase().alias("_t"))
    t = t.with_columns(pl.col("_t").str.extract_all(rx).alias("_m")).filter(pl.col("_m").list.len() > 0).explode("_m").with_columns(
        pl.col("_m").str.extract(r"^([A-Z]+)").alias("_w"), pl.col("_m").str.extract(r"(\d+)$").alias("_y"))
    mmap = pl.DataFrame({"_w": list(FULL) + list(ABBR), "smonth": list(FULL.values()) + list(ABBR.values())},
                        schema={"_w": pl.Utf8, "smonth": pl.Int32})
    t = t.join(mmap, on="_w").with_columns(
        pl.when(pl.col("_y").str.len_chars() == 4).then(pl.col("_y").cast(pl.Int32))
        .when(pl.col("_y").str.len_chars() == 2).then(2000 + pl.col("_y").cast(pl.Int32)).otherwise(None).alias("syear"))
    one = t.group_by("bank_txn_id").agg(pl.col("smonth").n_unique().alias("_k"))
    t = t.join(one.filter(pl.col("_k") == 1), on="bank_txn_id", how="semi")
    return t.sort("syear", nulls_last=True).unique("bank_txn_id", keep="first").select("bank_txn_id", "smonth", "syear")


def service_month(ctx: Ctx, pending: pl.DataFrame) -> pl.DataFrame:
    from .ar_exact import _names
    cl = _clauses(ctx)
    if cl is None:
        return empty_rows()
    per = cl.filter(pl.col("kind") == "MONTH").select("entity", "docnorm", pl.col("period").str.slice(0, 4).cast(pl.Int32).alias("pyear"),
                                                      pl.col("period").str.slice(5, 2).cast(pl.Int32).alias("smonth"))
    items = _open_ari(ctx).join(per, on=["entity", "docnorm"]).with_columns((pl.col("open_amount") * 100).round(0).cast(pl.Int64).alias("cents"))
    lines = _credit_lines(ctx, pending)
    lines = lines.join(_line_period(lines.select("bank_txn_id", "alltext").unique("bank_txn_id")), on="bank_txn_id")
    c = _near(lines).join(items, left_on=["entity", "cur", "cents", "smonth"], right_on=["entity", "currency", "cents", "smonth"])
    bdate = pl.col("booking_date").str.to_date()
    c = c.filter((pl.col("issue_date") <= pl.col("booking_date"))
                 & (pl.col("syear").is_null() | (pl.col("syear") == pl.col("pyear")))
                 & ((pl.col("pyear") * 12 + pl.col("smonth")) <= bdate.dt.year() * 12 + bdate.dt.month()))
    c = _evidence(c.join(_names(ctx), on="party_id", how="left")).filter(pl.col("ev"))
    # without a stated year the most recent service month before the booking date; one item per line and line per item
    c = c.sort(["bank_txn_id", "pyear"], descending=[False, True]).with_columns(
        (pl.col("pyear") == pl.col("pyear").max().over("bank_txn_id")).alias("_top")).filter(pl.col("_top"))
    c = c.filter(pl.len().over("bank_txn_id") == 1).sort(["booking_date", "bank_txn_id"]).unique("item_id", keep="first", maintain_order=True)
    exc = pl.when(pl.col("fx")).then(pl.lit("FX_DIFFERENCE")).otherwise(pl.lit(""))
    return mk_rows(c.with_columns(pl.col("open_amount").alias("amount")), "pdf_month", exc)


# ---------------------------------------------------------------- credit notes paid out / refunded (§5.9)
def credit_note_payouts(ctx: Ctx, pending: pl.DataFrame) -> pl.DataFrame:
    """ARC paid out by our transfer (debit) / APC refunded by the supplier (credit): exact amount, party evidenced."""
    cl = _clauses(ctx)
    if cl is None:
        return empty_rows()
    pay = cl.filter(pl.col("kind") == "CN_PAYOUT").select("entity", "docnorm").unique()
    items = ctx.items.filter(pl.col("item_type").is_in(["ARC", "APC"]) & (pl.col("open_amount") > 0)).join(pay, on=["entity", "docnorm"]).select(
        "item_id", "item_type", "entity", "party_id", "currency", "docnorm", "issue_date", "open_amount",
        norm_expr("party_name").str.slice(0, 10).alias("k0"), (pl.col("open_amount") * 100).round(0).cast(pl.Int64).alias("cents"))
    if items.height == 0:
        return empty_rows()
    names = pl.concat([
        ctx.customers.select(pl.col("customer_id").alias("party_id"), "iban", norm_expr("name").str.slice(0, 10).alias("k1"),
                             norm_expr("trade_name").str.slice(0, 10).alias("k2")),
        ctx.suppliers.select(pl.col("supplier_id").alias("party_id"), *(["iban"] if "iban" in ctx.suppliers.columns else [pl.lit(None, pl.Utf8).alias("iban")]),
                             norm_expr("name").str.slice(0, 10).alias("k1"), pl.lit(None, pl.Utf8).alias("k2"))], how="vertical_relaxed").unique("party_id")
    lines = pending.filter(pl.col("orig_amt").is_null()).join(_acc_entity(ctx), on="account_id").select(
        "bank_txn_id", "booking_date", "currency", pl.col("acc_entity").alias("entity"), norm_expr("alltext").alias("tnorm"),
        pl.col("counterparty_account").alias("cp_iban"), pl.when(pl.col("is_credit")).then(pl.lit("APC")).otherwise(pl.lit("ARC")).alias("item_type"),
        (pl.col("abs_amt") * 100).round(0).cast(pl.Int64).alias("cents"))
    c = lines.join(items, on=["entity", "currency", "cents", "item_type"]).filter(pl.col("issue_date") <= pl.col("booking_date"))
    c = _evidence(c.join(names, on="party_id", how="left")).filter(pl.col("doc") | pl.col("ev"))
    c = c.sort(["doc", "issue_date"], descending=[True, False]).unique("bank_txn_id", keep="first", maintain_order=True)
    c = c.sort(["booking_date", "bank_txn_id"]).unique("item_id", keep="first", maintain_order=True)
    return mk_rows(c.with_columns(pl.col("open_amount").alias("amount")), "pdf_cn_payout")


# ---------------------------------------------------------------- cancelled and replaced invoices (§5.7)
def replaced_invoices(ctx: Ctx, pending: pl.DataFrame) -> pl.DataFrame:
    """A receipt citing a cancelled invoice X settles the replacement invoice (exact amount)."""
    cl = _clauses(ctx)
    if cl is None:
        return empty_rows()
    rp = cl.filter(pl.col("kind") == "REPL").select("entity", "docnorm", norm_expr("replaces").alias("old"))
    items = ctx.items.filter(pl.col("item_type") == "ARI").select("item_id", "entity", "docnorm", "currency", pl.col("open_amount").round(2).alias("amt"))
    rp = rp.join(items, on=["entity", "docnorm"])
    toks = pending.filter(pl.col("is_credit") & pl.col("orig_amt").is_null()).join(_acc_entity(ctx), on="account_id").select(
        "bank_txn_id", "currency", pl.col("acc_entity").alias("entity"), pl.col("abs_amt").round(2).alias("amt"),
        pl.col("alltext").str.extract_all(r"[A-Z0-9][A-Z0-9\-/]{5,}").alias("t")).explode("t").filter(pl.col("t").is_not_null()).with_columns(
        norm_expr("t").alias("old"))
    c = toks.join(rp, on=["entity", "old", "currency", "amt"]).unique("bank_txn_id").unique("item_id")
    return mk_rows(c.with_columns(pl.col("amt").alias("amount")), "pdf_replaced")


RESOLVERS = [replaced_invoices, pdf_clauses, service_month, credit_note_payouts]

if __name__ == "__main__":
    import sys
    for ph in sys.argv[1:] or ["live"]:
        parse_phase(ph, force=True)
