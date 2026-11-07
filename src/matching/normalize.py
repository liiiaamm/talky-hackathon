"""Shared normalisation for party names and document numbers (Person B and Person C).

Everything here is deterministic and pure. Person C's PDF-clause table keyed by
(entity, document_number) must use `doc_keys(document_number).strong` as join key.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_NON_ALNUM = re.compile(r"[^A-Z0-9]+")
_TOKEN = re.compile(r"[A-Z0-9]+")
_DIGITS = re.compile(r"\d+")

# Legal-form and filler tokens removed from party names before token matching.
# Generic legal suffixes only (no customer-specific content).
LEGAL = frozenset("""
SL SLU SA SAU SAS SARL EURL SNC SCOOP COOP SC SCA SCP SRL SRLS SPA SNC SAPA GMBH CO KG EK AG OHG UG MBH BV NV VOF
LTD LIMITED PLC LLC LLP INC CORP CORPORATION COMPANY LDA LTDA ME EIRELI CV DE RL SP ZOO SPZOO PTY SARLAU SA
Y E ET UND AND THE LA LE LES EL DEL DER DIE DAS DU DES DA DO DOS DAS DI
""".split())


def ascii_upper(s: str) -> str:
    """Strip accents, uppercase. 'Ibáñez Riegos' -> 'IBANEZ RIEGOS'."""
    if not s:
        return ""
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii").upper()


def norm_text(s: str) -> str:
    """Accent-free uppercase with every non-alphanumeric run collapsed to one space."""
    return _NON_ALNUM.sub(" ", ascii_upper(s)).strip()


def compact(s: str) -> str:
    """Alphanumerics only: 'FV25-0108480' -> 'FV250108480'."""
    return _NON_ALNUM.sub("", ascii_upper(s))


def name_tokens(name: str) -> list[str]:
    """Distinctive tokens of a party name (legal forms and 1-char tokens dropped, order kept)."""
    toks = [t for t in _TOKEN.findall(ascii_upper(name)) if len(t) > 1 and t not in LEGAL]
    return toks or _TOKEN.findall(ascii_upper(name))


@dataclass(frozen=True)
class DocumentKeys:
    original: str
    strong: str          # compact form; collisions are rare (prefix + number)
    digits: str          # all digits, leading zeros stripped; medium strength
    tails: tuple         # numeric groups (>=5 digits) zero-stripped, e.g. '0108480' of 'FV25-0108480'; weak


def doc_keys(document_number: str) -> DocumentKeys:
    s = compact(document_number)
    digits = "".join(_DIGITS.findall(s)).lstrip("0")
    groups = _DIGITS.findall(ascii_upper(document_number))
    tails = tuple(sorted({g.lstrip("0") for g in groups if len(g) >= 5 and g.lstrip("0") != digits}))
    return DocumentKeys(document_number, s, digits if len(digits) >= 5 else "", tails)


def normalize_document_number(document_number: str) -> DocumentKeys:
    """Integration-contract alias of `doc_keys`."""
    return doc_keys(document_number)


_REF_TOKEN = re.compile(r"[A-Z0-9](?:[A-Z0-9]|[-/.](?=[A-Z0-9]))*")


def text_reference_tokens(blob: str) -> tuple[set, set]:
    """Candidate reference keys found in bank text.

    Returns (strong_keys, digit_keys): compact alnum tokens (len>=5, containing a digit) and
    zero-stripped digit runs (len>=5). Joined forms such as 'FV25-0108480' are kept whole.
    """
    up = ascii_upper(blob)
    strong, digit = set(), set()
    for tok in _REF_TOKEN.findall(up):
        c = _NON_ALNUM.sub("", tok)
        if len(c) >= 5 and any(ch.isdigit() for ch in c):
            strong.add(c)
    for run in _DIGITS.findall(up):
        r = run.lstrip("0")
        if len(r) >= 5:
            digit.add(r)
    return strong, digit


MONTHS = {
    # ES, EN, DE, FR, IT, PT (+ common abbreviations)
    1: "ENERO JANUARY JANUAR JANVIER GENNAIO JANEIRO ENE JAN GEN",
    2: "FEBRERO FEBRUARY FEBRUAR FEVRIER FEBBRAIO FEVEREIRO FEB FEV",
    3: "MARZO MARCH MAERZ MARZ MARS MARCO MAR",
    4: "ABRIL APRIL AVRIL APRILE ABR APR AVR",
    5: "MAYO MAY MAI MAGGIO MAIO MAG",
    6: "JUNIO JUNE JUNI JUIN GIUGNO JUNHO JUN GIU",
    7: "JULIO JULY JULI JUILLET LUGLIO JULHO JUL LUG",
    8: "AGOSTO AUGUST AOUT AGO AUG",
    9: "SEPTIEMBRE SEPTEMBER SEPTEMBRE SETTEMBRE SETEMBRO SEP SEPT SET",
    10: "OCTUBRE OCTOBER OKTOBER OCTOBRE OTTOBRE OUTUBRO OCT OKT OTT OUT",
    11: "NOVIEMBRE NOVEMBER NOVEMBRE NOVEMBRO NOV",
    12: "DICIEMBRE DECEMBER DEZEMBER DECEMBRE DICEMBRE DEZEMBRO DIC DEC DEZ",
}
_MONTH_WORD = {w: m for m, ws in MONTHS.items() for w in ws.split()}
_MONTH_RE = re.compile(r"\b([A-Z]{3,10})\.?\s*'?(\d{2}|\d{4})\b")


def service_month(blob: str):
    """('MANT. MAYO 25' | 'SERVICE APRIL 2025') -> (2025, 5) or None. Needs a month word + year."""
    for w, y in _MONTH_RE.findall(norm_text(blob)):
        m = _MONTH_WORD.get(w)
        if m:
            yy = int(y)
            return (2000 + yy if yy < 100 else yy, m)
    return None
