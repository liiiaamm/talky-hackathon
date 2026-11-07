"""Score a submission on a fold with the OFFICIAL scorer (starter/bankrec_score.py)."""
import csv, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(HERE, "..", "..", "bankrec-docs-and-masters")
sys.path.insert(0, os.path.join(DOCS, "starter"))
import bankrec_score as S  # noqa: E402

EXC = ["PARTIAL", "OVERPAYMENT", "SHORT_DISCOUNT", "SHORT_BANK_CHARGES", "SHORT_WITHHOLDING", "SHORT_DISPUTE",
       "DUPLICATE_PAYMENT", "UNIDENTIFIED", "RETURNED", "FX_DIFFERENCE"]


def score(fold, path):
    d = os.path.join(HERE, "..", "work", "folds", fold)
    cats = [r["category"] for r in csv.DictReader(open(os.path.join(DOCS, "masters", "categories.csv"), encoding="utf-8"))]
    g = S.load_golden(os.path.join(d, "golden.csv.gz"), os.path.join(d, "txns.csv.gz"), EXC, cats)
    return S.score_submission(g, open(path, "rb").read(), path)


if __name__ == "__main__":
    r = score(sys.argv[1], sys.argv[2])
    print(json.dumps({k: r[k] for k in ("ok", "score", "metrics", "errors")}, indent=1))
    for k, v in sorted(r["segments"].items(), key=lambda x: -x[1]["n"])[:30]:
        print(f"  {k:30s} n={v['n']:7d} acc={v['accuracy']:.3f}")
