from datetime import date

from bankrec.ar_due_subset import _cascade, _unique_subset


def test_unique_subset_counts_and_returns_indices():
    n, sol = _unique_subset([100, 250, 400, 75], 350)
    assert n == 1 and sorted(sol) == [0, 1]
    n, sol = _unique_subset([100, 100, 250], 350)       # two equal invoices -> two subsets
    assert n == 2 and sol is None
    assert _unique_subset([100, 200], 999) == (0, None)


def test_cascade_takes_recently_due_items_and_abstains_when_ambiguous():
    bd = date(2025, 3, 31)
    its = [("I1", "ARI", 100, date(2025, 3, 20)), ("I2", "ARI", 250, date(2025, 4, 5)),
           ("OLD", "ARI", 250, date(2024, 6, 1)), ("C1", "ARC", -50, date(2025, 3, 25))]
    got = _cascade(its, bd, 300)                       # OLD is outside every window: I1 + I2 - C1 is the only solution
    assert sorted(r[0] for r in got) == ["C1", "I1", "I2"]
    its.append(("I3", "ARI", 300, date(2025, 3, 30)))  # a single invoice also matches but needs >= 2 items...
    its.append(("I4", "ARI", 200, date(2025, 3, 29)))  # ...and I4 + I1 is a second multi-item solution -> abstain
    assert _cascade(its, bd, 300) is None
