"""Bounded exact subset-sum over integer cents.

Items are tried in the given order (callers put the most plausible / oldest first), so the
first solution found is the preferred one. The search stops after `max_solutions` solutions
or `node_budget` nodes; `exhausted=False` means the search was cut off.
Values may be negative (credit notes) - pruning uses the reachable [min, max] interval.
"""
from __future__ import annotations


def find_subsets(values, target, max_size=60, node_budget=20000, max_solutions=2, required=()):
    """Return (solutions, exhausted). Each solution is a sorted tuple of indices into `values`.

    `required` indices are always included (e.g. items explicitly referenced in the text).
    """
    req = set(required)
    base = sum(values[i] for i in req)
    rest = [i for i in range(len(values)) if i not in req]
    vals = [values[i] for i in rest]
    n = len(vals)
    # suffix bounds
    suf_pos = [0] * (n + 1)
    suf_neg = [0] * (n + 1)
    for k in range(n - 1, -1, -1):
        v = vals[k]
        suf_pos[k] = suf_pos[k + 1] + (v if v > 0 else 0)
        suf_neg[k] = suf_neg[k + 1] + (v if v < 0 else 0)
    sols = []
    nodes = 0
    stack = [(0, base, ())]
    goal = target
    limit = max_size - len(req)
    exhausted = True
    while stack:
        k, s, chosen = stack.pop()
        nodes += 1
        if nodes > node_budget:
            exhausted = False
            break
        if s == goal and (chosen or req):
            sols.append(tuple(sorted(list(req) + [rest[j] for j in chosen])))
            if len(sols) >= max_solutions:
                exhausted = False
                break
            continue
        if k == n or len(chosen) >= limit:
            continue
        if not (s + suf_neg[k] <= goal <= s + suf_pos[k]):
            continue
        # push "skip" first so "take" (earlier = preferred item) is explored first (LIFO)
        stack.append((k + 1, s, chosen))
        stack.append((k + 1, s + vals[k], chosen + (k,)))
    return sols, exhausted


def mitm_all(vals, target, max_solutions=8):
    """All non-empty subsets (sorted index tuples) of `vals` (<= ~24 positive ints) with sum == target.

    Meet in the middle; stops at `max_solutions`. Used on short windows only, so the 2**(n/2) halves stay tiny."""
    n = len(vals)
    h = n // 2

    def halves(v, off):
        out = [(0, ())]
        for k, x in enumerate(v):
            out += [(s + x, t + (off + k,)) for s, t in out]
        return out

    right = {}
    for s, t in halves(vals[h:], h):
        right.setdefault(s, []).append(t)
    sols = []
    for s, t in halves(vals[:h], 0):
        for u in right.get(target - s, ()):
            if t or u:
                sols.append(t + u)
                if len(sols) >= max_solutions:
                    return sols
    return sols


def pair_solutions(vals, target, max_solutions=8):
    """Index pairs (i < j) with vals[i] + vals[j] == target (single items are not pairs)."""
    seen = {}
    sols = []
    for j, v in enumerate(vals):
        for i in seen.get(target - v, ()):
            sols.append((i, j))
            if len(sols) >= max_solutions:
                return sols
        seen.setdefault(v, []).append(j)
    return sols


def windowed_solutions(inv_vals, cn_vals, target, window=16, max_solutions=8):
    """Oldest-window search. `inv_vals` are positive open amounts oldest first; `cn_vals` positive credit-note amounts.

    Customers settle from the oldest open invoices with a few skipped, so the set is looked for inside a prefix of the
    invoices, optionally netted with ONE credit note (target + credit note). Returns [(width, cn_index|None, invoice
    indices)] sorted by width (index of the newest invoice used + 1)."""
    iv = inv_vals[:window]
    out = []
    for c in [None] + list(range(len(cn_vals))):
        t = target + (cn_vals[c] if c is not None else 0)
        for s in mitm_all(iv, t, max_solutions):
            out.append((max(s) + 1, c, s))
    out.sort(key=lambda x: (x[0], -1 if x[1] is None else x[1], x[2]))
    return out
