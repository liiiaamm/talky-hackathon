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
