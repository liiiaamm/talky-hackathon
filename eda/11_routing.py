"""A·6 routing analysis on the H1 snapshot (standalone proposals of every collection resolver on the same pool).

    python eda/11_routing.py [tag]      # reads out/snap_<tag>_alt.parquet, out/snap_<tag>_pool.parquet, out/snap_<tag>_proposal_all_categories.parquet
                                        # and the pipeline snapshot sub/gold (BANKREC_SNAP_DIR, default ../../out of the main copy)
Prints: per-resolver precision on the pool, perfect-routing ceiling, conflict matrix, observable signals, replay of fixed orders
and of simple arbiters (scored with the official maths through bankrec.score). Read-only, < 1 GB RAM.
"""
import itertools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import polars as pl  # noqa: E402

from bankrec.score import score  # noqa: E402

tag = sys.argv[1] if len(sys.argv) > 1 else "h1"
W = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
O = os.path.join(W, "out")
S = os.environ.get("BANKREC_SNAP_DIR", "/home/liame/claude_hackathon/out")
base = pl.read_parquet(f"{S}/snap_{tag}_sub.parquet")
gold = pl.read_parquet(f"{S}/snap_{tag}_gold.parquet")
pool = pl.read_parquet(f"{O}/snap_{tag}_pool.parquet")
alt = pl.read_parquet(f"{O}/snap_{tag}_alt.parquet")
cats = pl.read_parquet(f"{O}/snap_{tag}_proposal_all_categories.parquet").with_columns(pl.lit("all_categories").alias("resolver"))
alt = pl.concat([alt, cats], how="diagonal_relaxed")
fixed = base.join(pool, on="bank_txn_id", how="anti")          # deterministic + strong categories + AP: order-independent
GROUPS = {"remittance": ["remittances"], "pdf_clauses": ["replaced_invoices", "pdf_clauses"], "ar_text": ["ar_text"],
          "ar_exact": ["exact_amount", "adjusted_amount"], "ar_multi": ["multi_invoice"]}
RES = [r for g in GROUPS.values() for r in g]
CUR = ["remittance", "pdf_clauses", "ar_text", "ar_exact", "ar_multi"]


def sc(sub):
    r = score(sub, gold, verbose=False)
    return {k: r[k] for k in ["score", "f1", "precision", "recall", "txn_acc", "exc_macro"]}


# ---------------------------------------------------------------- per (line, resolver) correctness, score-compatible
g = gold.with_columns(pl.col("item_id").fill_null(""), pl.col("amount").cast(pl.Float64, strict=False),
                      pl.col("category").fill_null(""), pl.col("exception").fill_null(""))
gi = g.filter(pl.col("item_id") != "").select("bank_txn_id", "item_id", pl.col("amount").alias("ga"), pl.col("exception").alias("ge"))
gl = g.group_by("bank_txn_id").agg((pl.col("item_id") != "").sum().alias("g_n"),
                                    pl.col("category").filter(pl.col("item_id") == "").first().alias("g_cat"),
                                    pl.col("exception").filter(pl.col("item_id") == "").first().alias("g_exc"),
                                    pl.col("segment").first())
a = alt.with_columns(pl.col("item_id").fill_null(""), pl.col("category").fill_null(""), pl.col("exception").fill_null(""))
ai = a.filter(pl.col("item_id") != "").join(gi, on=["bank_txn_id", "item_id"], how="left").with_columns(
    (pl.col("ga").is_not_null() & ((pl.col("amount") - pl.col("ga")).abs() <= pl.max_horizontal(pl.lit(0.01), 0.005 * pl.col("ga").abs()) + 1e-9)).alias("tp"))
ai = ai.with_columns((pl.col("tp") & (pl.col("exception") == pl.col("ge"))).alias("okr"))
per_items = ai.group_by("bank_txn_id", "resolver").agg(
    pl.len().alias("n"), pl.col("tp").sum().alias("tp"), pl.col("okr").sum().alias("okr"),
    pl.col("item_id").sort().str.join("|").alias("key"), pl.col("exception").sort().str.join("|").alias("excs"),
    pl.col("amount").sum().alias("asum"), pl.col("module").first().alias("sub_module"))
per_cat = a.filter(pl.col("item_id") == "").group_by("bank_txn_id", "resolver").agg(
    pl.col("category").first().alias("cat"), pl.col("exception").first().alias("cexc"), pl.col("module").first().alias("sub_module"))
lr = (pl.concat([per_items, per_cat], how="diagonal_relaxed").group_by("bank_txn_id", "resolver").agg(pl.all().drop_nulls().first())
      .join(gl, on="bank_txn_id", how="left")
      .with_columns(pl.col("n").fill_null(0), pl.col("tp").fill_null(0), pl.col("okr").fill_null(0))
      .with_columns(pl.when(pl.col("g_n") > 0).then(pl.col("cat").is_null() & (pl.col("n") == pl.col("g_n")) & (pl.col("okr") == pl.col("g_n")))
                    .otherwise(pl.col("cat").is_not_null() & (pl.col("n") == 0) & (pl.col("cat") == pl.col("g_cat")) & (pl.col("cexc") == pl.col("g_exc").fill_null("")))
                    .alias("ok"),
                    pl.when(pl.col("n") > 0).then(pl.col("key")).otherwise(pl.lit("CAT:") + pl.col("cat").fill_null("") + "/" + pl.col("cexc").fill_null("")).alias("key")))
N = pool.height
print(f"pool lines {N:,}   gold pairs on pool {gi.join(pool, on='bank_txn_id', how='semi').height:,}")
print("\n== standalone proposals on the pool (line accuracy = official txn rule)")
print(lr.group_by("resolver").agg(pl.len().alias("lines"), pl.col("ok").mean().round(4).alias("line_acc"), pl.col("n").sum().alias("pairs"),
                                  pl.col("tp").sum().alias("tp")).with_columns((pl.col("tp") / pl.col("pairs")).round(4).alias("pair_prec"))
      .sort("lines", descending=True))

# ---------------------------------------------------------------- baseline on the pool and the ceiling
bpool = base.join(pool, on="bank_txn_id", how="semi")
bres = score(base, gold, verbose=False)
bt = bres["txn_table"].join(pool, on="bank_txn_id", how="semi")
print(f"\nbaseline (current order) on pool: line acc {bt['ok'].mean():.4f} ({int(bt['ok'].sum()):,} lines)  full score {bres['score']}")
any_ok = lr.group_by("bank_txn_id").agg(pl.col("ok").any().alias("any_ok"), pl.col("tp").max().alias("best_tp"))
ceil_lines = int(any_ok["any_ok"].sum())
print(f"perfect routing ceiling: {ceil_lines:,} lines correct ({ceil_lines / N:.4f}) vs baseline {int(bt['ok'].sum()):,}  ->  +{ceil_lines - int(bt['ok'].sum()):,} lines"
      f"  = +{0.3 * 100 * (ceil_lines - int(bt['ok'].sum())) / gl.height:.2f} pts TxnAcc term")
nres = lr.group_by("bank_txn_id").agg(pl.col("resolver").filter(pl.col("resolver") != "all_categories").n_unique().alias("nprop"),
                                       pl.col("key").filter(pl.col("resolver") != "all_categories").n_unique().alias("nkeys"),
                                       pl.col("ok").filter(pl.col("resolver") != "all_categories").any().alias("item_ok"))
print("\n== lines by number of item-resolvers proposing / distinct proposals")
print(nres.group_by("nprop", "nkeys").agg(pl.len().alias("lines"), pl.col("item_ok").mean().round(3).alias("some_resolver_ok")).sort("nprop", "nkeys"))


# ---------------------------------------------------------------- replay helper: pick one resolver per line
def assemble(choice: pl.DataFrame, name="x"):
    """choice: bank_txn_id, resolver (one row per line; missing -> all_categories)."""
    ch = pool.join(choice, on="bank_txn_id", how="left").with_columns(pl.col("resolver").fill_null("all_categories"))
    rows = alt.join(ch, on=["bank_txn_id", "resolver"], how="semi")
    return pl.concat([fixed, rows.drop("resolver")], how="diagonal_relaxed")


def order_choice(order_res: list[str]):
    rank = {r: i for i, r in enumerate(order_res)}
    return (lr.filter(pl.col("resolver").is_in(order_res)).with_columns(pl.col("resolver").replace_strict(rank, default=99).alias("rk"))
            .sort("rk").group_by("bank_txn_id").agg(pl.col("resolver").first()))


cur_res = [r for gname in CUR for r in GROUPS[gname]]
rep = sc(assemble(order_choice(cur_res)))
print(f"\nreplay of current order from standalone proposals: {rep}  (actual pipeline {bres['score']}) -> replay fidelity check")

# oracle: the correct proposal if any, else current choice
cur_ch = order_choice(cur_res)
orc = (lr.filter(pl.col("ok")).with_columns(pl.col("resolver").replace_strict({r: i for i, r in enumerate(cur_res + ['all_categories'])}).alias("rk"))
       .sort("rk").group_by("bank_txn_id").agg(pl.col("resolver").first()))
orc = pl.concat([orc, cur_ch.join(orc, on="bank_txn_id", how="anti")])
print("oracle routing (line-level):", sc(assemble(orc)))
# where does the oracle gain? (current pick wrong, other ok)
cw = cur_ch.join(lr.select("bank_txn_id", "resolver", "ok"), on=["bank_txn_id", "resolver"], how="left")
gain = cw.filter(~pl.col("ok")).join(orc, on="bank_txn_id", suffix="_orc").filter(pl.col("resolver") != pl.col("resolver_orc"))
print("\n== oracle gains: current pick (wrong) -> correct resolver")
print(gain.group_by("resolver", "resolver_orc").len().sort("len", descending=True).head(25))
lost_cat = cw.filter(~pl.col("ok")).join(lr.filter((pl.col("resolver") == "all_categories") & pl.col("ok")).select("bank_txn_id"), on="bank_txn_id", how="semi")
print(f"lines where the claimer is wrong but all_categories (=not claiming) would be right: {lost_cat.height:,}")
print(lost_cat.group_by("resolver").len().sort("len", descending=True))

# ---------------------------------------------------------------- fixed orders (groups permutations + intra-group splits)
print("\n== fixed orders (group permutations), full official score")
res = []
for perm in itertools.permutations(GROUPS):
    order_res = [r for gname in perm for r in GROUPS[gname]]
    m = sc(assemble(order_choice(order_res)))
    res.append((m["score"], ",".join(perm), m))
res.sort(reverse=True)
for s_, o_, m in res[:12]:
    print(f"  {s_:6.2f}  {o_:<55} F1 {m['f1']:.4f} P {m['precision']:.4f} R {m['recall']:.4f} acc {m['txn_acc']:.4f} exc {m['exc_macro']:.4f}")
print("  current:", [x for x in res if x[1] == ",".join(CUR)][0][:2])

# ---------------------------------------------------------------- conflicts: pairwise who is right
print("\n== pairwise conflicts (both propose, different keys): wins of A vs B")
w = lr.filter(pl.col("resolver") != "all_categories").select("bank_txn_id", "resolver", "key", "ok")
pw = w.join(w, on="bank_txn_id", suffix="_b").filter((pl.col("resolver") < pl.col("resolver_b")) & (pl.col("key") != pl.col("key_b")))
print(pw.group_by("resolver", "resolver_b").agg(pl.len().alias("conflicts"), pl.col("ok").sum().alias("A_ok"), pl.col("ok_b").sum().alias("B_ok"),
                                                (~pl.col("ok") & ~pl.col("ok_b")).sum().alias("none_ok")).sort("conflicts", descending=True))

# ---------------------------------------------------------------- signals
print("\n== signal: agreement (number of item-resolvers proposing the same item set)")
agree = w.group_by("bank_txn_id", "key").agg(pl.len().alias("votes"), pl.col("ok").first())
print(agree.group_by("votes").agg(pl.len().alias("proposals"), pl.col("ok").mean().round(4).alias("acc")).sort("votes"))
print("\n== signal: sub-module of the proposal (precision by line), conflicting lines only")
conf_ids = nres.filter(pl.col("nkeys") > 1).select("bank_txn_id")
print(lr.join(conf_ids, on="bank_txn_id", how="semi").group_by("resolver", "sub_module").agg(pl.len().alias("lines"), pl.col("ok").mean().round(3).alias("acc"))
      .sort("lines", descending=True).head(40))
lr.write_parquet(f"{O}/routing_{tag}_lines.parquet")
