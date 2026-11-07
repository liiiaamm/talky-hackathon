"""Attribute every missing allocation pair from an existing snapshot, without loading ctx.

python eda/11_pair_recall.py --sub /path/snap_h1_sub.parquet --gold /path/snap_h1_gold.parquet --out-dir out/recall_h1
Ownership is routing advice by segment, not evidence that a particular resolver can recover the pair.
"""
import argparse
import json
from pathlib import Path

import polars as pl

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--sub', required=True)
p.add_argument('--gold', required=True)
p.add_argument('--out-dir', required=True)
a = p.parse_args()
out = Path(a.out_dir)
out.mkdir(parents=True, exist_ok=True)
s, g = pl.read_parquet(a.sub), pl.read_parquet(a.gold)
s = s.with_columns(pl.col('item_id').fill_null(''), pl.col('amount').cast(pl.Float64, strict=False))
g = g.with_columns(pl.col('item_id').fill_null(''), pl.col('amount').cast(pl.Float64, strict=False))
keys = ['bank_txn_id', 'item_id']
si, gi = s.filter(pl.col('item_id') != ''), g.filter(pl.col('item_id') != '')
txn = s.group_by('bank_txn_id').agg(pl.col('module').first(), (pl.col('item_id') != '').sum().alias('proposed_items'), pl.col('category').drop_nulls().first().alias('proposed_category'))
sizes = gi.group_by('bank_txn_id').len(name='gold_items').with_columns(pl.when(pl.col('gold_items') == 1).then(pl.lit('1')).when(pl.col('gold_items') <= 5).then(pl.lit('2-5')).when(pl.col('gold_items') <= 20).then(pl.lit('6-20')).when(pl.col('gold_items') <= 100).then(pl.lit('21-100')).otherwise(pl.lit('>100')).alias('size_bucket'))
matched = gi.join(si.select(*keys, pl.col('amount').alias('proposed_amount')), on=keys, how='left').join(txn, on='bank_txn_id', how='left').join(sizes, on='bank_txn_id', how='left')
matched = matched.with_columns((pl.col('proposed_amount').is_not_null() & ((pl.col('amount') - pl.col('proposed_amount')).abs() <= pl.max_horizontal(pl.lit(.01), pl.col('amount').abs() * .005) + 1e-9)).alias('tp'))
matched = matched.with_columns(pl.when(pl.col('module').is_null()).then(pl.lit('bank_line_missing')).when(pl.col('proposed_items') == 0).then(pl.lit('category_instead_of_items')).when(pl.col('proposed_amount').is_not_null()).then(pl.lit('amount_outside_tolerance')).when(pl.col('proposed_items') < pl.col('gold_items')).then(pl.lit('too_few_or_wrong_items')).otherwise(pl.lit('wrong_items')).alias('cause'))
det_segments = ['ar_dd_batch','ar_dd_returned','ar_dd_represented_transfer','ar_card_settlement','ar_card_chargeback','ar_cheque_deposit','ar_cheque_returned','ar_credit_note_paid_out']
matched = matched.with_columns(pl.when(pl.col('segment').str.starts_with('ap_') | pl.col('segment').is_in(det_segments)).then(pl.lit('A')).when(pl.col('segment').str.contains('remittance')).then(pl.lit('C')).otherwise(pl.lit('B/C')).alias('owner'))
miss = matched.filter(~pl.col('tp'))
miss.write_parquet(out/'missing_pairs.parquet')
miss.group_by('owner','cause').len(name='missing_pairs').sort('missing_pairs',descending=True).write_csv(out/'causes.csv')
miss.group_by('owner','segment','cause','module').len(name='missing_pairs').sort('missing_pairs',descending=True).write_csv(out/'cause_segments.csv')
# Precision denominator includes all submitted pairs, attributed by the golden bank segment.
segments = g.group_by('bank_txn_id').agg(pl.col('segment').first())
submitted = si.join(segments, on='bank_txn_id', how='left').join(sizes.select('bank_txn_id','size_bucket'), on='bank_txn_id', how='left')
for col in ['segment','module','size_bucket']:
    recall = matched.group_by(col).agg(pl.len().alias('gold_pairs'),pl.col('tp').sum().alias('true_positive'))
    denom = submitted.group_by(col).len(name='submitted_pairs')
    table = recall.join(denom, on=col, how='full',coalesce=True).with_columns(pl.col('gold_pairs','true_positive','submitted_pairs').fill_null(0)).with_columns((pl.col('true_positive')/pl.col('gold_pairs')).alias('recall'),(pl.col('true_positive')/pl.col('submitted_pairs')).alias('precision')).sort('gold_pairs',descending=True)
    table.write_csv(out/f'by_{col}.csv')
top = miss.group_by('bank_txn_id','segment','module','owner','cause').agg(pl.len().alias('missing_pairs'),pl.col('gold_items').first()).sort('missing_pairs',descending=True).head(30)
bank = Path(__file__).resolve().parents[1]/'data/parquet/train/bank_transactions.parquet'
if bank.exists():
    top = top.join(pl.scan_parquet(bank).select('bank_txn_id','text','ref1','ref2','end_to_end_id','counterparty_name').filter(pl.col('bank_txn_id').is_in(top['bank_txn_id'].to_list())).collect(), on='bank_txn_id', how='left')
top.write_csv(out/'top30.csv')
tp = matched['tp'].sum()
summary = dict(gold_pairs=gi.height,submitted_pairs=si.height,true_positive=tp,missing_pairs=miss.height,precision=tp/si.height,recall=tp/gi.height)
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
with pl.Config(tbl_rows=50,tbl_width_chars=150):
 print(pl.read_csv(out/'causes.csv'))
 print(pl.read_csv(out/'by_segment.csv').head(20))
 print(top.select('bank_txn_id','segment','module','owner','cause','missing_pairs','gold_items'))
