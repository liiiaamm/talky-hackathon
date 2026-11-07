"""Inference-only perturbation: unknown entity token and renamed bank payer strings."""
import argparse
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import joblib
import numpy as np
import polars as pl
from bankrec.categories import features
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out',default='out/drift_probe.json')
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
model=joblib.load(os.environ.get('BANKREC_CAT_MODEL',root/'models/categories_to2024.joblib'))
summary={}
for tag,lo,hi in [('h1','2025-01-01','2025-06-30'),('h2','2024-07-01','2024-12-31')]:
 bank=pl.scan_parquet(root/'data/parquet/train/bank_transactions.parquet').filter(pl.col('booking_date').is_between(pl.lit(lo),pl.lit(hi))).select('bank_txn_id','entity','amount','statement_format','bank_code','bank_subcode','text','ref1','ref2','end_to_end_id','remittance_info','counterparty_name').collect()
 bank=bank.with_columns(pl.col('amount').cast(pl.Float64).alias('amt')).with_columns(pl.col('amt').abs().alias('abs_amt'),pl.concat_str([pl.col(c).fill_null('') for c in ['text','ref1','ref2','end_to_end_id','remittance_info','counterparty_name']],separator=' ').str.to_uppercase().alias('alltext'))
 # Bounded 2000-row inference batches, always the same bank lines.
 counts={'n':bank.height,'unknown_entity_prediction_changed':0,'renamed_payer_prediction_changed':0,'unknown_entity_strong_claims_changed':0,'renamed_payer_strong_claims_changed':0}
 for batch in bank.iter_slices(2000):
  X=features(batch); base=model.predict_proba(X)
  unknown=model.predict_proba(features(batch.with_columns(pl.lit('ZA01').alias('entity'))))
  renamed=batch.with_columns(pl.col('counterparty_name').fill_null('').alias('_old'))
  # Literal per-row replacement affects the payer name wherever repeated in bank text; does not alter document references.
  texts=[]
  for old,text in renamed.select('_old','alltext').iter_rows():
   texts.append(text.replace(old.upper(),'RENAMED CUSTOMER') if old else text)
  rename=model.predict_proba(features(batch.with_columns(pl.Series('alltext',texts))))
  bp=base.argmax(1); up=unknown.argmax(1); rp=rename.argmax(1)
  classes=np.asarray(model.classes_)
  def claim(P):
   ix=P.argmax(1)
   return np.where((P.max(1)>=.97)&~np.isin(classes[ix],['ITEM','UNIDENTIFIED']),classes[ix],'')
  counts['unknown_entity_prediction_changed']+=int((bp!=up).sum())
  counts['renamed_payer_prediction_changed']+=int((bp!=rp).sum())
  counts['unknown_entity_strong_claims_changed']+=int((claim(base)!=claim(unknown)).sum())
  counts['renamed_payer_strong_claims_changed']+=int((claim(base)!=claim(rename)).sum())
 summary[tag]=counts
 print(tag,counts,flush=True)
Path(args.out).parent.mkdir(parents=True,exist_ok=True)
Path(args.out).write_text(json.dumps(summary,indent=2)+'\n')
