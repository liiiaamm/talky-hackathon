"""Original receipt references and AR credit-note payouts."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import polars as pl
from bankrec.data import Ctx
from bankrec.deterministic import dd_returns,credit_note_payouts


class ARDeterministic(unittest.TestCase):
    def test_return_batch_reference_beats_newer_same_amount_receipt(self):
        bank=pl.DataFrame({'bank_txn_id':['B1'],'account_id':['ES01-BANK-EUR'],'booking_date':['2025-02-03'],
            'is_credit':[False],'abs_amt':[100.0],
            'alltext':['DEVOLUCION RECIBO REM 010125010101 MANDATO VTM-ES-0000001']})
        items=pl.DataFrame({'item_id':['ARI-A','ARI-B'],'item_type':['ARI','ARI'],'entity':['ES01']*2,
            'docnorm':['INV1','INV2'],'party_id':['C1']*2,'total':[100.0]*2,'issue_date':['2024-12-01','2025-01-01']})
        dd=pl.DataFrame({'batch_id':['010125010101','010125020101'],'account_id':['ES01-BANK-EUR']*2,
            'collection_date':['2025-01-01','2025-02-01'],'mandate_id':['VTM-ES-0000001']*2,
            'amount':['100.00']*2,'remittance_info':['INV1','INV2']})
        ctx=Ctx('train',bank,items,dd=dd,accounts=bank.select('account_id'),all_items=items,
            customers=pl.DataFrame({'customer_id':['C1'],'sepa_mandate_id':['VTM-ES-0000001']}))
        self.assertEqual(dd_returns(ctx,ctx.bank)['item_id'].to_list(),['ARI-A'])

    def payout_context(self):
        bank=pl.DataFrame({'bank_txn_id':['B1'],'account_id':['ES01-BANK-EUR'],'booking_date':['2025-02-03'],
            'is_credit':[False],'abs_amt':[100.0],'alltext':['RAPPEL Q4 ARC250001']})
        items=pl.DataFrame({'item_id':['ARC-A'],'item_type':['ARC'],'entity':['ES01'],
            'docnorm':['ARC250001'],'open_amount':[100.0],'issue_date':['2025-01-01']})
        return Ctx('train',bank,items,accounts=bank.select('account_id'),all_items=items)

    def test_explicit_credit_note_payout(self):
        ctx=self.payout_context()
        rows=credit_note_payouts(ctx,ctx.bank)
        self.assertEqual(rows['item_id'].to_list(),['ARC-A'])
        self.assertEqual(rows['exception'].to_list(),[''])

    def test_payout_requires_document_reference(self):
        ctx=self.payout_context();ctx.bank=ctx.bank.with_columns(pl.lit('CUSTOMER REFUND').alias('alltext'))
        self.assertEqual(credit_note_payouts(ctx,ctx.bank).height,0)

    def test_ambiguous_note_is_not_arbitrarily_selected(self):
        ctx=self.payout_context();ctx.items=pl.concat([ctx.items,ctx.items.with_columns(pl.lit('ARC-B').alias('item_id'))])
        self.assertEqual(credit_note_payouts(ctx,ctx.bank).height,0)

if __name__=='__main__': unittest.main()
