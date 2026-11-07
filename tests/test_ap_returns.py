"""AP identity and lost-reference return regressions."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import polars as pl
from bankrec.data import Ctx
from bankrec.deterministic import ap_direct_debits,payment_returns


class APReturns(unittest.TestCase):
    def context(self):
        bank=pl.DataFrame({'bank_txn_id':['B1'],'account_id':['ES01-BANK-EUR'],'booking_date':['2025-02-01'],
            'is_credit':[True],'currency':['EUR'],'abs_amt':[100.0],'alltext':['DEVOLUCION TRANSFERENCIA AC01 PENA EMBALAJES'],
            'end_to_end_id':[None]},schema_overrides={'end_to_end_id':pl.String})
        items=pl.DataFrame({'item_id':['API-A'],'item_type':['API'],'entity':['ES01'],'docnorm':['INV1'],
            'party_name':['Peña Embalajes'],'currency':['EUR'],'open_amount':[100.0],'total':[100.0],
            'due_date':['2025-01-20'],'issue_date':['2025-01-01'],'payment_method':['dd']})
        pr=pl.DataFrame({'run_id':['PR010125010963'],'run_type':['transfers'],'execution_date':['2025-01-09'],
            'debtor_account_id':['ES01-BANK-EUR'],'end_to_end_id':['PR010125010963-0001'],
            'beneficiary':['Peña Embalajes'],'currency':['EUR'],'transfer_amount':['100.00'],
            'document_number':['INV1'],'document_type':['INVOICE'],'document_amount':['100.00']})
        return Ctx('train',bank,items,payruns=pr,accounts=bank.select('account_id'),all_items=items)

    def test_return_without_e2e_uses_unique_transfer(self):
        ctx=self.context(); rows=payment_returns(ctx,ctx.bank)
        self.assertEqual(rows['item_id'].to_list(),['API-A'])
        self.assertEqual(rows['exception'].to_list(),['RETURNED'])

    def test_ambiguous_return_does_not_pick_arbitrary_transfer(self):
        ctx=self.context()
        ctx.payruns=pl.concat([ctx.payruns,ctx.payruns.with_columns(pl.lit('PR010125010963-0002').alias('end_to_end_id'))])
        self.assertEqual(payment_returns(ctx,ctx.bank).height,0)

    def test_same_numeric_amount_in_another_currency_is_rejected(self):
        ctx=self.context();ctx.payruns=ctx.payruns.with_columns(pl.lit('USD').alias('currency'))
        self.assertEqual(payment_returns(ctx,ctx.bank).height,0)

    def test_future_transfer_is_not_a_return_candidate(self):
        ctx=self.context();ctx.payruns=ctx.payruns.with_columns(pl.lit('2025-02-03').alias('execution_date'))
        self.assertEqual(payment_returns(ctx,ctx.bank).height,0)

    def test_supplier_accents_match_unaccented_bank_name(self):
        ctx=self.context();ctx.bank=ctx.bank.with_columns(pl.lit(False).alias('is_credit'),pl.lit('RECIBO PENA EMBALAJES').alias('alltext'))
        rows=ap_direct_debits(ctx,ctx.bank)
        self.assertEqual(rows['item_id'].to_list(),['API-A'])

if __name__=='__main__':unittest.main()
