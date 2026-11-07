"""Structural regressions for mixed-currency aggregated supplier payments."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import polars as pl
from bankrec.data import Ctx
from bankrec.deterministic import payment_runs


class MixedCurrencyRun(unittest.TestCase):
    def context(self, count=2):
        bank = pl.DataFrame({'bank_txn_id':['B1'], 'account_id':['US01-BANK-USD'],
            'booking_date':['2025-01-09'], 'currency':['USD'], 'amt':[-208.0], 'abs_amt':[208.0],
            'is_credit':[False], 'orig_amt':[None], 'orig_currency':[None],
            'end_to_end_id':[None], 'alltext':[f'BULK PAYMENT PR010125010963 {count} TRANSFERS']},
            schema_overrides={'orig_amt':pl.Float64,'orig_currency':pl.String,'end_to_end_id':pl.String})
        items=pl.DataFrame({'item_id':['API-A','API-B'],'item_type':['API','API'], 'entity':['US01','US01'],
            'docnorm':['INV1','INV2'], 'party_name':['Supplier A','Supplier B'], 'currency':['EUR','USD'],
            'open_amount':[100.0,100.0], 'total':[100.0,100.0]})
        pr=pl.DataFrame({'run_id':['PR010125010963']*2,'run_type':['aggregated']*2,
            'execution_date':['2025-01-09']*2,'debtor_account_id':['US01-BANK-USD']*2,
            'end_to_end_id':['PR010125010963-0001','PR010125010963-0002'],
            'beneficiary':['Supplier A','Supplier B'],'currency':['EUR','USD'],
            'transfer_amount':['100.00']*2,'document_number':['INV1','INV2'],
            'document_type':['INVOICE']*2,'document_amount':['100.00']*2})
        return Ctx('train', bank, items, payruns=pr, accounts=bank.select('account_id'),all_items=items)

    def test_count_identifies_mixed_currency_aggregate(self):
        ctx=self.context()
        rows=payment_runs(ctx,ctx.bank).sort('item_id')
        self.assertEqual(rows['item_id'].to_list(),['API-A','API-B'])
        self.assertEqual(rows['amount'].to_list(),[100.,100.])
        self.assertEqual(rows['exception'].to_list(),['FX_DIFFERENCE',''])

    def test_wrong_transfer_count_does_not_bypass_amount_check(self):
        ctx=self.context(count=3)
        self.assertEqual(payment_runs(ctx,ctx.bank).height,0)

    def test_fx_uses_invoice_currency_even_when_transfer_currency_matches_bank(self):
        ctx=self.context()
        ctx.bank=ctx.bank.with_columns(pl.lit('PAY PR010125010963-0001').alias('alltext'),
            pl.lit(-110.0).alias('amt'),pl.lit(110.0).alias('abs_amt'))
        ctx.payruns=ctx.payruns.with_columns(pl.lit('USD').alias('currency'),
            pl.Series('transfer_amount',['110.00','100.00']))
        rows=payment_runs(ctx,ctx.bank)
        self.assertEqual(rows['item_id'].to_list(),['API-A'])
        self.assertEqual(rows['exception'].to_list(),['FX_DIFFERENCE'])

    def test_confirming_combines_weekend_maturities(self):
        ctx=self.context()
        ctx.bank=ctx.bank.with_columns(pl.lit('EUR').alias('currency'),pl.lit(-70.0).alias('amt'),
            pl.lit(70.0).alias('abs_amt'),pl.lit('2025-01-06').alias('booking_date'),
            pl.lit('PAGO CONFIRMING REMESA CF010125010963 2 PROVEEDORES').alias('alltext'))
        ctx.items=pl.concat([ctx.items,ctx.items.head(1).with_columns(pl.lit('API-C').alias('item_id'),
            pl.lit('INV3').alias('docnorm'),pl.lit('Supplier C').alias('party_name'))]).with_columns(
            pl.lit('EUR').alias('currency'),pl.Series('total',[30.,40.,100.]),pl.Series('open_amount',[30.,40.,100.]),
            pl.Series('due_date',['2025-01-04','2025-01-05','2025-01-07']))
        ctx.all_items=ctx.items
        ctx.payruns=pl.concat([ctx.payruns,ctx.payruns.head(1).with_columns(pl.lit('INV3').alias('document_number'),
            pl.lit('Supplier C').alias('beneficiary'))]).with_columns(pl.lit('CF010125010963').alias('run_id'),
            pl.lit('confirming').alias('run_type'),pl.lit('EUR').alias('currency'),pl.lit('2025-01-02').alias('execution_date'),
            pl.Series('end_to_end_id',['CF010125010963-0001','CF010125010963-0002','CF010125010963-0003']),
            pl.Series('transfer_amount',['30.00','40.00','100.00']),pl.Series('document_amount',['30.00','40.00','100.00']))
        self.assertEqual(payment_runs(ctx,ctx.bank).sort('item_id')['item_id'].to_list(),['API-A','API-B'])

    def test_duplicate_bank_debits_are_ambiguous(self):
        ctx=self.context()
        ctx.bank=pl.concat([ctx.bank,ctx.bank.with_columns(pl.lit('B2').alias('bank_txn_id'))])
        self.assertEqual(payment_runs(ctx,ctx.bank).height,0)

    def test_single_currency_total_mismatch_is_rejected(self):
        ctx=self.context()
        ctx.payruns=ctx.payruns.with_columns(pl.lit('USD').alias('currency'))
        self.assertEqual(payment_runs(ctx,ctx.bank).height,0)

    def test_truncated_reference_selects_first_debit_chronologically(self):
        ctx=self.context()
        ctx.payruns=ctx.payruns.with_columns(pl.lit('transfers').alias('run_type'),
            pl.Series('transfer_amount',['100.00','200.00']),
            pl.Series('document_amount',['100.00','200.00']))
        ctx.items=ctx.items.with_columns(pl.Series('open_amount',[100.,200.]),
            pl.Series('total',[100.,200.]))
        ctx.all_items=ctx.items
        late=ctx.bank.with_columns(pl.lit('LATE').alias('bank_txn_id'),
            pl.lit('2025-01-10').alias('booking_date'),pl.lit(-100.).alias('amt'),
            pl.lit(100.).alias('abs_amt'),pl.lit('PAGO PR010125010963').alias('alltext'))
        early=late.with_columns(pl.lit('EARLY').alias('bank_txn_id'),
            pl.lit('2025-01-09').alias('booking_date'))
        for first,second in [(late,early),(early,late)]:
            ctx.bank=pl.concat([first,second])
            for _ in range(5):
                rows=payment_runs(ctx,ctx.bank)
                self.assertEqual(rows['bank_txn_id'].to_list(),['EARLY'])
                self.assertEqual(rows['item_id'].to_list(),['API-A'])

if __name__=='__main__': unittest.main()
