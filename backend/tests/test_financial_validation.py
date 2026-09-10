from app.schemas.document import DocumentType, ExtractedField, ExtractionResult, ValidationStatus
from app.services.financial_validation import run_financial_validations


def _extraction(**field_values):
    return ExtractionResult(
        fields={name: ExtractedField(value=value) for name, value in field_values.items()}
    )


def test_balance_sheet_passes_when_consistent():
    extraction = _extraction(
        total_current_assets=100,
        total_non_current_assets=400,
        total_assets=500,
        total_current_liabilities=50,
        total_non_current_liabilities=150,
        total_liabilities=200,
        total_equity=300,
    )
    checks = run_financial_validations(DocumentType.BALANCE_SHEET, extraction)
    by_name = {c.check_name: c for c in checks}
    assert by_name["Total Assets = Current Assets + Non-Current Assets"].status == ValidationStatus.PASS
    assert by_name["Total Liabilities = Current Liabilities + Non-Current Liabilities"].status == ValidationStatus.PASS
    assert (
        by_name["Balance Sheet Equation: Total Assets = Total Liabilities + Total Equity"].status
        == ValidationStatus.PASS
    )


def test_balance_sheet_fails_when_inconsistent():
    extraction = _extraction(
        total_current_assets=100,
        total_non_current_assets=400,
        total_assets=999,  # wrong on purpose
    )
    checks = run_financial_validations(DocumentType.BALANCE_SHEET, extraction)
    by_name = {c.check_name: c for c in checks}
    assert by_name["Total Assets = Current Assets + Non-Current Assets"].status == ValidationStatus.FAIL
    assert by_name["Total Assets = Current Assets + Non-Current Assets"].variance == 499


def test_balance_sheet_not_applicable_when_fields_missing():
    extraction = _extraction(total_current_assets=100)  # non_current_assets & total missing
    checks = run_financial_validations(DocumentType.BALANCE_SHEET, extraction)
    by_name = {c.check_name: c for c in checks}
    assert by_name["Total Assets = Current Assets + Non-Current Assets"].status == ValidationStatus.NOT_APPLICABLE


def test_invoice_validation():
    extraction = _extraction(subtotal=100, tax_amount=10, total_amount=110, amount_paid=110, balance_due=0)
    checks = run_financial_validations(DocumentType.INVOICE, extraction)
    by_name = {c.check_name: c for c in checks}
    assert by_name["Total = Subtotal + Tax (+ Shipping - Discount)"].status == ValidationStatus.PASS
    assert by_name["Balance Due = Total - Amount Paid"].status == ValidationStatus.PASS


def test_profit_and_loss_validation_within_tolerance():
    # small rounding difference (well within 1% tolerance)
    extraction = _extraction(total_revenue=1000, cost_of_goods_sold=600, gross_profit=400.5)
    checks = run_financial_validations(DocumentType.PROFIT_AND_LOSS, extraction)
    by_name = {c.check_name: c for c in checks}
    assert by_name["Gross Profit = Revenue - Cost of Goods Sold"].status == ValidationStatus.PASS


def test_cash_flow_validation():
    extraction = _extraction(
        net_cash_from_operating_activities=500,
        net_cash_from_investing_activities=-200,
        net_cash_from_financing_activities=-100,
        net_increase_decrease_in_cash=200,
        cash_at_beginning_of_period=1000,
        cash_at_end_of_period=1200,
    )
    checks = run_financial_validations(DocumentType.CASH_FLOW, extraction)
    by_name = {c.check_name: c for c in checks}
    assert by_name["Net Change in Cash = Operating + Investing + Financing"].status == ValidationStatus.PASS
    assert by_name["Ending Cash = Beginning Cash + Net Change in Cash"].status == ValidationStatus.PASS
