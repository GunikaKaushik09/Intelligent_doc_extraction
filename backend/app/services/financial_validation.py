"""
Financial validation rules engine (spec 4.4).

Each rule:
  * Reads the specific fields it needs from the already-extracted data.
  * If ANY required field is missing (None), returns NOT_APPLICABLE --
    it never assumes, defaults or invents a missing value.
  * Otherwise computes the expected value from a formula, compares it to
    the reported value within a relative tolerance, and returns PASS/FAIL
    with the formula, inputs, calculated value, reported value and variance.
"""
from __future__ import annotations

from app.core.config import get_settings
from app.schemas.document import (
    DocumentType,
    ExtractionResult,
    FinancialValidationCheck,
    ValidationStatus,
)

settings = get_settings()


def _get(extraction: ExtractionResult, name: str) -> float | None:
    field = extraction.fields.get(name)
    if field is None or field.value is None:
        return None
    try:
        return float(field.value)
    except (TypeError, ValueError):
        return None


def _check(
    check_name: str,
    formula: str,
    inputs: dict[str, float | None],
    calculated: float | None,
    reported: float | None,
) -> FinancialValidationCheck:
    missing = [k for k, v in inputs.items() if v is None] + (["reported_value"] if reported is None else [])
    if missing or calculated is None:
        return FinancialValidationCheck(
            check_name=check_name,
            formula=formula,
            input_values={k: v for k, v in inputs.items()},
            calculated_value=calculated,
            reported_value=reported,
            variance=None,
            variance_pct=None,
            tolerance_pct=settings.VALIDATION_TOLERANCE_PCT,
            status=ValidationStatus.NOT_APPLICABLE,
            reason=f"Required field(s) not present in source document: {', '.join(missing) or 'reported value'}",
        )

    variance = round(reported - calculated, 2)
    denom = abs(calculated) if calculated != 0 else 1.0
    variance_pct = round(abs(variance) / denom, 6)
    passed = variance_pct <= settings.VALIDATION_TOLERANCE_PCT

    return FinancialValidationCheck(
        check_name=check_name,
        formula=formula,
        input_values=inputs,
        calculated_value=round(calculated, 2),
        reported_value=round(reported, 2),
        variance=variance,
        variance_pct=variance_pct,
        tolerance_pct=settings.VALIDATION_TOLERANCE_PCT,
        status=ValidationStatus.PASS if passed else ValidationStatus.FAIL,
        reason=None if passed else "Reported value differs from the calculated value beyond tolerance.",
    )


def _validate_balance_sheet(e: ExtractionResult) -> list[FinancialValidationCheck]:
    checks = []

    ca = _get(e, "total_current_assets")
    nca = _get(e, "total_non_current_assets")
    ta = _get(e, "total_assets")
    if ca is not None and nca is not None and ta is not None:
        checks.append(
            _check(
                "Total Assets = Current Assets + Non-Current Assets",
                "total_assets == total_current_assets + total_non_current_assets",
                {"total_current_assets": ca, "total_non_current_assets": nca},
                ca + nca,
                ta,
            )
        )
    else:
        checks.append(
            _check(
                "Total Assets = Current Assets + Non-Current Assets",
                "total_assets == total_current_assets + total_non_current_assets",
                {"total_current_assets": ca, "total_non_current_assets": nca},
                (ca + nca) if ca is not None and nca is not None else None,
                ta,
            )
        )

    cl = _get(e, "total_current_liabilities")
    ncl = _get(e, "total_non_current_liabilities")
    tl = _get(e, "total_liabilities")
    checks.append(
        _check(
            "Total Liabilities = Current Liabilities + Non-Current Liabilities",
            "total_liabilities == total_current_liabilities + total_non_current_liabilities",
            {"total_current_liabilities": cl, "total_non_current_liabilities": ncl},
            (cl + ncl) if cl is not None and ncl is not None else None,
            tl,
        )
    )

    # Many banking statements (including the supplied HDFC annual reports) do
    # not print a standalone "Total Equity" or "Total Liabilities" row. They
    # print "Total" under CAPITAL AND LIABILITIES and another "Total" under
    # ASSETS. Validate that directly rather than inventing component totals.
    tle = _get(e, "total_liabilities_and_equity")
    checks.append(
        _check(
            "Balance Sheet: Total Capital & Liabilities = Total Assets",
            "total_liabilities_and_equity == total_assets",
            {"total_liabilities_and_equity": tle},
            tle,
            ta,
        )
    )

    eq = _get(e, "total_equity")
    if tl is not None and eq is not None and ta is not None:
        checks.append(
            _check(
                "Balance Sheet Equation: Total Assets = Total Liabilities + Total Equity",
                "total_assets == total_liabilities + total_equity",
                {"total_liabilities": tl, "total_equity": eq},
                tl + eq,
                ta,
            )
        )
    else:
        checks.append(
            _check(
                "Balance Sheet Equation: Total Assets = Total Liabilities + Total Equity",
                "total_assets == total_liabilities + total_equity",
                {"total_liabilities": tl, "total_equity": eq},
                None,
                ta,
            )
        )
    return checks


def _validate_cash_flow(e: ExtractionResult) -> list[FinancialValidationCheck]:
    op = _get(e, "net_cash_from_operating_activities")
    inv = _get(e, "net_cash_from_investing_activities")
    fin = _get(e, "net_cash_from_financing_activities")
    net_change = _get(e, "net_increase_decrease_in_cash")
    fx = _get(e, "fx_translation_adjustment")
    begin = _get(e, "cash_at_beginning_of_period")
    end = _get(e, "cash_at_end_of_period")

    include_fx = fx is not None
    checks = [
        _check(
            "Net Change in Cash = Operating + Investing + Financing",
            "net_increase_decrease_in_cash == operating + investing + financing + fx_adjustment" if include_fx else "net_increase_decrease_in_cash == operating + investing + financing",
            {
                "net_cash_from_operating_activities": op,
                "net_cash_from_investing_activities": inv,
                "net_cash_from_financing_activities": fin,
                **({"fx_translation_adjustment": fx} if include_fx else {}),
            },
            (op + inv + fin + fx) if include_fx and None not in (op, inv, fin, fx) else ((op + inv + fin) if None not in (op, inv, fin) else None),
            net_change,
        ),
        _check(
            "Ending Cash = Beginning Cash + Net Change in Cash",
            "cash_at_end_of_period == cash_at_beginning_of_period + net_increase_decrease_in_cash",
            {"cash_at_beginning_of_period": begin, "net_increase_decrease_in_cash": net_change},
            (begin + net_change) if (begin is not None and net_change is not None) else None,
            end,
        ),
    ]
    return checks


def _validate_profit_and_loss(e: ExtractionResult) -> list[FinancialValidationCheck]:
    # The supplied P&L documents are banking statements. Use the case-study
    # formulas for the actual rows present in those statements rather than
    # forcing a manufacturing-style COGS/gross-profit model.
    interest_earned = _get(e, "interest_earned")
    other_income = _get(e, "other_income")
    total_income = _get(e, "total_income")
    interest_expended = _get(e, "interest_expended")
    opex = _get(e, "operating_expenses")
    provisions = _get(e, "provisions_and_contingencies")
    total_expenditure = _get(e, "total_expenditure")
    before_minority = _get(e, "profit_before_minority_interest")
    minority = _get(e, "minority_interest")
    net_profit = _get(e, "net_profit_attributable") or _get(e, "net_profit")

    checks = [
        # Keep the generic P&L checks when those fields are actually present.
        # They remain NOT_APPLICABLE for the supplied banking statements, which
        # do not expose COGS/gross-profit/tax rows.
        _check(
            "Gross Profit = Revenue - Cost of Goods Sold",
            "gross_profit == total_revenue - cost_of_goods_sold",
            {"total_revenue": _get(e, "total_revenue"), "cost_of_goods_sold": _get(e, "cost_of_goods_sold")},
            (_get(e, "total_revenue") - _get(e, "cost_of_goods_sold")) if None not in (_get(e, "total_revenue"), _get(e, "cost_of_goods_sold")) else None,
            _get(e, "gross_profit"),
        ),
        _check(
            "Operating Income = Gross Profit - Operating Expenses",
            "operating_income == gross_profit - operating_expenses",
            {"gross_profit": _get(e, "gross_profit"), "operating_expenses": opex},
            (_get(e, "gross_profit") - opex) if None not in (_get(e, "gross_profit"), opex) else None,
            _get(e, "operating_income"),
        ),
        _check(
            "Net Profit = Operating Income - Tax Expense",
            "net_profit == operating_income - tax_expense",
            {"operating_income": _get(e, "operating_income"), "tax_expense": _get(e, "tax_expense")},
            (_get(e, "operating_income") - _get(e, "tax_expense")) if None not in (_get(e, "operating_income"), _get(e, "tax_expense")) else None,
            _get(e, "net_profit"),
        ),
        _check(
            "Total Income = Interest Earned + Other Income",
            "total_income == interest_earned + other_income",
            {"interest_earned": interest_earned, "other_income": other_income},
            (interest_earned + other_income) if None not in (interest_earned, other_income) else None,
            total_income,
        ),
        _check(
            "Total Expenditure = Interest Expended + Operating Expenses + Provisions",
            "total_expenditure == interest_expended + operating_expenses + provisions_and_contingencies",
            {
                "interest_expended": interest_expended,
                "operating_expenses": opex,
                "provisions_and_contingencies": provisions,
            },
            (interest_expended + opex + provisions) if None not in (interest_expended, opex, provisions) else None,
            total_expenditure,
        ),
        _check(
            "Profit Before Minority = Total Income - Total Expenditure",
            "profit_before_minority_interest == total_income - total_expenditure",
            {"total_income": total_income, "total_expenditure": total_expenditure},
            (total_income - total_expenditure) if None not in (total_income, total_expenditure) else None,
            before_minority,
        ),
        _check(
            "Net Profit Attributable = Profit Before Minority - Minority Interest",
            "net_profit_attributable == profit_before_minority_interest - minority_interest",
            {"profit_before_minority_interest": before_minority, "minority_interest": minority},
            (before_minority - minority) if None not in (before_minority, minority) else None,
            net_profit,
        ),
    ]
    return checks


def _validate_invoice(e: ExtractionResult) -> list[FinancialValidationCheck]:
    subtotal = _get(e, "subtotal")
    tax = _get(e, "tax_amount")
    discount = _get(e, "discount")
    shipping = _get(e, "shipping_charge")
    total = _get(e, "total_amount")
    paid = _get(e, "amount_paid")
    balance_due = _get(e, "balance_due")

    additive_terms = {"subtotal": subtotal, "tax_amount": tax}
    computed_total = None
    if subtotal is not None and tax is not None:
        computed_total = subtotal + tax
        if discount is not None:
            computed_total -= discount
            additive_terms["discount"] = discount
        if shipping is not None:
            computed_total += shipping
            additive_terms["shipping_charge"] = shipping

    checks = [
        _check(
            "Total = Subtotal + Tax (+ Shipping - Discount)",
            "total_amount == subtotal + tax_amount + shipping_charge - discount",
            additive_terms,
            computed_total,
            total,
        ),
        _check(
            "Balance Due = Total - Amount Paid",
            "balance_due == total_amount - amount_paid",
            {"total_amount": total, "amount_paid": paid},
            (total - paid) if (total is not None and paid is not None) else None,
            balance_due,
        ),
    ]
    return checks


_VALIDATORS = {
    DocumentType.BALANCE_SHEET: _validate_balance_sheet,
    DocumentType.CASH_FLOW: _validate_cash_flow,
    DocumentType.PROFIT_AND_LOSS: _validate_profit_and_loss,
    DocumentType.INVOICE: _validate_invoice,
}


def run_financial_validations(
    document_type: DocumentType, extraction: ExtractionResult
) -> list[FinancialValidationCheck]:
    validator = _VALIDATORS.get(document_type)
    if validator is None:
        return []
    return validator(extraction)
