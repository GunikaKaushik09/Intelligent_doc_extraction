"""
Prompt templates used to turn raw OCR/native-extracted text into the
structured fields + tables the API contract requires. One template per
in-scope document type (spec section 2).

The instructions are intentionally strict about not inventing values,
because spec 4.2 explicitly forbids hallucinated fields.
"""
from __future__ import annotations

_COMMON_RULES = """
You are a meticulous financial document data-extraction engine.

STRICT RULES:
1. Only extract values that are literally present in the provided document text.
   If a field is not present or you are not confident it is present, set its
   "value" to null. NEVER invent, guess, infer or estimate a value.
2. For every extracted field, include a short verbatim (or near-verbatim)
   snippet of the source text that supports it in "evidence.source_text",
   and the page number it came from in "evidence.page_number" (the text is
   annotated with "[PAGE n]" markers -- use those to determine the page).
3. Return EVERY meaningful field and table row visible in the document, not
   just a small fixed list. Use clear, human-readable snake_case keys.
4. Numbers must be returned as plain numbers (no currency symbols, no thousand
   separators) in a separate numeric form when possible, but keep the
   original formatted string too if useful context (e.g. "1,234.50" ->
   1234.50). Negative values in parentheses, e.g. "(500)", are negative
   numbers: -500.
5. Respond with ONLY a single valid JSON object -- no markdown fences, no
   commentary, no preamble.

Output JSON schema:
{
  "fields": {
    "<field_name>": {
      "value": <string|number|null>,
      "evidence": {"page_number": <int|null>, "source_text": "<string|null>"}
    },
    ...
  },
  "tables": [
    {
      "name": "<table name, e.g. 'Line Items' or 'Current Assets'>",
      "columns": ["<col1>", "<col2>", ...],
      "rows": [
        {"cells": {"<col1>": <value>, "<col2>": <value>, ...},
         "evidence": {"page_number": <int|null>, "source_text": "<string|null>"}}
      ]
    }
  ]
}
"""

BALANCE_SHEET_GUIDANCE = """
Document type: CONSOLIDATED BALANCE SHEET.
Pay particular attention to (extract whenever present):
- company_name, reporting_period_end_date, currency, comparative_period_end_date
- total_current_assets, total_non_current_assets, total_assets
- total_current_liabilities, total_non_current_liabilities, total_liabilities
- total_equity, total_liabilities_and_equity
- Individual line items (cash_and_equivalents, inventory, receivables,
  property_plant_equipment, goodwill, payables, borrowings, retained_earnings,
  share_capital, etc.) as fields AND as a structured table with a column for
  the current period value and a column for the comparative/prior period
  value when the document shows both.
"""

CASH_FLOW_GUIDANCE = """
Document type: CONSOLIDATED CASH FLOW STATEMENT.
Pay particular attention to (extract whenever present):
- company_name, reporting_period_end_date, currency, comparative_period_end_date
- net_cash_from_operating_activities, net_cash_from_investing_activities,
  net_cash_from_financing_activities
- net_increase_decrease_in_cash, cash_at_beginning_of_period, cash_at_end_of_period
- effect_of_exchange_rate_changes (if present)
- A structured table of every individual line item under each of the three
  activity sections (operating/investing/financing), with current and
  comparative period columns where shown.
"""

PROFIT_AND_LOSS_GUIDANCE = """
Document type: CONSOLIDATED PROFIT & LOSS STATEMENT (Income Statement).
Pay particular attention to (extract whenever present):
- company_name, reporting_period_end_date, currency, comparative_period_end_date
- total_revenue, cost_of_goods_sold / cost_of_sales, gross_profit
- operating_expenses, operating_income, other_income, finance_costs,
  tax_expense, net_profit / profit_for_the_period
- earnings_per_share (basic/diluted) if present
- A structured table of every individual line item (revenue lines, expense
  lines, etc.) with current and comparative period columns where shown.
"""

INVOICE_GUIDANCE = """
Document type: INVOICE (may be a photographed / scanned receipt or invoice).
Pay particular attention to (extract whenever present):
- invoice_number, invoice_date, due_date, currency
- seller_name, seller_address, buyer_name, buyer_address
- subtotal, discount, tax_amount, tax_rate, shipping_charge, total_amount,
  amount_paid, balance_due, payment_terms
- A structured "Line Items" table with columns such as description, quantity,
  unit_price, line_total for every row visible on the invoice/receipt.
"""

_GUIDANCE_BY_TYPE = {
    "balance_sheet": BALANCE_SHEET_GUIDANCE,
    "cash_flow": CASH_FLOW_GUIDANCE,
    "profit_and_loss": PROFIT_AND_LOSS_GUIDANCE,
    "invoice": INVOICE_GUIDANCE,
}


def build_extraction_prompt(document_type: str, ocr_text: str) -> str:
    guidance = _GUIDANCE_BY_TYPE.get(document_type, "")
    return (
        f"{_COMMON_RULES}\n{guidance}\n\n"
        f"DOCUMENT TEXT (page-annotated, extracted via OCR/native text extraction):\n"
        f"-----\n{ocr_text}\n-----\n\n"
        f"Now return the JSON object described above for this document."
    )
