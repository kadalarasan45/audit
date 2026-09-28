# Power BI / Power Query data model

Every file here is one refresh of `auditbot`. To build the report:

1. Power BI Desktop > Get Data > Folder > point at this folder > Combine & Load,
   or Get Data > Text/CSV for individual tables.
2. Mark `dim_date[date]` as the date table.
3. Create the relationships:
   - dim_date[date]            1 --> * fact_transactions[date]
   - dim_date[date]            1 --> * fact_bank_recon[date]
   - dim_date[date]            1 --> * fact_ageing[bill_date]
   - dim_date[date]            1 --> * fact_exceptions[date]
   - dim_party[party_key]      1 --> * fact_transactions[party_key]
   - dim_party[party_key]      1 --> * fact_ageing[party_key]

Suggested measures:

    Exceptions High   = CALCULATE(COUNTROWS(fact_exceptions),
                                  fact_exceptions[severity] = "HIGH")
    Exception Value   = SUM(fact_exceptions[amount])
    ITC at Risk       = SUM(fact_itc_books_only[tax])
    Overdue Debtors   = CALCULATE(SUM(fact_ageing[balance]),
                                  fact_ageing[side] = "Debtors",
                                  fact_ageing[overdue] = TRUE())
    Collection Days   = AVERAGE(fact_billwise_receipts[days_taken])
    GST Shortfall     = SUM(fact_gst_payment[shortfall])

`kpi.csv` holds the same headline numbers the HTML dashboard shows, so a card
visual can read them directly without any DAX.
