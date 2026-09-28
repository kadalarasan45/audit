DROP CLIENT FILES IN THIS FOLDER
================================

Any format.      CSV, Excel (.xlsx/.xls/.xlsm), PDF, tab-separated text.
Any filename.    "doc_4.xlsx" works as well as "sales_register.xlsx".
Any arrangement. Subfolders are fine. Order does not matter.
Any subset.      Send what you have; missing documents are skipped, not fatal.

The system reads each file's COLUMN HEADINGS and works out what it is.
Filenames are only ever used as a tie-breaker.

Then run either of these from the project folder:

    ./run.sh detect data/inbox      <- just tell me what you think these are
    ./run.sh run --inbox data/inbox <- identify them and reconcile everything

Or, if you would rather not touch a terminal:

    ./run.sh serve                  <- opens a page you can drag files onto

What it can recognise
---------------------
  Bank statement                 Receipts register
  Bank ledger (books)            Payments register
  Cash book                      Debtors outstanding
  Sales register                 Creditors outstanding
  Purchase register              Salary register
  GSTR-2B (portal download)      Stock summary
  GSTR-1 summary                 Production / conversion
  GSTR-3B summary

If a file is not recognised, `detect` says so and prints the headings it
found. Add those headings to config/column_maps.yaml and it will be picked
up next time - no code change needed.

Nothing in this folder is sent anywhere. Everything runs on this machine.
