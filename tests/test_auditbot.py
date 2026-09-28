"""Test suite. Run with:  .venv/bin/python -m unittest discover -s tests -v"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from auditbot import io_utils as io                                    # noqa: E402
from auditbot.config import Config                                     # noqa: E402
from auditbot.exceptions import REGISTRY, build_register               # noqa: E402
from auditbot.matching import (Pass, allocate_fifo, cascade_match,     # noqa: E402
                               find_duplicates, similarity)
from auditbot.normalize import (clean_amount, fy_bounds, inr,          # noqa: E402
                                invoice_tail, is_round_sum, norm_invoice,
                                norm_party, parse_date, parse_period,
                                validate_gstin, validate_pan)
from auditbot.recon.ageing import age_bills, bucket_labels             # noqa: E402
from auditbot.recon.trade import invoice_sequence                      # noqa: E402


class TestAmounts(unittest.TestCase):
    def test_indian_grouping(self):
        self.assertEqual(clean_amount("1,23,456.78"), 123456.78)
        self.assertEqual(clean_amount("12,34,56,789"), 123456789.0)

    def test_negatives_and_noise(self):
        self.assertEqual(clean_amount("(1,200)"), -1200.0)
        self.assertEqual(clean_amount("-450.50"), -450.50)
        self.assertEqual(clean_amount("₹ 5,000"), 5000.0)
        self.assertEqual(clean_amount(""), 0.0)
        self.assertEqual(clean_amount("-"), 0.0)
        self.assertEqual(clean_amount(None), 0.0)
        self.assertEqual(clean_amount(float("nan")), 0.0)

    def test_dr_cr_suffix(self):
        self.assertEqual(clean_amount("5,000 Dr"), -5000.0)
        self.assertEqual(clean_amount("4,500 Cr"), 4500.0)

    def test_format(self):
        self.assertEqual(inr(12345678.5), "1,23,45,678.50")
        self.assertEqual(inr(-4500), "-4,500.00")
        self.assertEqual(inr(999), "999.00")

    def test_round_sum(self):
        self.assertTrue(is_round_sum(500000, 100000))
        self.assertFalse(is_round_sum(512000, 100000))
        self.assertFalse(is_round_sum(50000, 100000))


class TestDates(unittest.TestCase):
    def test_day_first_is_the_default(self):
        # 05-04-2025 is 5 April in India, not 4 May
        self.assertEqual(parse_date("05-04-2025"), pd.Timestamp("2025-04-05"))
        self.assertEqual(parse_date("05/04/2025"), pd.Timestamp("2025-04-05"))

    def test_formats(self):
        for value in ("5-Apr-25", "2025-04-05", "05.04.2025", "05 Apr 2025"):
            self.assertEqual(parse_date(value), pd.Timestamp("2025-04-05"), value)

    def test_excel_serial(self):
        self.assertEqual(parse_date(45752), pd.Timestamp("2025-04-05"))

    def test_junk(self):
        for value in (None, "", "-", "nil", float("nan")):
            self.assertIsNone(parse_date(value))

    def test_period(self):
        self.assertEqual(parse_period("042025"), "2025-04")     # GST portal MMYYYY
        self.assertEqual(parse_period("Apr-2025"), "2025-04")
        self.assertEqual(parse_period("2025-04"), "2025-04")

    def test_fy(self):
        self.assertEqual(fy_bounds("2025-26"),
                         (pd.Timestamp("2025-04-01"), pd.Timestamp("2026-03-31")))


class TestNames(unittest.TestCase):
    def test_legal_suffixes_collapse(self):
        self.assertEqual(norm_party("M/s. Sharma & Sons Pvt. Ltd."),
                         norm_party("SHARMA AND SONS PRIVATE LIMITED"))

    def test_invoice_key(self):
        self.assertEqual(norm_invoice("INV/2025-26/0042"), "INV2025260042")
        self.assertEqual(norm_invoice("inv 2025-26 0042"), "INV2025260042")
        self.assertEqual(invoice_tail("ABC/25-26/000123", 6), "000123")

    def test_similarity(self):
        self.assertEqual(similarity("SHARMA", "SHARMA"), 100.0)
        self.assertGreater(similarity("GLOBAL POLYMERS", "GLOBAL POLYMER"), 88)
        self.assertLess(similarity("ABC", "XYZ"), 40)


class TestGstin(unittest.TestCase):
    def test_valid(self):
        ok, why = validate_gstin("27AAPFU0939F1ZV")
        self.assertTrue(ok, why)

    def test_bad_check_digit(self):
        ok, why = validate_gstin("27AAPFU0939F1ZX")
        self.assertFalse(ok)
        self.assertIn("check digit", why)

    def test_length_and_blank(self):
        self.assertFalse(validate_gstin("27AAPFU0939F1Z")[0])
        self.assertFalse(validate_gstin("")[0])

    def test_pan(self):
        self.assertTrue(validate_pan("AAPFU0939F"))
        self.assertFalse(validate_pan("AAPFU0939"))
        self.assertFalse(validate_pan(""))


class TestReaders(unittest.TestCase):
    def test_sep_survives_a_banner_row(self):
        text = ("HDFC BANK LIMITED\nStatement of account\n\n"
                "Date,Narration,Withdrawal Amt.,Deposit Amt.\n"
                "01-04-2025,NEFT ABC,5000.00,\n")
        self.assertEqual(io._sniff_sep(text), ",")

    def test_header_detected_below_banner(self):
        cfg = Config.load()
        raw = pd.DataFrame([
            ["HDFC BANK LTD", None, None, None],
            ["Statement of account", None, None, None],
            ["Date", "Narration", "Withdrawal Amt.", "Deposit Amt."],
            ["01-04-2025", "NEFT-ABC", "5,000.00", ""],
        ])
        self.assertEqual(io.find_header_row(raw, cfg.aliases("bank")), 2)
        mapped = io.map_columns(raw, cfg.aliases("bank"))
        self.assertEqual(list(mapped.columns), ["txn_date", "description", "debit", "credit"])
        self.assertEqual(len(mapped), 1)

    def test_unknown_columns_are_kept_not_dropped(self):
        cfg = Config.load()
        raw = pd.DataFrame([["Date", "Narration", "Cost Centre"],
                            ["01-04-2025", "x", "HO"]])
        mapped = io.map_columns(raw, cfg.aliases("bank"))
        self.assertIn("x_costcentre", mapped.columns)

    def test_pdf_column_edges_split_multi_word_headings(self):
        # "Withdrawal Amt." is two words in one column; "Deposit Amt." another
        def w(text, x0, x1):
            return {"text": text, "x0": x0, "x1": x1, "top": 10}
        header = [w("Date", 40, 70), w("Narration", 90, 150),
                  w("Withdrawal", 300, 360), w("Amt.", 362, 385),
                  w("Deposit", 430, 470), w("Amt.", 472, 495)]
        edges = io._column_edges(header)
        self.assertEqual(len(edges) - 1, 4)          # four columns, not six

    def test_pdf_row_keeps_a_blank_column_blank(self):
        """A credit-only row must not slide its number under the debit heading."""
        def w(text, x0, x1):
            return {"text": text, "x0": x0, "x1": x1, "top": 20}
        header = [w("Date", 40, 70), w("Narration", 90, 150),
                  w("Debit", 300, 340), w("Credit", 430, 470)]
        edges = io._column_edges(header)
        row = [w("07-04-2025", 40, 95), w("NEFT", 100, 130), w("CR", 132, 145),
               w("120000.00", 430, 470)]            # credit column only
        cells = [" ".join(x["text"] for x in row
                          if edges[i] <= (x["x0"] + x["x1"]) / 2 < edges[i + 1])
                 for i in range(len(edges) - 1)]
        self.assertEqual(cells[2], "")               # debit stays empty
        self.assertEqual(cells[3], "120000.00")

    def test_total_rows_are_dropped(self):
        cfg = Config.load()
        raw = pd.DataFrame([["Date", "Narration", "Withdrawal Amt."],
                            ["01-04-2025", "x", "100"],
                            ["Total", "", "100"]])
        self.assertEqual(len(io.map_columns(raw, cfg.aliases("bank"))), 1)


def _frame(ids, parties, invs, totals, dates):
    return pd.DataFrame({"row_id": ids, "party_key": parties, "inv_key": invs,
                         "total": totals, "txn_date": pd.to_datetime(dates)})


class TestMatching(unittest.TestCase):
    def setUp(self):
        self.left = _frame(["A1", "A2", "A3"], ["SHARMA", "GUPTA", "RAO"],
                           ["101", "102", "103"], [1000.0, 2000.0, 3000.0],
                           ["2025-04-01", "2025-04-05", "2025-04-09"])
        self.right = _frame(["B1", "B2", "B3"], ["SHARMA", "GUPTAA", "MEHTA"],
                            ["101", "999", "888"], [1000.0, 2000.0, 3000.0],
                            ["2025-04-01", "2025-04-06", "2025-04-20"])
        self.passes = [
            Pass("exact", keys=[("party_key", "party_key"), ("inv_key", "inv_key")]),
            Pass("fuzzy", amount=("total", "total"), date=("txn_date", "txn_date"),
                 date_tol=7, fuzzy=("party_key", "party_key"), confidence=85),
        ]

    def test_cascade_prefers_the_stronger_pass(self):
        res = cascade_match(self.left, self.right, self.passes)
        m = res["matched"].set_index("book_row_id")["match_pass"].to_dict()
        self.assertEqual(m["A1"], "exact")
        self.assertEqual(m["A2"], "fuzzy")

    def test_out_of_window_stays_unmatched(self):
        res = cascade_match(self.left, self.right, self.passes)
        self.assertEqual(res["left_only"]["row_id"].tolist(), ["A3"])
        self.assertEqual(res["right_only"]["row_id"].tolist(), ["B3"])

    def test_matching_is_one_to_one(self):
        right = pd.concat([self.right, self.right.assign(row_id="B1b")],
                          ignore_index=True)
        res = cascade_match(self.left, right, self.passes)
        self.assertEqual(res["matched"]["book_row_id"].nunique(), len(res["matched"]))
        self.assertEqual(res["matched"]["source_row_id"].nunique(), len(res["matched"]))

    def test_empty_inputs(self):
        res = cascade_match(pd.DataFrame(), self.right, self.passes)
        self.assertTrue(res["matched"].empty)


class TestBillwise(unittest.TestCase):
    def setUp(self):
        self.bills = pd.DataFrame({
            "row_id": ["D1", "D2"], "party_key": ["SHARMA"] * 2,
            "party": ["Sharma"] * 2, "bill_key": ["101", "102"],
            "bill_no": ["101", "102"],
            "bill_date": pd.to_datetime(["2025-04-01", "2025-05-01"]),
            "bill_amount": [1000.0, 2000.0]})
        self.vouchers = pd.DataFrame({
            "row_id": ["R1", "R2"], "party_key": ["SHARMA"] * 2,
            "party": ["Sharma"] * 2, "bill_key": ["", "102"],
            "voucher_no": ["RV1", "RV2"],
            "txn_date": pd.to_datetime(["2025-04-20", "2025-05-10"]),
            "amount": [1500.0, 2000.0]})

    def test_explicit_reference_wins_over_fifo(self):
        out = allocate_fifo(self.bills, self.vouchers)
        by_vch = out["allocations"].set_index("voucher_no")
        self.assertEqual(by_vch.loc["RV2", "basis"], "bill reference")
        self.assertEqual(by_vch.loc["RV2", "bill_no"], "102")

    def test_fifo_takes_the_oldest_bill(self):
        out = allocate_fifo(self.bills, self.vouchers)
        row = out["allocations"].set_index("voucher_no").loc["RV1"]
        self.assertEqual(row["bill_no"], "101")
        self.assertEqual(row["allocated"], 1000.0)

    def test_excess_stays_on_account(self):
        out = allocate_fifo(self.bills, self.vouchers)
        self.assertEqual(out["unapplied"]["unapplied_amount"].sum(), 500.0)

    def test_nothing_is_over_allocated(self):
        out = allocate_fifo(self.bills, self.vouchers)
        merged = out["allocations"].groupby("bill_no")["allocated"].sum()
        for bill_no, allocated in merged.items():
            limit = float(self.bills.set_index("bill_no").loc[bill_no, "bill_amount"])
            self.assertLessEqual(allocated, limit + 0.01)

    def test_days_taken(self):
        out = allocate_fifo(self.bills, self.vouchers)
        self.assertEqual(int(out["allocations"].set_index("voucher_no")
                             .loc["RV1", "days_taken"]), 19)


class TestDuplicates(unittest.TestCase):
    def test_same_party_amount_inside_the_window(self):
        df = pd.DataFrame({"party_key": ["A", "A", "B"], "amount": [100.0, 100.0, 100.0],
                           "txn_date": pd.to_datetime(["2025-04-01", "2025-04-02",
                                                       "2025-04-01"])})
        dups = find_duplicates(df, ["party_key"], amount_col="amount",
                               date_col="txn_date", date_window=3)
        self.assertEqual(len(dups), 2)

    def test_outside_the_window_is_not_a_duplicate(self):
        df = pd.DataFrame({"party_key": ["A", "A"], "amount": [100.0, 100.0],
                           "txn_date": pd.to_datetime(["2025-04-01", "2025-06-01"])})
        self.assertTrue(find_duplicates(df, ["party_key"], amount_col="amount",
                                        date_col="txn_date", date_window=3).empty)


class TestAgeing(unittest.TestCase):
    def test_bucket_labels(self):
        self.assertEqual(bucket_labels([30, 60]), ["0-30", "31-60", ">60"])

    def test_buckets_and_overdue(self):
        cfg = Config.load()
        asof = cfg.period_to
        bills = pd.DataFrame({
            "row_id": ["X1", "X2"], "party": ["A", "B"], "party_key": ["A", "B"],
            "bill_no": ["1", "2"], "bill_key": ["1", "2"],
            "bill_date": [asof - pd.Timedelta(days=10), asof - pd.Timedelta(days=400)],
            "due_date": [asof + pd.Timedelta(days=20), asof - pd.Timedelta(days=355)],
            "bill_amount": [100.0, 200.0], "balance": [100.0, 200.0],
            "credit_days": [30.0, 45.0]})
        out = age_bills(bills, cfg, "debtors")
        detail = out["detail"].set_index("row_id")
        self.assertEqual(detail.loc["X1", "bucket"], "0-30")
        self.assertEqual(detail.loc["X2", "bucket"], ">365")
        self.assertFalse(bool(detail.loc["X1", "overdue"]))
        self.assertTrue(bool(detail.loc["X2", "overdue"]))

    def test_empty_input_is_safe(self):
        out = age_bills(pd.DataFrame(), Config.load(), "debtors")
        self.assertTrue(out["detail"].empty)


class TestInvoiceSequence(unittest.TestCase):
    def test_gap_is_reported(self):
        sales = pd.DataFrame({"invoice_no": ["SI/1", "SI/2", "SI/4"]})
        out = invoice_sequence(sales)
        self.assertEqual(int(out.iloc[0]["missing_count"]), 1)
        self.assertIn("3", out.iloc[0]["missing_numbers"])

    def test_separate_series_are_not_mixed(self):
        sales = pd.DataFrame({"invoice_no": ["SI/1", "SI/2", "EX/1", "EX/2"]})
        out = invoice_sequence(sales)
        self.assertEqual(len(out), 2)
        self.assertEqual(out["missing_count"].sum(), 0)


class TestExceptionEngine(unittest.TestCase):
    def test_every_rule_has_a_unique_code(self):
        codes = [r.code for r in REGISTRY]
        self.assertEqual(len(codes), len(set(codes)))

    def test_severities_are_valid(self):
        self.assertTrue(all(r.severity in {"HIGH", "MEDIUM", "LOW"} for r in REGISTRY))

    def test_cash_limit_rule_fires_and_is_configurable(self):
        cfg = Config.load()
        pay = pd.DataFrame({
            "row_id": ["P1", "P2"], "party": ["Transporter", "Transporter"],
            "party_key": ["TRANSPORTER"] * 2, "is_cash": [True, True],
            "amount": [48500.0, 500.0],
            "txn_date": pd.to_datetime(["2025-07-12", "2025-07-13"]),
            "voucher_no": ["V1", "V2"], "narration": ["", ""], "mode": ["CASH"] * 2,
            "bill_no": ["", ""]})
        reg = build_register({"payments": pay}, {}, cfg)
        hits = reg[reg["code"] == "CSH-01"]
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits.iloc[0]["severity"], "HIGH")

    def test_engine_survives_completely_empty_input(self):
        reg = build_register({}, {}, Config.load())
        self.assertTrue(reg.empty)


class TestDetection(unittest.TestCase):
    """The detector must work with no filename clues at all."""

    @classmethod
    def setUpClass(cls):
        if not (ROOT / "data" / "input" / "sales_register.xlsx").exists():
            raise unittest.SkipTest("run `python -m auditbot sample` first")
        import shutil
        import tempfile
        from auditbot import detect
        cls.detect = detect
        cls.tmp = Path(tempfile.mkdtemp())
        (cls.tmp / "nested folder").mkdir()
        src = sorted((ROOT / "data" / "input").rglob("*.xlsx")) + \
            sorted((ROOT / "data" / "input").rglob("*.csv"))
        for i, f in enumerate(src, 1):
            dest = (cls.tmp / "nested folder" if i % 3 == 0 else cls.tmp)
            shutil.copy(f, dest / f"doc_{i}{f.suffix}")     # meaningless names
        cls.dets = detect.scan(cls.tmp, Config.load())

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_identifies_every_source_without_filename_clues(self):
        got = {d.source for d in self.dets if d.source}
        want = set(self.detect.SOURCE_SECTION)
        self.assertEqual(want - got, set(), f"not identified: {sorted(want - got)}")

    def test_no_source_is_claimed_twice(self):
        singles = [d.source for d in self.dets if d.source and d.source != "bank"]
        self.assertEqual(len(singles), len(set(singles)))

    def test_every_detection_explains_itself(self):
        for d in self.dets:
            self.assertTrue(d.why, f"{d.path.name} gave no reason")

    def test_a_ledger_is_never_read_as_a_voucher_register(self):
        """A bank ledger has Debit and Credit columns; inside the receipts
        alias section "Credit" means amount, so the gate must look at the raw
        headers or the ledger gets misread."""
        for d in self.dets:
            if d.source in ("bankbook", "bank", "cashbook"):
                self.assertNotIn(d.source, ("receipts", "payments"))
        sources = {d.source for d in self.dets if d.source}
        self.assertIn("bankbook", sources)

    def test_junk_file_is_reported_not_guessed(self):
        junk = self.tmp / "junk.csv"
        junk.write_text("hello,world\n1,2\n")
        det = self.detect.detect_file(junk, Config.load())
        self.assertIsNone(det.source)
        junk.unlink()

    def test_auto_detect_matches_the_configured_run(self):
        """Auto-detected sources must reconcile to the same numbers as the
        hand-configured paths."""
        from auditbot.pipeline import run
        cfg = Config.load()
        cfg._d["sources"] = {k: [] for k in self.detect.SOURCE_SECTION}
        cfg._d["sources"].update(self.detect.to_sources(self.dets))
        auto = run(cfg)
        configured = run(Config.load())
        self.assertEqual(len(auto.register), len(configured.register))
        self.assertEqual(auto.kpis["sales"], configured.kpis["sales"])
        self.assertEqual(auto.kpis["bank_match_rate"], configured.kpis["bank_match_rate"])


class TestMultipart(unittest.TestCase):
    def test_parses_fields_and_files(self):
        from auditbot.server import parse_multipart
        b = b"X-BOUND"
        body = (b"--X-BOUND\r\nContent-Disposition: form-data; name=\"client\"\r\n\r\n"
                b"Acme Ltd\r\n"
                b"--X-BOUND\r\nContent-Disposition: form-data; name=\"files\"; "
                b"filename=\"a.csv\"\r\nContent-Type: text/csv\r\n\r\n"
                b"date,amount\r\n01-04-2025,100\r\n"
                b"--X-BOUND--\r\n")
        fields, files = parse_multipart(body, b)
        self.assertEqual(fields["client"], "Acme Ltd")
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0][0], "a.csv")
        self.assertIn(b"01-04-2025", files[0][1])

    def test_strips_directory_from_the_filename(self):
        from auditbot.server import parse_multipart
        b = b"B"
        body = (b"--B\r\nContent-Disposition: form-data; name=\"files\"; "
                b"filename=\"../../etc/passwd\"\r\n\r\nx\r\n--B--\r\n")
        _, files = parse_multipart(body, b)
        self.assertEqual(files[0][0], "passwd")


class TestEndToEnd(unittest.TestCase):
    """Runs only when the sample data has been generated."""

    @classmethod
    def setUpClass(cls):
        if not (ROOT / "data" / "input" / "sales_register.xlsx").exists():
            raise unittest.SkipTest("run `python -m auditbot sample` first")
        from auditbot.pipeline import run
        cls.res = run()

    def test_all_sources_loaded(self):
        for key in ("bank", "sales", "purchase", "gstr2b", "receipts", "payments",
                    "debtors", "creditors", "salary", "inventory"):
            self.assertFalse(self.res.data[key].empty, key)

    def test_brs_reconciles_to_nil(self):
        brs = self.res.get("brs")
        diff = float(brs[brs["Ref"] == "H"].iloc[0]["Amount"])
        self.assertAlmostEqual(diff, 0.0, delta=1.0)

    def test_bill_allocation_never_exceeds_the_bill(self):
        status = self.res.get("receipts_bill_status")
        self.assertTrue((status["unpaid_amount"] >= -0.01).all())

    def test_planted_defects_are_all_found(self):
        codes = set(self.res.register["code"])
        for expected in ("CSH-01", "CSH-02", "CSH-03", "DUP-01", "DUP-02", "DUP-03",
                         "GST-01", "GST-02", "GST-03", "GST-04", "GST-07", "GST-09",
                         "BNK-01", "BNK-02", "INV-01", "INV-02", "INV-03",
                         "SAL-01", "SAL-02", "SAL-03", "SAL-04", "SAL-05", "ANL-05"):
            self.assertIn(expected, codes, f"{expected} did not fire")

    def test_register_is_ranked_high_first(self):
        sev = self.res.register["severity"].tolist()
        self.assertEqual(sev, sorted(sev, key={"HIGH": 0, "MEDIUM": 1, "LOW": 2}.get))

    def test_reports_can_be_written(self):
        import tempfile
        from auditbot import dashboard, powerbi, report_excel
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            xlsx = report_excel.write(self.res, tmp / "wp.xlsx")
            html = dashboard.write(self.res, tmp / "d.html")
            pbi = powerbi.write(self.res, tmp / "pbi")
            self.assertGreater(xlsx.stat().st_size, 20_000)
            self.assertGreater(html.stat().st_size, 10_000)
            self.assertGreater(len(list(pbi.glob("*.csv"))), 10)
            self.assertIn("<!doctype html>", html.read_text()[:40])


if __name__ == "__main__":
    unittest.main(verbosity=2)
