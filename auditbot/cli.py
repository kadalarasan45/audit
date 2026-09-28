"""Command line interface.

    python -m auditbot run              full pipeline + all reports
    python -m auditbot check            what the config resolves to
    python -m auditbot rules            list every check performed
    python -m auditbot sample           regenerate the demo dataset
"""
from __future__ import annotations

import argparse
import logging
import sys
import webbrowser
from pathlib import Path

import pandas as pd

from . import dashboard, detect, powerbi, report_excel
from .config import ROOT, Config
from .exceptions import rule_catalogue
from .io_utils import expand
from .normalize import inr
from .pipeline import run as run_pipeline


def _log(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("auditbot").setLevel(logging.DEBUG if verbose else logging.INFO)


def _rule(width: int = 74) -> None:
    print("-" * width)


def _apply_inbox(cfg: Config, folder: str) -> None:
    """Auto-detect mode: forget the configured paths and work out what every
    file in *folder* is from its column headings."""
    path = Path(folder)
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_dir():
        raise SystemExit(f"inbox folder not found: {path}")
    dets = detect.scan(path, cfg)
    if not dets:
        raise SystemExit(f"no readable files in {path}")
    print()
    print(detect.summary_frame(dets).to_string(index=False))
    unknown = [d for d in dets if not d.source]
    if unknown:
        print(f"\n  {len(unknown)} file(s) not recognised and skipped: "
              + ", ".join(d.path.name for d in unknown))
    cfg._d["sources"] = {k: [] for k in detect.SOURCE_SECTION}
    cfg._d["sources"].update(detect.to_sources(dets))
    print()


def cmd_run(args) -> int:
    cfg = Config.load(args.config)
    if args.out:
        cfg._d.setdefault("report", {})["output_dir"] = args.out
    if args.inbox:
        _apply_inbox(cfg, args.inbox)

    res = run_pipeline(cfg)
    outdir = cfg.outdir
    made: list[Path] = []

    if not args.no_excel:
        made.append(report_excel.write(res, outdir / str(
            cfg.get("report.workbook_name", "Audit_Working_Papers.xlsx"))))
    if not args.no_dashboard:
        made.append(dashboard.write(res, outdir / str(
            cfg.get("report.dashboard_name", "dashboard.html"))))
    if not args.no_powerbi:
        made.append(powerbi.write(res, outdir / "powerbi"))
    res.register.to_csv(outdir / "exception_register.csv", index=False,
                        encoding="utf-8-sig")
    made.append(outdir / "exception_register.csv")

    k = res.kpis
    print()
    _rule()
    print(f"  {cfg.client}   FY {cfg.fy}   ({k['period']})")
    _rule()
    for label, key, money in (
        ("Sales", "sales", True), ("Purchases", "purchases", True),
        ("Debtors closing", "debtors", True), ("Creditors closing", "creditors", True),
        ("Bank match rate", "bank_match_rate", False),
        ("Bank unreconciled", "bank_unreconciled", True),
        ("GSTR-2B match rate", "gst_match_rate", False),
        ("ITC at risk", "itc_at_risk", True),
        ("ITC unclaimed", "itc_unclaimed", True),
        ("GST short paid", "gst_shortfall", True),
        ("Debtors overdue", "debtors_overdue", True),
        ("Creditors overdue", "creditors_overdue", True),
    ):
        v = k.get(key, 0)
        print(f"  {label:<24}{('Rs. ' + inr(v)) if money else f'{v}%':>22}")
    _rule()
    print(f"  Exceptions: {k['exceptions_total']}  "
          f"(HIGH {k['exceptions_high']} | MEDIUM {k['exceptions_medium']} | "
          f"LOW {k['exceptions_low']})   from {k['rules_run']} checks")
    if not res.summary.empty:
        print()
        for _, r in res.summary.iterrows():
            print(f"    {r['category']:<14}{int(r['HIGH']):>5} high "
                  f"{int(r['MEDIUM']):>5} med {int(r['LOW']):>5} low   "
                  f"Rs. {inr(r['Value involved']):>18}")
    _rule()
    print("  Written:")
    for p in made:
        print(f"    {p.relative_to(ROOT) if ROOT in p.parents or p == ROOT else p}")
    _rule()

    if args.open and not args.no_dashboard:
        webbrowser.open((outdir / str(cfg.get("report.dashboard_name",
                                              "dashboard.html"))).as_uri())
    return 0


def cmd_check(args) -> int:
    cfg = Config.load(args.config)
    print(f"\nConfig      : {cfg.path}")
    print(f"Client      : {cfg.client}")
    print(f"GSTIN       : {cfg.get('client.gstin')}   state {cfg.state_code}")
    print(f"Period      : {cfg.period_from.date()} to {cfg.period_to.date()}")
    print(f"Output      : {cfg.outdir}\n")
    _rule()
    print(f"  {'source':<12}{'files':>6}  matched")
    _rule()
    missing = 0
    for name in (cfg.get("sources") or {}):
        files = expand(cfg.source_patterns(name), ROOT)
        if not files:
            missing += 1
        names = ", ".join(f.name for f in files) or "-- none found --"
        print(f"  {name:<12}{len(files):>6}  {names[:52]}")
    _rule()
    print(f"  {missing} source(s) unresolved; the pipeline skips those silently.\n")
    return 0


def cmd_rules(args) -> int:
    cat = rule_catalogue()
    with pd.option_context("display.max_colwidth", 60, "display.width", 200):
        print()
        print(cat.to_string(index=False))
        print(f"\n  {len(cat)} checks "
              f"({(cat['Severity'] == 'HIGH').sum()} high, "
              f"{(cat['Severity'] == 'MEDIUM').sum()} medium, "
              f"{(cat['Severity'] == 'LOW').sum()} low)\n")
    return 0


def cmd_detect(args) -> int:
    cfg = Config.load(args.config)
    folder = Path(args.folder)
    if not folder.is_absolute():
        folder = ROOT / folder
    dets = detect.scan(folder, cfg)
    if not dets:
        print(f"\n  no readable files in {folder}\n")
        return 1
    with pd.option_context("display.max_colwidth", 72, "display.width", 220):
        print()
        print(detect.summary_frame(dets).to_string(index=False))
    known = sum(1 for d in dets if d.source)
    print(f"\n  {known} of {len(dets)} files identified.\n")
    return 0


def cmd_serve(args) -> int:
    from .server import serve
    serve(port=args.port, open_browser=not args.no_browser)
    return 0


def cmd_sample(args) -> int:
    sys.path.insert(0, str(ROOT / "scripts"))
    import make_sample_data                      # noqa: E402
    make_sample_data.main()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="auditbot",
        description="Audit office automation: ingest, reconcile, flag exceptions, report.")
    p.add_argument("-c", "--config", help="path to config.yaml")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd")

    r = sub.add_parser("run", help="run the full pipeline and write the reports")
    r.add_argument("-o", "--out", help="output directory (overrides config)")
    r.add_argument("-i", "--inbox", metavar="FOLDER",
                   help="ignore the configured paths; auto-detect every file in "
                        "FOLDER from its column headings")
    r.add_argument("--no-excel", action="store_true")
    r.add_argument("--no-dashboard", action="store_true")
    r.add_argument("--no-powerbi", action="store_true")
    r.add_argument("--open", action="store_true", help="open the dashboard when done")
    r.set_defaults(func=cmd_run)

    d = sub.add_parser("detect", help="say what each file in a folder is, and stop")
    d.add_argument("folder", nargs="?", default="data/inbox")
    d.set_defaults(func=cmd_detect)

    sv = sub.add_parser("serve", help="open the drag-and-drop tester in a browser")
    sv.add_argument("-p", "--port", type=int, default=8765)
    sv.add_argument("--no-browser", action="store_true")
    sv.set_defaults(func=cmd_serve)

    sub.add_parser("check", help="show what the config resolves to").set_defaults(func=cmd_check)
    sub.add_parser("rules", help="list every check performed").set_defaults(func=cmd_rules)
    sub.add_parser("sample", help="regenerate the demo dataset").set_defaults(func=cmd_sample)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _log(getattr(args, "verbose", False))
    if not getattr(args, "cmd", None):
        args = parser.parse_args((argv or []) + ["run"])
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
