"""Self-contained HTML audit dashboard.

No network, no build step, no external libraries: one file the partner can
open or mail. Colours are the data-viz reference palette (categorical slots
1-3, the reserved status palette for severity, and an ordinal blue ramp for
ageing buckets) declared once as CSS custom properties, so light and dark are
both selected rather than an automatic flip.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .charts import compact, esc, hbar, legend, line_chart, stacked_bar, table
from .normalize import inr, r2
from .pipeline import Result

SEV_COLOR = {"HIGH": "var(--critical)", "MEDIUM": "var(--serious)", "LOW": "var(--warning)"}

CSS = """
*,*::before,*::after{box-sizing:border-box}
.viz-root{
  color-scheme:light;
  --plane:#f9f9f7; --surface-1:#fcfcfb;
  --text-primary:#0b0b0b; --text-secondary:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --ring:rgba(11,11,11,.10);
  --series-1:#2a78d6; --series-2:#eb6834; --series-3:#1baf7a;
  --good:#0ca30c; --warning:#fab219; --serious:#ec835a; --critical:#d03b3b;
  --ord-1:#86b6ef; --ord-2:#6da7ec; --ord-3:#5598e7;
  --ord-4:#2a78d6; --ord-5:#1c5cab; --ord-6:#104281;
}
@media (prefers-color-scheme:dark){
  :root:where(:not([data-theme="light"])) .viz-root{
    color-scheme:dark;
    --plane:#0d0d0d; --surface-1:#1a1a19;
    --text-primary:#fff; --text-secondary:#c3c2b7; --muted:#898781;
    --grid:#2c2c2a; --axis:#383835; --ring:rgba(255,255,255,.10);
    --series-1:#3987e5; --series-2:#d95926; --series-3:#199e70;
    --ord-1:#cde2fb; --ord-2:#9ec5f4; --ord-3:#6da7ec;
    --ord-4:#3987e5; --ord-5:#256abf; --ord-6:#184f95;
  }
}
:root[data-theme="dark"] .viz-root{
  color-scheme:dark;
  --plane:#0d0d0d; --surface-1:#1a1a19;
  --text-primary:#fff; --text-secondary:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --ring:rgba(255,255,255,.10);
  --series-1:#3987e5; --series-2:#d95926; --series-3:#199e70;
  --ord-1:#cde2fb; --ord-2:#9ec5f4; --ord-3:#6da7ec;
  --ord-4:#3987e5; --ord-5:#256abf; --ord-6:#184f95;
}
html,body{margin:0;padding:0}
body{background:var(--plane)}
.viz-root{
  background:var(--plane); color:var(--text-primary); min-height:100vh;
  font:400 15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif;
  padding:28px 16px 64px;
}
.wrap{max-width:1180px;margin:0 auto}
header.top{display:flex;flex-wrap:wrap;gap:16px;align-items:flex-end;
  justify-content:space-between;margin-bottom:26px}
h1{font-size:26px;margin:0 0 4px;letter-spacing:-.01em}
.meta{color:var(--text-secondary);font-size:13.5px;margin:0}
button.theme{background:var(--surface-1);color:var(--text-secondary);cursor:pointer;
  border:1px solid var(--ring);border-radius:8px;padding:8px 14px;font:inherit;font-size:13px}
button.theme:hover{color:var(--text-primary)}
section{background:var(--surface-1);border:1px solid var(--ring);border-radius:14px;
  padding:20px 22px;margin-bottom:18px}
h2{font-size:16px;margin:0 0 3px;letter-spacing:-.005em}
.sub{color:var(--muted);font-size:12.5px;margin:0 0 16px}
.kpis{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(196px,1fr));
  margin-bottom:18px}
.kpi{background:var(--surface-1);border:1px solid var(--ring);border-radius:14px;padding:16px 18px}
.kpi .k{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.055em;
  margin:0 0 8px}
.kpi .v{font-size:27px;font-weight:650;letter-spacing:-.02em;line-height:1.1;
  font-variant-numeric:tabular-nums}
.kpi .n{font-size:12.5px;color:var(--text-secondary);margin-top:6px}
.kpi.bad .v{color:var(--critical)} .kpi.ok .v{color:var(--good)}
.grid2{display:grid;gap:26px;grid-template-columns:repeat(auto-fit,minmax(330px,1fr))}
.chart{width:100%;height:auto;overflow:visible}
.chart .lbl{fill:var(--text-secondary);font-size:12.5px}
.chart .val{fill:var(--text-primary);font-size:12.5px;font-variant-numeric:tabular-nums}
.chart .seg{fill:#fff;font-size:12px;font-weight:600}
.chart .axl{fill:var(--muted);font-size:11.5px;font-variant-numeric:tabular-nums}
.chart .end{font-size:12px;font-weight:600}
.chart .grid{stroke:var(--grid);stroke-width:1}
.chart .axis{stroke:var(--axis);stroke-width:1}
.chart .ln{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.chart .dot{opacity:0;stroke:var(--surface-1);stroke-width:2}
.chart .dot:hover,.chart .dot:focus{opacity:1;r:5;outline:none}
.chart .hit{fill:transparent}
.chart .mk:hover .bar,.chart .mk:focus .bar{filter:brightness(1.08)}
.chart .mk:focus{outline:none}
.legend{display:flex;flex-wrap:wrap;gap:16px;margin:12px 0 0;font-size:12.5px;
  color:var(--text-secondary)}
.lg{display:inline-flex;align-items:center;gap:7px}
.lg i{width:11px;height:11px;border-radius:3px;display:inline-block}
.tbl{border-collapse:collapse;width:100%;font-size:13px;margin-top:6px}
.tbl th,.tbl td{border-bottom:1px solid var(--grid);padding:7px 10px;text-align:left;
  vertical-align:top}
.tbl th{color:var(--muted);font-weight:600;font-size:11.5px;text-transform:uppercase;
  letter-spacing:.05em;white-space:nowrap}
.tbl td.n,.tbl th.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.tbl tbody tr:hover{background:var(--plane)}
details{margin-top:14px}
summary{cursor:pointer;color:var(--text-secondary);font-size:12.5px;
  list-style:none;display:inline-flex;align-items:center;gap:6px}
summary::-webkit-details-marker{display:none}
summary::before{content:"▸";font-size:10px}
details[open] summary::before{content:"▾"}
.chip{display:inline-flex;align-items:center;gap:5px;border-radius:999px;padding:2px 9px;
  font-size:11px;font-weight:650;letter-spacing:.03em;white-space:nowrap;
  border:1px solid var(--ring)}
.chip.HIGH{color:var(--critical)} .chip.MEDIUM{color:var(--serious)}
.chip.LOW{color:var(--warning)}
.chip::before{content:"●";font-size:8px}
.filters{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:12px}
.filters button{background:var(--surface-1);border:1px solid var(--ring);border-radius:999px;
  padding:5px 13px;font:inherit;font-size:12.5px;color:var(--text-secondary);cursor:pointer}
.filters button[aria-pressed="true"]{background:var(--text-primary);color:var(--surface-1);
  border-color:var(--text-primary)}
.empty{color:var(--muted);font-size:13px;margin:8px 0}
#tip{position:fixed;pointer-events:none;opacity:0;transition:opacity .08s;z-index:40;
  background:var(--text-primary);color:var(--surface-1);border-radius:7px;
  padding:6px 10px;font-size:12.5px;white-space:pre;font-variant-numeric:tabular-nums;
  box-shadow:0 4px 14px rgba(0,0,0,.18)}
footer{color:var(--muted);font-size:12px;margin-top:26px;text-align:center}
@media (max-width:640px){.viz-root{padding:18px 12px 48px}h1{font-size:21px}
  .kpi .v{font-size:23px}section{padding:16px 14px}}
@media print{.chart .dot{opacity:1}button.theme,.filters{display:none}}
"""

JS = """
(function(){
  var tip=document.getElementById('tip');
  function show(e,t){tip.textContent=t;tip.style.opacity=1;move(e);}
  function move(e){
    var x=(e.clientX||0)+14,y=(e.clientY||0)+16;
    var r=tip.getBoundingClientRect();
    if(x+r.width>innerWidth-8)x=innerWidth-r.width-8;
    if(y+r.height>innerHeight-8)y=(e.clientY||0)-r.height-12;
    tip.style.left=x+'px';tip.style.top=y+'px';
  }
  document.addEventListener('mouseover',function(e){
    var el=e.target.closest('[data-tip]'); if(el)show(e,el.getAttribute('data-tip'));
  });
  document.addEventListener('mousemove',function(e){
    if(tip.style.opacity==='1'&&e.target.closest('[data-tip]'))move(e);
  });
  document.addEventListener('mouseout',function(e){
    if(e.target.closest('[data-tip]'))tip.style.opacity=0;
  });
  document.addEventListener('focusin',function(e){
    var el=e.target.closest('[data-tip]');
    if(el){var b=el.getBoundingClientRect();
      show({clientX:b.left+b.width/2,clientY:b.top},el.getAttribute('data-tip'));}
  });
  document.addEventListener('focusout',function(){tip.style.opacity=0;});

  var root=document.documentElement, btn=document.querySelector('button.theme');
  function label(){
    var d=root.getAttribute('data-theme')==='dark'||
      (!root.getAttribute('data-theme')&&matchMedia('(prefers-color-scheme:dark)').matches);
    btn.textContent=d?'Light mode':'Dark mode';
  }
  if(btn){
    try{var s=localStorage.getItem('auditbot-theme'); if(s)root.setAttribute('data-theme',s);}
    catch(err){}
    label();
    btn.addEventListener('click',function(){
      var d=root.getAttribute('data-theme')==='dark'||
        (!root.getAttribute('data-theme')&&matchMedia('(prefers-color-scheme:dark)').matches);
      root.setAttribute('data-theme',d?'light':'dark');
      try{localStorage.setItem('auditbot-theme',d?'light':'dark');}catch(err){}
      label();
    });
  }

  var rows=[].slice.call(document.querySelectorAll('#exrows tr'));
  [].forEach.call(document.querySelectorAll('.filters button'),function(b){
    b.addEventListener('click',function(){
      var f=b.getAttribute('data-sev');
      [].forEach.call(document.querySelectorAll('.filters button'),function(o){
        o.setAttribute('aria-pressed', o===b?'true':'false');});
      rows.forEach(function(r){
        r.style.display=(f==='ALL'||r.getAttribute('data-sev')===f)?'':'none';});
    });
  });
})();
"""


def _kpi(title: str, value: str, note: str = "", tone: str = "") -> str:
    return (f'<div class="kpi {tone}"><p class="k">{esc(title)}</p>'
            f'<div class="v">{esc(value)}</div>'
            + (f'<div class="n">{esc(note)}</div>' if note else "") + "</div>")


def _section(title: str, sub: str, body: str) -> str:
    return (f'<section><h2>{esc(title)}</h2><p class="sub">{esc(sub)}</p>{body}</section>')


def _ageing_chart(buckets: pd.DataFrame) -> str:
    if buckets is None or buckets.empty:
        return '<p class="empty">No ageing data.</p>'
    ramp = [f"var(--ord-{i})" for i in range(1, 7)]
    items = [(str(r["Ageing bucket"]), float(r["Amount"])) for _, r in buckets.iterrows()]
    items = [(k, v) for k, v in items if abs(v) > 0]
    if not items:
        return '<p class="empty">No ageing data.</p>'
    # one bar per bucket, coloured by bucket order (ordinal, not by rank)
    top = max(abs(v) for _, v in items) or 1
    width, height, label_w = 640, 26, 92
    plot_w = width - label_w - 104
    rows = []
    for i, (label, value) in enumerate(items):
        y, w = i * height, max(2.0, abs(value) / top * plot_w)
        color = ramp[min(i, len(ramp) - 1)]
        rows.append(
            f'<g class="mk" tabindex="0" role="listitem" '
            f'data-tip="{esc(label)} days Rs. {esc(inr(value))}">'
            f'<rect class="hit" x="0" y="{y}" width="{width}" height="{height}"/>'
            f'<text class="lbl" x="{label_w - 8}" y="{y + height / 2}" text-anchor="end" '
            f'dominant-baseline="central">{esc(label)}</text>'
            f'<rect class="bar" x="{label_w}" y="{y + 4}" width="{w:.1f}" '
            f'height="{height - 9}" rx="4" fill="{color}"/>'
            f'<text class="val" x="{label_w + w + 8:.1f}" y="{y + height / 2}" '
            f'dominant-baseline="central">{esc(compact(value))}</text></g>')
    return (f'<svg class="chart" viewBox="0 0 {width} {len(items) * height + 6}" '
            f'role="list" preserveAspectRatio="xMidYMin meet">'
            f'<line class="axis" x1="{label_w}" y1="0" x2="{label_w}" '
            f'y2="{len(items) * height}"/>' + "".join(rows) + "</svg>")


def _df_table(df: pd.DataFrame, cols: dict[str, str], numeric: tuple = (),
              limit: int = 40) -> str:
    if df is None or df.empty:
        return '<p class="empty">Nothing to show.</p>'
    sub = df[[c for c in cols if c in df.columns]].head(limit)
    headers = [cols[c] for c in sub.columns]
    rows = []
    for _, r in sub.iterrows():
        out = []
        for c in sub.columns:
            v = r[c]
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out.append(inr(v))
            elif isinstance(v, pd.Timestamp):
                out.append(v.strftime("%d-%b-%Y") if pd.notna(v) else "")
            else:
                out.append("" if pd.isna(v) else str(v))
        rows.append(out)
    nidx = tuple(i for i, c in enumerate(sub.columns) if c in numeric)
    return table(headers, rows, numeric=nidx)


def build(res: Result) -> str:
    k, cfg = res.kpis, res.cfg
    reg = res.register

    # ---------------------------------------------------------- KPI tiles
    tiles = [
        _kpi("Exceptions raised", f"{k['exceptions_total']:,}",
             f"{k['exceptions_high']} high · {k['exceptions_medium']} medium "
             f"· {k['exceptions_low']} low",
             "bad" if k["exceptions_high"] else "ok"),
        _kpi("ITC at risk", f"Rs. {compact(k['itc_at_risk'])}",
             "purchases not reflected in GSTR-2B",
             "bad" if k["itc_at_risk"] else "ok"),
        _kpi("Bank unreconciled", f"Rs. {compact(k['bank_unreconciled'])}",
             f"{k['bank_match_rate']}% of lines matched",
             "bad" if abs(float(k["bank_unreconciled"] or 0)) > 1 else "ok"),
        _kpi("Debtors overdue", f"Rs. {compact(k['debtors_overdue'])}",
             f"of Rs. {compact(k['debtors'])} outstanding"),
        _kpi("Creditors overdue", f"Rs. {compact(k['creditors_overdue'])}",
             f"of Rs. {compact(k['creditors'])} outstanding"),
        _kpi("GST short paid", f"Rs. {compact(k['gst_shortfall'])}",
             "liability less tax actually paid",
             "bad" if float(k["gst_shortfall"] or 0) > 0 else "ok"),
        _kpi("Cash breaches", f"{k['cash_above_limit']:,}",
             "s.40A(3) / s.269ST exceptions",
             "bad" if k["cash_above_limit"] else "ok"),
        _kpi("Duplicates found", f"{k['duplicates']:,}",
             "invoices, payments and receipts",
             "bad" if k["duplicates"] else "ok"),
    ]

    # -------------------------------------------------------- exceptions
    sev_counts = [(s, int((reg["severity"] == s).sum()), SEV_COLOR[s])
                  for s in ("HIGH", "MEDIUM", "LOW")] if not reg.empty else []
    sev_bar = stacked_bar(sev_counts, unit="exceptions")
    sev_legend = legend([(f"{s} ({n})", c) for s, n, c in sev_counts])

    area = res.summary
    area_bar = hbar(sorted([(r["category"], float(r["Total"])) for _, r in area.iterrows()],
                       key=lambda t: -t[1]),
                    unit="", value_fmt=lambda v: f"{int(v)}") if not area.empty else ""
    area_table = table(
        ["Area", "High", "Medium", "Low", "Total", "Value involved (Rs.)"],
        [[r["category"], int(r["HIGH"]), int(r["MEDIUM"]), int(r["LOW"]),
          int(r["Total"]), inr(r["Value involved"])] for _, r in area.iterrows()],
        numeric=(1, 2, 3, 4, 5)) if not area.empty else ""

    exceptions_body = (
        '<div class="grid2">'
        f"<div><h2 style='font-size:13.5px;color:var(--text-secondary);margin:0 0 10px'>"
        f"By severity</h2>{sev_bar}{sev_legend}</div>"
        f"<div><h2 style='font-size:13.5px;color:var(--text-secondary);margin:0 0 10px'>"
        f"By area (count)</h2>{area_bar}</div></div>"
        f"<details><summary>Table view</summary>{area_table}</details>")

    # ------------------------------------------------ reconciliation state
    recon_rows = []

    def add(area_name, measure, value, status):
        recon_rows.append([area_name, measure, value, status])

    add("Bank", "Statement lines matched to books",
        f"{k['bank_match_rate']}%",
        "Reconciled" if abs(float(k["bank_unreconciled"] or 0)) <= 1 else "Difference")
    add("Bank", "Entries in bank not in books", f"{k['bank_only']}", "Post in books")
    add("Bank", "Cheques entered but not cleared", f"{k['book_only']}", "Track")
    add("GST", "Purchase invoices matched to GSTR-2B", f"{k['gst_match_rate']}%",
        "Follow up" if k["itc_at_risk"] else "Agreed")
    add("GST", "ITC unclaimed (in 2B, not in books)",
        f"Rs. {inr(k['itc_unclaimed'])}", "Claim")
    add("Bill-wise", "Receipts left on account", f"Rs. {inr(k['unapplied_receipts'])}",
        "Allocate" if float(k["unapplied_receipts"] or 0) > 0 else "Fully applied")
    add("Bill-wise", "Payments left on account", f"Rs. {inr(k['unapplied_payments'])}",
        "Allocate" if float(k["unapplied_payments"] or 0) > 0 else "Fully applied")
    add("Payroll", "Gross salary cost", f"Rs. {inr(k['salary_cost'])}", "Processed")
    add("Inventory", "Closing stock value", f"Rs. {inr(k['closing_stock'])}", "Valued")

    brs = res.get("brs")
    brs_table = table(["Ref", "Particulars", "Amount (Rs.)"],
                      [[r["Ref"], r["Particulars"], inr(r["Amount"])]
                       for _, r in brs.iterrows()], numeric=(2,)) \
        if brs is not None and not brs.empty else '<p class="empty">No bank data.</p>'

    recon_body = (
        '<div class="grid2"><div>'
        + table(["Area", "Measure", "Value", "Action"], recon_rows, numeric=(2,))
        + "</div><div><h2 style='font-size:13.5px;color:var(--text-secondary);"
          "margin:0 0 10px'>Bank reconciliation statement</h2>" + brs_table
        + "</div></div>")

    # ------------------------------------------------------------ ageing
    ageing_body = (
        '<div class="grid2">'
        "<div><h2 style='font-size:13.5px;color:var(--text-secondary);margin:0 0 10px'>"
        "Debtors by age of bill</h2>" + _ageing_chart(res.get("debtors_buckets"))
        + "</div><div><h2 style='font-size:13.5px;color:var(--text-secondary);"
          "margin:0 0 10px'>Creditors by age of bill</h2>"
        + _ageing_chart(res.get("creditors_buckets")) + "</div></div>"
        + "<details><summary>Table view - top parties</summary>"
        + '<div class="grid2"><div>'
        + _df_table(res.get("debtors_party"),
                    {"party": "Debtor", "outstanding": "Outstanding",
                     "max_age_days": "Oldest (days)", "% of total": "% of total"},
                    numeric=("outstanding", "max_age_days", "% of total"), limit=10)
        + "</div><div>"
        + _df_table(res.get("creditors_party"),
                    {"party": "Creditor", "outstanding": "Outstanding",
                     "max_age_days": "Oldest (days)", "% of total": "% of total"},
                    numeric=("outstanding", "max_age_days", "% of total"), limit=10)
        + "</div></div></details>")

    # ------------------------------------------------------------- trend
    trend_body = '<p class="empty">No monthly data.</p>'
    st, pt = res.get("sales_trend"), res.get("purchase_trend")
    if st is not None and not st.empty:
        months = [str(m) for m in st["month"]]
        # "2025-04" repeated twelve times collides; "Apr" plus a year on the
        # January tick reads cleanly at this width
        axis_labels = []
        for m in months:
            ts = pd.Period(m, "M").to_timestamp()
            axis_labels.append(ts.strftime("%b") if ts.month not in (1, 4)
                               else ts.strftime("%b '%y"))
        sales = [float(v) for v in st["value"]]
        purch = ([float(pt.set_index("month")["value"].get(m, 0)) for m in months]
                 if pt is not None and not pt.empty else [0] * len(months))
        rec_col = "receipts_settled" if "receipts_settled" in st.columns else None
        recs = [float(v) for v in st[rec_col]] if rec_col else [0] * len(months)
        series = [("Sales", sales, "var(--series-1)"),
                  ("Purchases", purch, "var(--series-2)"),
                  ("Receipts", recs, "var(--series-3)")]
        trend_body = (
            line_chart(axis_labels, series)
            + legend([(n, c) for n, _, c in series])
            + "<details><summary>Table view</summary>"
            + table(["Month", "Sales (Rs.)", "Purchases (Rs.)", "Receipts (Rs.)"],
                    [[months[i], inr(sales[i]), inr(purch[i]), inr(recs[i])]
                     for i in range(len(months))], numeric=(1, 2, 3))
            + "</details>")

    # -------------------------------------------------- exception register
    top = reg.head(60) if not reg.empty else pd.DataFrame()
    ex_rows = []
    for _, r in top.iterrows():
        d = r["date"]
        ex_rows.append(
            f'<tr data-sev="{esc(r["severity"])}">'
            f'<td>{esc(r["code"])}</td>'
            f'<td><span class="chip {esc(r["severity"])}">{esc(r["severity"])}</span></td>'
            f'<td>{esc(r["category"])}</td>'
            f'<td>{esc(pd.Timestamp(d).strftime("%d-%b-%Y") if pd.notna(d) else "")}</td>'
            f'<td>{esc(str(r["party"])[:38])}</td>'
            f'<td class="n">{esc(inr(r["amount"]))}</td>'
            f'<td>{esc(r["observation"])}</td>'
            f'<td>{esc(r["suggested_action"])}</td></tr>')
    register_body = (
        '<div class="filters" role="group" aria-label="Filter by severity">'
        '<button data-sev="ALL" aria-pressed="true">All</button>'
        '<button data-sev="HIGH" aria-pressed="false">High</button>'
        '<button data-sev="MEDIUM" aria-pressed="false">Medium</button>'
        '<button data-sev="LOW" aria-pressed="false">Low</button></div>'
        '<div style="overflow-x:auto"><table class="tbl"><thead><tr>'
        "<th>Code</th><th>Severity</th><th>Area</th><th>Date</th><th>Party</th>"
        '<th class="n">Amount (Rs.)</th><th>Observation</th><th>Suggested action</th>'
        f'</tr></thead><tbody id="exrows">{"".join(ex_rows)}</tbody></table></div>'
        f'<p class="sub" style="margin-top:12px">Showing the {len(top)} highest-value '
        f'findings of {len(reg)}. The full register is in the Excel working papers.</p>')

    head = (
        '<header class="top"><div>'
        f"<h1>{esc(cfg.client)}</h1>"
        f'<p class="meta">Audit exception dashboard · FY {esc(cfg.fy)} '
        f'· {esc(k["period"])}</p>'
        f'<p class="meta">Generated {esc(k["generated"])} · '
        f'{k["rules_run"]} checks performed · '
        f'{len(res.data.get("bank", []))} bank lines, '
        f'{len(res.data.get("sales", []))} sales and '
        f'{len(res.data.get("purchase", []))} purchase invoices read</p>'
        '</div><button class="theme" type="button">Dark mode</button></header>')

    body = (
        '<div class="wrap">' + head
        + f'<div class="kpis">{"".join(tiles)}</div>'
        + _section("Exceptions", "Every finding, by how urgent it is and where it came from.",
                   exceptions_body)
        + _section("Reconciliation status",
                   "Where each sub-ledger stands against its external source.", recon_body)
        + _section("Debtors and creditors ageing",
                   f"Bill-wise position as on {cfg.period_to.date()}.", ageing_body)
        + _section("Monthly trend", "Sales, purchases and collections through the year.",
                   trend_body)
        + _section("Exception register", "Ranked by severity, then by value involved.",
                   register_body)
        + '<footer>Generated by auditbot · figures are as extracted from the '
          'client records and are subject to audit verification.</footer></div>')

    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        f"<title>{esc(cfg.client)} - Audit Dashboard</title>"
        f"<style>{CSS}</style></head>"
        f"<body class=\"viz-root\">{body}<div id=\"tip\" role=\"status\"></div>"
        f"<script>{JS}</script></body></html>")


def write(res: Result, path: Path | None = None) -> Path:
    path = Path(path or (res.outdir / str(res.cfg.get("report.dashboard_name",
                                                      "dashboard.html"))))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build(res), encoding="utf-8")
    return path
