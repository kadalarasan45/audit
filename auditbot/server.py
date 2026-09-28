"""A local drag-and-drop tester.

    ./run.sh serve      ->  http://127.0.0.1:8765

Drop any mix of client files in any format, named anything. The server works
out what each one is, shows you its guess so you can correct it in a dropdown,
then runs the full pipeline and hands back the dashboard and the working
papers.

Standard library only, bound to loopback. Nothing is uploaded anywhere -- the
files are copied into a run folder on this machine and stay there.
"""
from __future__ import annotations

import html
import io
import json
import logging
import shutil
import socketserver
import threading
import time
import traceback
import urllib.parse
import webbrowser
import zipfile
from datetime import datetime
from http.server import BaseHTTPRequestHandler
from pathlib import Path

from . import dashboard, detect, powerbi, report_excel
from .config import ROOT, Config
from .charts import compact
from .normalize import inr
from .pipeline import run as run_pipeline

log = logging.getLogger("auditbot.server")

RUNS = ROOT / "output" / "runs"
STATE: dict[str, dict] = {}
MAX_UPLOAD = 200 * 1024 * 1024        # 200 MB per submission


def esc(v) -> str:
    return html.escape(str(v), quote=True)


# ------------------------------------------------------- multipart form

def parse_multipart(body: bytes, boundary: bytes) -> tuple[dict, list[tuple[str, bytes]]]:
    """Minimal multipart/form-data reader (no cgi module -- it is gone in 3.13)."""
    fields: dict[str, str] = {}
    files: list[tuple[str, bytes]] = []
    sep = b"--" + boundary
    for part in body.split(sep):
        if not part or part in (b"--\r\n", b"--", b"\r\n"):
            continue
        head, _, data = part.partition(b"\r\n\r\n")
        if not _:
            continue
        data = data[:-2] if data.endswith(b"\r\n") else data
        disp = ""
        for line in head.split(b"\r\n"):
            if line.lower().startswith(b"content-disposition"):
                disp = line.decode("utf-8", "replace")
        if not disp:
            continue
        name = _param(disp, "name")
        filename = _param(disp, "filename")
        if filename:
            if data:
                files.append((Path(filename.replace("\\", "/")).name, data))
        elif name:
            fields[name] = data.decode("utf-8", "replace")
    return fields, files


def _param(disposition: str, key: str) -> str:
    for chunk in disposition.split(";"):
        chunk = chunk.strip()
        if chunk.lower().startswith(key.lower() + "="):
            return chunk.split("=", 1)[1].strip().strip('"')
    return ""


# ------------------------------------------------------------- chrome

def page(title: str, body: str, subtitle: str = "") -> bytes:
    return (f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)} - auditbot</title><style>{dashboard.CSS}{EXTRA_CSS}</style></head>
<body class="viz-root"><div class="wrap">
<header class="top"><div><h1>{esc(title)}</h1>
<p class="meta">{esc(subtitle)}</p></div>
<button class="theme" type="button">Dark mode</button></header>
{body}
<footer>auditbot · running locally on this machine · nothing leaves it</footer>
</div><div id="tip" role="status"></div><script>{dashboard.JS}</script></body></html>"""
            ).encode("utf-8")


EXTRA_CSS = """
.drop{border:2px dashed var(--axis);border-radius:16px;padding:52px 24px;text-align:center;
  transition:.15s;background:var(--surface-1);cursor:pointer}
.drop.over{border-color:var(--series-1);background:var(--plane)}
.drop h3{margin:0 0 6px;font-size:18px}
.drop p{margin:0;color:var(--text-secondary);font-size:13.5px}
.drop input{display:none}
.btn{background:var(--text-primary);color:var(--surface-1);border:0;border-radius:9px;
  padding:11px 22px;font:inherit;font-size:14px;font-weight:600;cursor:pointer}
.btn:disabled{opacity:.45;cursor:not-allowed}
.btn.ghost{background:var(--surface-1);color:var(--text-primary);
  border:1px solid var(--ring);font-weight:500}
.row{display:flex;gap:14px;flex-wrap:wrap;align-items:flex-end;margin:18px 0}
label.f{display:flex;flex-direction:column;gap:6px;font-size:12px;color:var(--muted);
  text-transform:uppercase;letter-spacing:.05em}
label.f input,label.f select{font:inherit;font-size:14px;padding:9px 11px;border-radius:9px;
  border:1px solid var(--ring);background:var(--surface-1);color:var(--text-primary);
  text-transform:none;letter-spacing:0}
select.src{font:inherit;font-size:13px;padding:6px 9px;border-radius:8px;
  border:1px solid var(--ring);background:var(--surface-1);color:var(--text-primary);width:100%}
select.src.unknown{border-color:var(--critical)}
.pill{display:inline-block;border-radius:999px;padding:2px 9px;font-size:11px;font-weight:650;
  border:1px solid var(--ring)}
.pill.hi{color:var(--good)} .pill.mid{color:var(--warning)} .pill.lo{color:var(--critical)}
#filelist{margin-top:16px;font-size:13.5px;color:var(--text-secondary)}
#filelist div{padding:4px 0;border-bottom:1px solid var(--grid)}
.spin{display:inline-block;width:15px;height:15px;border:2px solid var(--ring);
  border-top-color:var(--text-primary);border-radius:50%;animation:sp .7s linear infinite;
  vertical-align:-3px;margin-right:8px}
@keyframes sp{to{transform:rotate(360deg)}}
.dl{display:flex;gap:10px;flex-wrap:wrap;margin-top:14px}
.dl a{text-decoration:none}
.note{background:var(--plane);border-left:3px solid var(--series-1);padding:12px 16px;
  border-radius:0 9px 9px 0;font-size:13.5px;color:var(--text-secondary);margin:14px 0}
iframe.dash{width:100%;height:78vh;border:1px solid var(--ring);border-radius:12px;
  background:var(--surface-1)}
"""


# -------------------------------------------------------------- pages

def upload_page() -> bytes:
    fy_now = datetime.now().year - (0 if datetime.now().month >= 4 else 1)
    body = f"""
<section>
<h2>1. Tell me about the client</h2>
<p class="sub">Only the period matters to the checks; the name just labels the reports.</p>
<form id="f" method="post" action="/upload" enctype="multipart/form-data">
  <div class="row">
    <label class="f">Client name
      <input name="client" value="Test Client" size="34" required></label>
    <label class="f">Financial year
      <select name="fy">
        {"".join(f'<option value="{y}-{str(y + 1)[2:]}"{" selected" if y == fy_now else ""}>'
                 f'{y}-{str(y + 1)[2:]}</option>' for y in range(fy_now - 5, fy_now + 2))}
      </select></label>
    <label class="f">Home state code
      <input name="state" value="33" size="4" maxlength="2"
             title="GST state code, e.g. 33 Tamil Nadu, 27 Maharashtra, 29 Karnataka"></label>
  </div>

  <h2 style="margin-top:26px">2. Drop the files in</h2>
  <p class="sub">Any format (CSV, Excel, PDF), any filename, any order, subfolders fine.
     Send whatever you have &mdash; missing documents are simply skipped.</p>
  <div class="drop" id="drop" tabindex="0" role="button"
       aria-label="Choose files or drop them here">
    <h3>Drop files here</h3>
    <p>or click to choose &mdash; bank statements, Tally exports, GSTR-2B, payroll, stock</p>
    <input type="file" id="input" name="files" multiple
           accept=".csv,.xlsx,.xls,.xlsm,.pdf,.txt,.tsv">
  </div>
  <div id="filelist"></div>
  <div class="row"><button class="btn" id="go" type="submit" disabled>
    Identify the files</button></div>
</form>
</section>

<section>
<h2>What it looks for</h2>
<p class="sub">Fifteen document types. It reads the column headings, not the filename.</p>
<table class="tbl"><thead><tr><th>Document</th><th>What it unlocks</th></tr></thead><tbody>
<tr><td>Bank statement + bank ledger</td><td>Bank reconciliation, unrecorded charges, stale cheques</td></tr>
<tr><td>Sales &amp; purchase registers</td><td>Turnover, invoice series, party control accounts</td></tr>
<tr><td>GSTR-2B</td><td>ITC at risk, unclaimed ITC, per-invoice value drift</td></tr>
<tr><td>GSTR-1 / GSTR-3B summary</td><td>Books vs returns, CGST/SGST payment tracking</td></tr>
<tr><td>Receipts &amp; payments</td><td>Bill-wise knock-off, cash limits, duplicate payments</td></tr>
<tr><td>Debtors / creditors outstanding</td><td>Ageing, overdue, Rule 37 ITC reversal</td></tr>
<tr><td>Salary register</td><td>PF/ESI, net pay arithmetic, ghost employees</td></tr>
<tr><td>Stock summary + production</td><td>Negative stock, movement variance, yield</td></tr>
<tr><td>Cash book</td><td>Cash above the s.40A(3) limit, negative cash</td></tr>
</tbody></table>
</section>

<script>
(function(){{
  var drop=document.getElementById('drop'), input=document.getElementById('input'),
      list=document.getElementById('filelist'), go=document.getElementById('go'),
      form=document.getElementById('f');
  function show(){{
    var fs=input.files; list.innerHTML='';
    if(!fs.length){{go.disabled=true;return;}}
    var total=0;
    for(var i=0;i<fs.length;i++){{
      total+=fs[i].size;
      var d=document.createElement('div');
      d.textContent=fs[i].name+'  ('+(fs[i].size/1024).toFixed(0)+' KB)';
      list.appendChild(d);
    }}
    var s=document.createElement('div');
    s.innerHTML='<b>'+fs.length+' file(s), '+(total/1048576).toFixed(1)+' MB</b>';
    list.appendChild(s); go.disabled=false;
  }}
  drop.addEventListener('click',function(){{input.click();}});
  drop.addEventListener('keydown',function(e){{
    if(e.key==='Enter'||e.key===' '){{e.preventDefault();input.click();}} }});
  input.addEventListener('change',show);
  ['dragenter','dragover'].forEach(function(t){{
    drop.addEventListener(t,function(e){{e.preventDefault();drop.classList.add('over');}});}});
  ['dragleave','drop'].forEach(function(t){{
    drop.addEventListener(t,function(e){{e.preventDefault();drop.classList.remove('over');}});}});
  drop.addEventListener('drop',function(e){{input.files=e.dataTransfer.files;show();}});
  form.addEventListener('submit',function(){{
    go.disabled=true; go.innerHTML='<span class="spin"></span>Reading the files...';}});
}})();
</script>"""
    return page("Audit automation", body, "Drop a client's files in and see what comes out")


def review_page(run_id: str) -> bytes:
    st = STATE[run_id]
    dets = st["detections"]
    options = [("", "-- ignore this file --")] + [
        (s, detect.LABEL[s]) for s in detect.SOURCE_SECTION]

    rows = []
    for i, d in enumerate(dets):
        conf = d.confidence
        cls = "hi" if conf >= 0.8 else ("mid" if conf >= 0.4 else "lo")
        sel = "".join(
            f'<option value="{esc(v)}"{" selected" if v == (d.source or "") else ""}>'
            f"{esc(lbl)}</option>" for v, lbl in options)
        rows.append(
            f"<tr><td>{esc(d.path.name)}</td>"
            f'<td><select class="src{"" if d.source else " unknown"}" '
            f'name="src_{i}">{sel}</select></td>'
            f'<td><span class="pill {cls}">{conf:.0%}</span></td>'
            f'<td class="n">{d.rows:,}</td>'
            f'<td style="font-size:12.5px;color:var(--text-secondary)">{esc(d.why)}</td></tr>')

    known = sum(1 for d in dets if d.source)
    unknown = len(dets) - known
    warn = (f'<div class="note"><b>{unknown} file(s) were not recognised.</b> '
            f"Pick what they are from the dropdown, or leave them ignored. If a file "
            f"should have been recognised, tell me its column headings and I will add "
            f"them to the alias map.</div>") if unknown else ""

    body = f"""
<section>
<h2>This is what I think you sent</h2>
<p class="sub">{known} of {len(dets)} files identified from their column headings.
   Change anything I got wrong &mdash; the dropdown wins.</p>
{warn}
<form method="post" action="/run">
<input type="hidden" name="run" value="{esc(run_id)}">
<table class="tbl"><thead><tr>
<th>File</th><th style="width:230px">Read as</th><th>Confidence</th>
<th class="n">Rows</th><th>Why</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table>
<div class="row">
  <button class="btn" type="submit" id="go">Run the reconciliation</button>
  <a class="btn ghost" href="/" style="text-decoration:none;display:inline-block">Start over</a>
</div>
</form>
</section>
<script>
document.querySelector('form').addEventListener('submit',function(){{
  var b=document.getElementById('go');
  b.disabled=true;b.innerHTML='<span class="spin"></span>Reconciling...';}});
</script>"""
    return page("Check the file types", body,
                f"{st['client']} · FY {st['fy']}")


def result_page(run_id: str) -> bytes:
    st = STATE[run_id]
    k = st["kpis"]
    reg_n, high = k["exceptions_total"], k["exceptions_high"]

    tiles = "".join(
        f'<div class="kpi"><p class="k">{esc(t)}</p><div class="v">{esc(v)}</div>'
        f'<div class="n">{esc(n)}</div></div>'
        for t, v, n in [
            ("Exceptions", f"{reg_n:,}", f"{high} high severity"),
            ("Bank match rate", f"{k['bank_match_rate']}%",
             f"Rs. {inr(k['bank_unreconciled'])} unreconciled"),
            ("ITC at risk", f"Rs. {compact(k['itc_at_risk'])}", "not in GSTR-2B"),
            ("Debtors overdue", f"Rs. {compact(k['debtors_overdue'])}",
             f"of Rs. {compact(k['debtors'])} outstanding"),
        ])

    loaded = "".join(
        f"<tr><td>{esc(detect.LABEL.get(s, s))}</td><td class=n>{n:,}</td></tr>"
        for s, n in st["loaded"].items() if n)

    body = f"""
<section>
<h2>Done &mdash; {reg_n} exceptions from {k['rules_run']} checks</h2>
<p class="sub">Took {st['seconds']:.1f} seconds.</p>
<div class="kpis">{tiles}</div>
<div class="dl">
  <a class="btn" href="/dash?r={esc(run_id)}" target="_blank">Open the dashboard</a>
  <a class="btn ghost" href="/file?r={esc(run_id)}&amp;p=Audit_Working_Papers.xlsx">
     Working papers (.xlsx)</a>
  <a class="btn ghost" href="/file?r={esc(run_id)}&amp;p=exception_register.csv">
     Exception register (.csv)</a>
  <a class="btn ghost" href="/file?r={esc(run_id)}&amp;p=powerbi.zip">Power BI tables (.zip)</a>
  <a class="btn ghost" href="/">Test another set</a>
</div>
</section>

<section><h2>Records read</h2>
<table class="tbl"><thead><tr><th>Document</th><th class="n">Rows</th></tr></thead>
<tbody>{loaded}</tbody></table></section>

<section><h2>Dashboard</h2>
<iframe class="dash" src="/dash?r={esc(run_id)}" title="Audit dashboard"></iframe></section>"""
    return page("Results", body, f"{st['client']} · FY {st['fy']}")


def error_page(msg: str, detail: str = "") -> bytes:
    body = (f'<section><h2>That did not work</h2><p class="sub">{esc(msg)}</p>'
            + (f'<pre style="overflow:auto;font-size:12px;color:var(--text-secondary)">'
               f"{esc(detail)}</pre>" if detail else "")
            + '<div class="row"><a class="btn" href="/">Start over</a></div></section>')
    return page("Error", body)


# ------------------------------------------------------------ pipeline

def do_run(run_id: str, overrides: dict[int, str]) -> None:
    st = STATE[run_id]
    folder = Path(st["folder"])
    sources: dict[str, list[str]] = {}
    for i, d in enumerate(st["detections"]):
        source = overrides.get(i, d.source or "")
        d.source = source or None
        if source:
            sources.setdefault(source, []).append(str(d.path))

    cfg = Config.load()
    year = int(st["fy"].split("-")[0])
    cfg._d["client"].update({
        "name": st["client"], "fy": st["fy"], "state_code": st["state"],
        "period_from": f"{year}-04-01", "period_to": f"{year + 1}-03-31"})
    cfg._d["sources"] = {k: [] for k in detect.SOURCE_SECTION}
    cfg._d["sources"].update(sources)
    cfg._d.setdefault("report", {})["output_dir"] = str(folder.relative_to(ROOT))

    t0 = time.time()
    res = run_pipeline(cfg)
    report_excel.write(res, folder / "Audit_Working_Papers.xlsx")
    dashboard.write(res, folder / "dashboard.html")
    pbi = powerbi.write(res, folder / "powerbi")
    res.register.to_csv(folder / "exception_register.csv", index=False,
                        encoding="utf-8-sig")
    with zipfile.ZipFile(folder / "powerbi.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(pbi.iterdir()):
            z.write(f, f.name)

    st["kpis"] = res.kpis
    st["seconds"] = time.time() - t0
    st["loaded"] = {k: len(v) for k, v in res.data.items() if v is not None}


# ------------------------------------------------------------- handler

class Handler(BaseHTTPRequestHandler):
    server_version = "auditbot"

    def log_message(self, fmt, *args):
        log.debug("%s - %s", self.address_string(), fmt % args)

    # -- helpers
    def _send(self, body: bytes, ctype="text/html; charset=utf-8", code=200,
              headers: dict | None = None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _redirect(self, to: str):
        self.send_response(303)
        self.send_header("Location", to)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _query(self) -> dict:
        return {k: v[0] for k, v in urllib.parse.parse_qs(
            urllib.parse.urlparse(self.path).query).items()}

    def _run_folder(self, run_id: str) -> Path | None:
        st = STATE.get(run_id)
        return Path(st["folder"]) if st else None

    # -- routes
    def do_GET(self):
        route = urllib.parse.urlparse(self.path).path
        q = self._query()
        try:
            if route == "/":
                return self._send(upload_page())
            if route == "/review":
                return self._send(review_page(q["r"])) if q.get("r") in STATE \
                    else self._redirect("/")
            if route == "/result":
                return self._send(result_page(q["r"])) if q.get("r") in STATE \
                    else self._redirect("/")
            if route == "/dash":
                folder = self._run_folder(q.get("r", ""))
                if folder and (folder / "dashboard.html").exists():
                    return self._send((folder / "dashboard.html").read_bytes())
                return self._send(error_page("That run is gone."), code=404)
            if route == "/file":
                folder = self._run_folder(q.get("r", ""))
                name = Path(q.get("p", "")).name          # no traversal
                target = folder / name if folder else None
                if target and target.is_file():
                    return self._send(
                        target.read_bytes(), "application/octet-stream",
                        headers={"Content-Disposition":
                                 f'attachment; filename="{name}"'})
                return self._send(error_page("No such file in this run."), code=404)
            if route == "/health":
                return self._send(json.dumps({"ok": True, "runs": len(STATE)}).encode(),
                                  "application/json")
            self._send(error_page("Page not found."), code=404)
        except Exception as exc:                                   # pragma: no cover
            log.exception("GET %s failed", route)
            self._send(error_page(str(exc), traceback.format_exc()), code=500)

    def do_POST(self):
        route = urllib.parse.urlparse(self.path).path
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_UPLOAD:
                return self._send(error_page(
                    f"That is over the {MAX_UPLOAD // 1048576} MB limit for one "
                    f"submission. Send the big files in a second run."), code=413)
            ctype = self.headers.get("Content-Type", "")
            body = self.rfile.read(length) if length else b""

            if route == "/upload":
                boundary = ctype.split("boundary=")[-1].strip().strip('"').encode()
                fields, files = parse_multipart(body, boundary)
                if not files:
                    return self._send(error_page("No files came through."), code=400)

                run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
                folder = RUNS / run_id
                indir = folder / "input"
                indir.mkdir(parents=True, exist_ok=True)
                for name, data in files:
                    (indir / name).write_bytes(data)

                cfg = Config.load()
                dets = detect.scan(indir, cfg)
                STATE[run_id] = {
                    "folder": str(folder), "input": str(indir), "detections": dets,
                    "client": fields.get("client") or "Test Client",
                    "fy": fields.get("fy") or "2025-26",
                    "state": (fields.get("state") or "33")[:2],
                }
                return self._redirect(f"/review?r={run_id}")

            if route == "/run":
                fields = {k: v[0] for k, v in urllib.parse.parse_qs(
                    body.decode("utf-8", "replace")).items()}
                run_id = fields.get("run", "")
                if run_id not in STATE:
                    return self._redirect("/")
                overrides = {int(k[4:]): v for k, v in fields.items()
                             if k.startswith("src_")}
                do_run(run_id, overrides)
                return self._redirect(f"/result?r={run_id}")

            self._send(error_page("Page not found."), code=404)
        except Exception as exc:
            log.exception("POST %s failed", route)
            self._send(error_page(
                "Something broke while processing those files. The traceback is "
                "below - send it to me and I will fix it.",
                traceback.format_exc()), code=500)


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def serve(port: int = 8765, open_browser: bool = True) -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    httpd = Server(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}"
    print(f"\n  auditbot is running at  {url}")
    print("  Drop your files on that page. Nothing leaves this machine.")
    print("  Ctrl-C to stop.\n")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped")
    finally:
        httpd.server_close()
