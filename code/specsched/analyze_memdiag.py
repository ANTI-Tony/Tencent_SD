"""Open-loop GPU-memory diagnosis (runs/abl_arm_diag; not part of E28, produces no comparison data).

Reads, per run: the allocator counters each process logged about itself (memprobe_<role>_<pid>.jsonl), the driver's
per-process view (gpumem_apps_<job>.csv, mapped to roles through gpumem_ps_<job>.txt), the server log's allocator
warnings, the client's per-request errors, and the trainer's log. Judged by the reading rule written in
ideas/2026-11-24_同起点对照与累计回本_预注册.md before the runs.

  python analyze_memdiag.py <runs/abl_arm_diag> <out dir> [<smoke run dir> ...]
"""
import datetime
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

GIB = 2.0 ** 30


def jl(path):
    out = []
    if not os.path.exists(path):
        return out
    for line in open(path, errors="replace"):
        line = line.strip()
        if line.startswith("{"):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def naive_epoch(stamp, fmt):
    """Node-local wall-clock string read as if it were UTC."""
    return datetime.datetime.strptime(stamp, fmt).replace(tzinfo=datetime.UTC).timestamp()


def clock_offset(path):
    """epoch = naive_epoch(local stamp) + offset, from the sampler's own clock file."""
    ep, day, hms, _host = open(path).read().split()
    return float(ep) - naive_epoch(f"{day} {hms[:15]}", "%Y/%m/%d %H:%M:%S.%f")


def load_samplers(root):
    """Driver-side samples of both jobs: per-process MiB and whole-card MiB on an epoch axis, and pid -> command."""
    apps, gpu, cmd = [], [], {}
    for cf in sorted(glob.glob(f"{root}/gpumem_clock_*.txt")):
        job = cf.split("_")[-1].split(".")[0]
        off = clock_offset(cf)
        for line in open(f"{root}/gpumem_apps_{job}.csv"):
            p = [x.strip() for x in line.split(",")]
            if len(p) == 3 and p[1].isdigit():
                apps.append((naive_epoch(p[0], "%Y/%m/%d %H:%M:%S.%f") + off, int(p[1]), float(p[2]), job))
        for line in open(f"{root}/gpumem_gpu_{job}.csv"):
            p = [x.strip() for x in line.split(",")]
            if len(p) == 3 and p[1].replace(".", "").isdigit():
                gpu.append((naive_epoch(p[0], "%Y/%m/%d %H:%M:%S.%f") + off, float(p[1]), float(p[2]), job))
        for line in open(f"{root}/gpumem_ps_{job}.txt", errors="replace"):
            p = line.split(None, 2)
            if len(p) == 3 and p[0].isdigit():
                cmd[int(p[0])] = (int(p[1]), p[2].strip())
    return (pd.DataFrame(apps, columns=["t", "pid", "mib", "job"]), pd.DataFrame(gpu, columns=["t", "used", "total", "job"]),
            cmd, {os.path.basename(c).split("_")[-1].split(".")[0]: clock_offset(c) for c in glob.glob(f"{root}/gpumem_clock_*.txt")})


def probe(path):
    rec = jl(path)
    head = next((r for r in rec if r.get("event") == "start"), {})
    rows = [r for r in rec if "resv" in r]
    return head, pd.DataFrame(rows)


def server_log(path, off):
    """Allocator warnings (epoch, requested bytes, free bytes) and whether the engine died before the harness's shutdown."""
    warns, dead_early, shutdown_line, dead_line, errors = [], False, None, None, 0
    year = None
    for i, line in enumerate(open(path, errors="replace")):
        if year is None:
            m = re.search(r" (\d\d)-(\d\d) (\d\d:\d\d:\d\d) ", line)
            if m:
                year = True
        m = re.search(r"\[W(\d\d)(\d\d) (\d\d:\d\d:\d\d\.\d{6})\d* CUDACachingAllocator.*memory allocation failed.*allocate (\d+) bytes \(free: (\d+), total: (\d+)\)", line)
        if m:
            t = naive_epoch(f"2026/{m.group(1)}/{m.group(2)} {m.group(3)}", "%Y/%m/%d %H:%M:%S.%f") + off
            warns.append((t, int(m.group(4)), int(m.group(5))))
        if shutdown_line is None and "[shutdown]" in line:
            shutdown_line = i
        if dead_line is None and "EngineDeadError" in line:
            dead_line = i
        if " ERROR " in line and "async_llm.py:724" not in line:
            errors += 1
    if dead_line is not None and (shutdown_line is None or dead_line < shutdown_line):
        dead_early = True
    return warns, dead_early, errors


def run_row(d, apps, gpu, cmd, offs):
    done = json.load(open(f"{d}/done.json"))
    ev = jl(f"{d}/events.jsonl")
    hz = jl(f"{d}/harness.out")
    name = d.rstrip("/").split("/")[-2]
    scen = d.rstrip("/").split("/")[-3]
    job = str(done.get("job"))
    off = offs.get(job)
    if off is None:                                   # a smoke run outside the diagnosis jobs: no sampler clock
        off = -(-5 * 3600)                            # node clock is US Eastern standard time (UTC-5)
    tB = next(e["t"] for e in ev if e.get("event") == "phase" and e.get("name") == "B")
    tBe = next(e["t"] for e in ev if e.get("event") == "phase_end")
    tEnd = next((e["t"] for e in ev if e.get("event") == "done"), tBe)
    ctl = jl(f"{d}/controller.jsonl")
    ticks = [c for c in ctl if "event" not in c and "at" in c]
    first = next((c for c in ticks if c["at"] > tB + 2 and (c.get("gen_tok_s") or 0) > 0 and (c.get("C") or 0) > 1), None)
    t0 = first["at"] - 1.0 if first else tB
    inB = [c for c in ticks if t0 <= c["at"] < tBe]
    op = next((h for h in hz if h.get("event") == "open_loop"), None)
    res = next((h for h in hz if h.get("event") == "trainer_reserve"), None)
    b = json.load(open(f"{d}/phaseB.json"))
    a2 = json.load(open(f"{d}/phaseA2.json"))
    r = dict(run=f"{scen}/{name}", host=done.get("host", "").split(".")[0], job=job,
             arrival="closed loop C=256" if op is None else f"open loop {op['rate']:g} req/s",
             rate=None if op is None else op["rate"], reserve_gib=None if res is None else float(res["gb"]),
             n_prompts=b["num_prompts"], completed=b["completed"], failed=b["failed"],
             error_strings=sum(1 for e in b["errors"] if e), B_s=b["duration"], B_tok_s=b["output_throughput"],
             client_max_inflight=b.get("max_concurrent_requests"), A2_completed=a2["completed"], A2_failed=a2["failed"],
             ttft_p50_s=float(np.percentile(b["ttfts"], 50)), ttft_p99_s=float(np.percentile(b["ttfts"], 99)),
             max_waiting=max((c.get("waiting") or 0) for c in inB) if inB else None,
             max_running=max((c.get("C") or 0) for c in inB) if inB else None)
    # ---- server: its own allocator counters
    sp = glob.glob(f"{d}/memprobe_server_*.jsonl")
    r["server_probe_files"] = len(sp)
    series = {}
    if sp:
        head, s = probe(max(sp, key=os.path.getsize))
        r["server_pid"] = head.get("pid")
        pre = s[(s.t >= t0 - 10) & (s.t < t0)]
        b_ = s[(s.t >= t0) & (s.t < tBe)]
        r["srv_alloc_preB_gib"] = pre.alloc.max() / GIB if len(pre) else None
        r["srv_resv_preB_gib"] = pre.resv.max() / GIB if len(pre) else None
        r["srv_alloc_peak_gib"] = s.alloc_peak.max() / GIB
        r["srv_resv_peak_gib"] = s.resv_peak.max() / GIB
        r["srv_resv_minus_alloc_peak_gib"] = (s.resv_peak.max() - s.alloc_peak.max()) / GIB
        r["srv_alloc_maxB_gib"] = b_.alloc.max() / GIB if len(b_) else None
        r["srv_resv_maxB_gib"] = b_.resv.max() / GIB if len(b_) else None
        if len(b_):
            r["srv_resv_max_at_s"] = float(b_.t[b_.resv.idxmax()] - t0)
            hi = b_[b_.resv >= pre.resv.max() + 0.9 * (b_.resv.max() - pre.resv.max())] if len(pre) else b_
            r["srv_resv_90pct_growth_at_s"] = float(hi.t.iloc[0] - t0) if len(hi) else None
        r["srv_retries"] = int(s.retries.max())
        r["srv_ooms"] = int(s.ooms.max())
        r["srv_dev_alloc"] = int(s.dev_alloc.max())
        r["srv_dev_free"] = int(s.dev_free.max())
        r["srv_dev_alloc_in_B"] = int(b_.dev_alloc.max() - pre.dev_alloc.max()) if len(b_) and len(pre) else None
        r["srv_dev_free_in_B"] = int(b_.dev_free.max() - pre.dev_free.max()) if len(b_) and len(pre) else None
        series["server"] = s.assign(rel=s.t - t0)
    # ---- trainer: its own allocator counters
    tp = glob.glob(f"{d}/memprobe_trainer_*.jsonl")
    r["trainer_probe_files"] = len(tp)
    if tp:
        head, s = probe(max(tp, key=os.path.getsize))
        r["trainer_pid"] = head.get("pid")
        r["trn_alloc_peak_gib"] = s.alloc_peak.max() / GIB
        r["trn_resv_peak_gib"] = s.resv_peak.max() / GIB
        r["trn_retries"] = int(s.retries.max())
        r["trn_ooms"] = int(s.ooms.max())
        r["trn_dev_alloc"] = int(s.dev_alloc.max())
        b_ = s[(s.t >= t0) & (s.t < tBe)]
        pre = s[s.t < t0]
        r["trn_dev_alloc_in_B"] = int(b_.dev_alloc.max() - pre.dev_alloc.max()) if len(b_) and len(pre) else None
        r["trn_last_probe_s"] = float(s.t.max() - t0)
        series["trainer"] = s.assign(rel=s.t - t0)
    # ---- trainer log
    tr = jl(f"{d}/trainer.jsonl")
    tout = open(f"{d}/trainer.out", errors="replace").read()
    r["trainer_stop_event"] = any(w.get("event") == "stop" for w in tr)
    r["trainer_traceback"] = "Traceback" in tout
    r["trainer_oom"] = "OutOfMemoryError" in tout
    m = re.search(r"RESERVED ([\d.]+) GiB", tout)
    r["trainer_reserved_line_gib"] = float(m.group(1)) if m else None
    win = [w for w in tr if "busy_s" in w]
    r["trainer_last_window_s"] = float(win[-1]["t"] - t0) if win else None
    r["trainer_alive_to_end"] = bool(win) and win[-1]["t"] >= tBe - 5
    r["trainer_consumed"] = win[-1]["consumed"] if win else None
    r["trainer_peak_log_gb"] = max((w.get("peak_mem_gb") or 0) for w in win) if win else None
    inw = [w for w in win if t0 <= w["t"] < tBe]
    r["train_busy_frac_B"] = float(sum(w["busy_s"] for w in inw) / max(sum(w["win_s"] for w in inw), 1e-9)) if inw else None
    # ---- server log
    if os.path.exists(f"{d}/server.log"):
        warns, dead_early, errors = server_log(f"{d}/server.log", off)
        r["srv_alloc_warnings"] = len(warns)
        r["srv_warn_at_s"] = ";".join(f"{t - t0:.1f}" for t, _, _ in warns)
        r["srv_warn_req_mib"] = ";".join(f"{n / 2 ** 20:.0f}" for _, n, _ in warns)
        r["srv_warn_free_mib"] = ";".join(f"{f / 2 ** 20:.0f}" for _, _, f in warns)
        r["server_died_before_shutdown"] = dead_early
        r["srv_other_error_lines"] = errors
    # ---- the driver's view
    if len(apps):
        a = apps[(apps.t >= tB - 120) & (apps.t <= tEnd + 5)]
        b_ = a[(a.t >= t0) & (a.t < tBe)]
        tot = b_.groupby("t").mib.sum()
        r["drv_sum_max_gib"] = tot.max() / 1024 if len(tot) else None
        for role, pid in (("server", r.get("server_pid")), ("trainer", r.get("trainer_pid"))):
            x = b_[b_.pid == pid]
            r[f"drv_{role}_max_gib"] = x.mib.max() / 1024 if len(x) else None
            y = a[(a.pid == pid) & (a.t < t0) & (a.t >= t0 - 10)]
            r[f"drv_{role}_preB_gib"] = y.mib.max() / 1024 if len(y) else None
            series[f"drv_{role}"] = a[a.pid == pid].assign(rel=a.t - t0)
        g = gpu[(gpu.t >= t0) & (gpu.t < tBe)]
        r["drv_card_used_max_gib"] = g.used.max() / 1024 if len(g) else None
        r["drv_card_total_gib"] = g.total.max() / 1024 if len(g) else None
    r["t0"], r["tBe"] = t0, tBe
    return r, series, inB


def main(root, out, extra):
    os.makedirs(out, exist_ok=True)
    apps, gpu, cmd, offs = load_samplers(root)
    rows, long = [], []
    dirs = sorted(d for d in glob.glob(f"{root}/*/*/r*") if os.path.exists(f"{d}/done.json")) + list(extra)
    for d in dirs:
        r, series, inB = run_row(d, apps, gpu, cmd, offs)
        if d in extra:
            r["run"] = "smoke:" + r["run"]
        rows.append(r)
        for name, s in series.items():
            col = "mib" if name.startswith("drv") else None
            s = s[(s.rel >= -30) & (s.rel <= (r["tBe"] - r["t0"]) + 10)]
            for x in s.itertuples():
                if col:
                    long.append(dict(run=r["run"], series=name, rel=round(x.rel, 2), gib=x.mib / 1024))
                else:
                    long.append(dict(run=r["run"], series=name + "_alloc", rel=round(x.rel, 2), gib=x.alloc / GIB))
                    long.append(dict(run=r["run"], series=name + "_resv", rel=round(x.rel, 2), gib=x.resv / GIB))
        for c in inB:
            long.append(dict(run=r["run"], series="running", rel=round(c["at"] - r["t0"], 2), gib=c.get("C") or 0))
            long.append(dict(run=r["run"], series="waiting", rel=round(c["at"] - r["t0"], 2), gib=c.get("waiting") or 0))
    df = pd.DataFrame(rows)
    df.to_csv(f"{out}/memdiag_runs.csv", index=False)
    pd.DataFrame(long).to_csv(f"{out}/memdiag_series.csv", index=False)
    with pd.option_context("display.width", 260, "display.max_columns", 60, "display.max_rows", 100):
        f2 = lambda x: f"{x:.2f}"
        print("== requests and processes")
        print(df[["run", "host", "arrival", "reserve_gib", "n_prompts", "completed", "failed", "error_strings", "A2_completed",
                  "A2_failed", "max_running", "max_waiting", "client_max_inflight", "B_s", "B_tok_s", "ttft_p50_s", "ttft_p99_s"]]
              .to_string(index=False, float_format=f2))
        print("\n== server process: its own allocator (GiB)")
        cols = [c for c in ["run", "srv_alloc_preB_gib", "srv_resv_preB_gib", "srv_alloc_peak_gib", "srv_resv_peak_gib",
                            "srv_resv_minus_alloc_peak_gib", "srv_resv_max_at_s", "srv_resv_90pct_growth_at_s", "srv_retries",
                            "srv_ooms", "srv_dev_alloc_in_B", "srv_dev_free_in_B"] if c in df]
        print(df[cols].to_string(index=False, float_format=f2))
        print("\n== server log and the driver's view (GiB)")
        cols = [c for c in ["run", "srv_alloc_warnings", "srv_warn_at_s", "srv_warn_req_mib", "srv_warn_free_mib",
                            "server_died_before_shutdown", "srv_other_error_lines", "drv_server_preB_gib", "drv_server_max_gib",
                            "drv_trainer_max_gib", "drv_sum_max_gib", "drv_card_used_max_gib", "drv_card_total_gib"] if c in df]
        print(df[cols].to_string(index=False, float_format=f2))
        print("\n== trainer")
        cols = [c for c in ["run", "reserve_gib", "trainer_reserved_line_gib", "trn_alloc_peak_gib", "trn_resv_peak_gib",
                            "trn_retries", "trn_ooms", "trn_dev_alloc", "trn_dev_alloc_in_B", "trainer_stop_event",
                            "trainer_traceback", "trainer_oom", "trainer_alive_to_end", "trainer_last_window_s",
                            "trainer_consumed", "train_busy_frac_B"] if c in df]
        print(df[cols].to_string(index=False, float_format=f2))
    return df


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
