"""Scan every saturated phase (B) for throughput drops that no controlled variable explains.

Analysis only (not part of the frozen serving code). The rule was fixed on 2026-10-26 after one TIDE-style run
(m2_code.tide.r2, node c0901) showed a step-down, and before the other runs were scanned:

  clean sample   a 5-s timeline sample in phase B with >= 90 % of the B concurrency running, the trainer idle
                 (busy_s == 0) for the preceding 10 s, and no speculation toggle and no drafter swap in that 10 s
  reference      median throughput of the 50 clean samples from OTHER runs of the same scenario and speculation
                 state whose acceptance length is closest (speculation off: acceptance is not used)
  flag           some 60-s stretch (12 consecutive clean samples) has median throughput / reference < 0.85

Usage: scan_anomaly.py OUT.csv REF_ROOT [EXTRA_ROOT ...]
  REF_ROOT    runs that define the reference (data/runs/abl: frozen matrix + TIDE-style runs)
  EXTRA_ROOT  runs scored against that reference but not added to it (anomaly folder, environment check)
"""
import csv
import glob
import json
import os
import statistics
import sys

SPEC_ON_FIXED = {"vanilla": 1, "static": 1, "fixed": 1, "fixed800": 1, "nospec": 0}
WIN, NEED, THR, K = 10.0, 12, 0.85, 50


def jl(p):
    if not os.path.exists(p):
        return []
    out = []
    for line in open(p):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def load(rundir):
    parts = rundir.rstrip("/").split("/")
    scen, strat, rep = parts[-3], parts[-2], parts[-1]
    ev = jl(f"{rundir}/events.jsonl")
    tb = [e for e in ev if e.get("event") == "phase" and e.get("name") == "B"]
    te = [e for e in ev if e.get("event") == "phase_end" and e.get("name") == "B"]
    if not tb or not te:
        return None
    t0, t1, cb = tb[0]["t"], te[0]["t"], tb[0].get("C", 256)
    tl = [r for r in jl(f"{rundir}/timeline.jsonl") if t0 < r["t"] <= t1]
    busy = [(r["t"], r.get("busy_s", 0.0) or 0.0) for r in jl(f"{rundir}/trainer.jsonl") if "busy_s" in r]
    ctl = [r for r in jl(f"{rundir}/controller.jsonl") if "at" in r and "spec" in r]
    swaps = [e["t"] for e in ev if e.get("event") == "swap"]
    prev = None
    for r in ctl:                                     # controller-driven swaps show up as a counter step
        s = r.get("swapped")
        if prev is not None and s is not None and s != prev:
            swaps.append(r["at"])
        prev = s if s is not None else prev
    samples = []
    for r in tl:
        t = r["t"]
        lo = t - WIN
        b = sum(x for (tt, x) in busy if lo < tt <= t + 0.5)
        if strat in SPEC_ON_FIXED:
            spec_states = {SPEC_ON_FIXED[strat]}
        else:
            spec_states = {int(bool(c["spec"])) for c in ctl if lo < c["at"] <= t}
        if len(spec_states) != 1:
            continue
        spec = spec_states.pop()
        clean = (r.get("running", 0) >= 0.9 * cb and b == 0 and not any(lo < s <= t for s in swaps)
                 and (spec == 0 or r.get("acc_len") is not None))
        samples.append(dict(t=t - t0, thr=r["gen_tok_s"], acc=r.get("acc_len"), spec=spec, clean=clean,
                            running=r.get("running"), preempt=r.get("preempt")))
    return dict(dir=rundir, scen=scen, strat=strat, rep=rep, samples=samples, dur=t1 - t0)


def main():
    out, ref_root, extra = sys.argv[1], sys.argv[2], sys.argv[3:]
    ref_runs = [load(d) for d in sorted(glob.glob(f"{ref_root}/*/*/r*")) if os.path.isdir(d)]
    ref_runs = [r for r in ref_runs if r]
    ext_runs = []
    for root in extra:
        ext_runs += [r for r in (load(d) for d in sorted(glob.glob(f"{root}/*/*/r*")) if os.path.isdir(d)) if r]
    pool = {}
    for r in ref_runs:
        for s in r["samples"]:
            if s["clean"]:
                pool.setdefault((r["scen"], s["spec"]), []).append((r["dir"], s["acc"], s["thr"]))
    rows = []
    for r in ref_runs + ext_runs:
        ratios = []
        for s in r["samples"]:
            if not s["clean"]:
                ratios.append(None)
                continue
            cand = [(a, th) for (d, a, th) in pool.get((r["scen"], s["spec"]), []) if d != r["dir"]]
            if not cand:
                ratios.append(None)
                continue
            if s["spec"]:
                cand.sort(key=lambda x: abs(x[0] - s["acc"]))
                cand = cand[:K]
            ref = statistics.median(th for (_, th) in cand)
            ratios.append(s["thr"] / ref if ref > 0 else None)
        clean_r = [x for x in ratios if x is not None]
        worst, worst_t = None, None
        seq = [(s["t"], x) for s, x in zip(r["samples"], ratios) if x is not None]
        for i in range(len(seq) - NEED + 1):
            m = statistics.median(x for (_, x) in seq[i:i + NEED])
            if worst is None or m < worst:
                worst, worst_t = m, seq[i][0]
        rows.append(dict(scenario=r["scen"], strategy=r["strat"], rep=r["rep"],
                         root=os.path.relpath(os.path.dirname(os.path.dirname(os.path.dirname(r["dir"]))),
                                              os.path.dirname(ref_root.rstrip("/"))),
                         b_seconds=round(r["dur"], 1), n_clean=len(clean_r),
                         median_ratio=round(statistics.median(clean_r), 3) if clean_r else "",
                         worst60_ratio=round(worst, 3) if worst is not None else "",
                         worst60_start_s=round(worst_t, 1) if worst_t is not None else "",
                         flagged=int(worst is not None and worst < THR)))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    scored = [x for x in rows if x["worst60_ratio"] != ""]
    print(f"{len(rows)} runs, {len(scored)} with >= {NEED} clean samples in a row; flagged: "
          f"{sum(x['flagged'] for x in rows)}")
    for x in sorted(scored, key=lambda x: x["worst60_ratio"])[:12]:
        print(f"  {x['root']:>14} {x['scenario']:8} {x['strategy']:9} {x['rep']}  n_clean={x['n_clean']:3d}  "
              f"median={x['median_ratio']}  worst60={x['worst60_ratio']} at B+{x['worst60_start_s']}s"
              f"{'  FLAG' if x['flagged'] else ''}")
    no = [x for x in rows if x["worst60_ratio"] == ""]
    if no:
        print(f"  not scorable (< {NEED} clean samples): " + ", ".join(f"{x['scenario']}.{x['strategy']}.{x['rep']}"
                                                                   for x in no))


if __name__ == "__main__":
    main()
