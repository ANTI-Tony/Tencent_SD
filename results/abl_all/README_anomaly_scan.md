# Interference scan files

- `anomaly_scan_decision.csv`: the scan that decided which runs to rerun (2026-10-26 local time, before any rerun).
  Rule fixed in `code/specsched/scan_anomaly.py` before it was run. Reference pool = runs then in `data/runs/abl`
  (72-run frozen matrix + the TIDE-style runs finished at that time: m1_code r1/r2, m1_zh r1/r2, m2_code r1);
  `abl_anomaly` = the TIDE-style m2_code r2 run already moved aside; `abl_envcheck` = environment-check runs.
  Flagged: m2_code.tide.r2, m1_zh.simple.r3, m1_zh.switch.r2, m1_code.switch.r3, m1_code.simple.r1.
- `anomaly_scan.csv`: the same rule after the reruns (pool = main results incl. reruns). Used to check that no run
  in the main results is flagged; the originals' scores shift slightly because the pool changed.
