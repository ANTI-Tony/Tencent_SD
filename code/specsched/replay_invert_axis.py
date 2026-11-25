"""Reviewer item (analysis only): the paper says the learning curve is interpolated in log(1+N), while the frozen
controller inverts it linearly in N between calibration points (controller.py invert_curve). Replay every saturated
rule-3 tick of the frozen v6 and E28a Full runs at its logged horizon with both inversions and count decision flips."""
import glob, os, sys
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import replay_arm_value as rv

LINEAR = rv.invert


def invert_log(m, dom, acc):
    pts = rv.curve(m, dom)
    if acc <= pts[0][1]:
        return 0.0
    for (n0, a0), (n1, a1) in zip(pts, pts[1:]):
        if a0 <= acc <= a1 and a1 > a0:
            w = (acc - a0) / (a1 - a0)
            return (1 + n0) ** (1 - w) * (1 + n1) ** w - 1
    return pts[-1][0]


def main():
    rows = []
    for st, arm in [("abl_arm", "full"), ("abl", "v6"), ("abl_mix", "v6")]:
        for d in sorted(glob.glob(os.path.join(rv.ROOT, "data", "runs", st, "*", arm, "r*"))):
            if not os.path.exists(f"{d}/done.json") or "_flagged" in d:
                continue
            scen = d.split("/")[-3]
            m = rv.MODELS[rv.mkey(scen)]
            for r in rv.load_ticks(d):
                if r["reason"] not in rv.RULE3:
                    continue
                rv.invert = LINEAR
                a = rv.plan(m, r, r["H"])
                rv.invert = invert_log
                b = rv.plan(m, r, r["H"])
                rv.invert = LINEAR
                rows.append(dict(set=st, scenario=scen, seed=int(d[-1]), t=r["t"], duty_log=r["duty"], duty_lin=a["duty"], duty_logaxis=b["duty"],
                                 V_lin=a["V"], V_logaxis=b["V"], need_lin=a["need_req"], need_logaxis=b["need_req"]))
    df = pd.DataFrame(rows)
    df["train_lin"], df["train_logaxis"] = df.duty_lin > 0, df.duty_logaxis > 0
    out = os.path.join(rv.ROOT, "results", "abl_arm", "value_replay")
    df.to_csv(os.path.join(out, "invert_axis_ticks.csv"), index=False)
    g = df.groupby(["set", "scenario"]).apply(lambda x: pd.Series(dict(
        ticks=len(x), train_lin=int(x.train_lin.sum()), train_logaxis=int(x.train_logaxis.sum()),
        lin_train_log_not=int((x.train_lin & ~x.train_logaxis).sum()), log_train_lin_not=int((~x.train_lin & x.train_logaxis).sum()),
        duty_changed=int((x.duty_lin != x.duty_logaxis).sum()),
        need_ratio_median=float((x.need_logaxis / x.need_lin.where(x.need_lin > 0)).median())))).reset_index()
    g.to_csv(os.path.join(out, "invert_axis_summary.csv"), index=False)
    print(g.to_string(index=False))
    print("total ticks %d; train decision differs in %d; duty differs in %d; linear replay reproduces the logged duty in %d" % (
        len(df), int((df.train_lin != df.train_logaxis).sum()), int((df.duty_lin != df.duty_logaxis).sum()),
        int((abs(df.duty_lin - df.duty_log) < 1e-9).sum())))


if __name__ == "__main__":
    main()
