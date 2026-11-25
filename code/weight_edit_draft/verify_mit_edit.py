#!/usr/bin/env python
"""Check that the HF checkpoint RikiyaT/Qwen3.5-0.8B-weight-space-programming differs from Qwen/Qwen3.5-0.8B in
exactly the two tensors of the p50c edit, with the sha256 values of the published manifest (raw bf16 bytes).
No torch needed: reads the safetensors headers and byte ranges directly.

  python verify_mit_edit.py --manifest data/mit_p50c_manifest.json [--base Qwen/Qwen3.5-0.8B] [--edited ...]
"""
import argparse
import glob
import hashlib
import json
import os
import struct


def tensors_of(snapshot):
    out = {}
    for fp in sorted(glob.glob(os.path.join(snapshot, "*.safetensors"))):
        with open(fp, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        for k, v in hdr.items():
            if k != "__metadata__":
                out[k] = (fp, 8 + n, v)
    return out


def sha(fp, base, v):
    o = v["data_offsets"]
    with open(fp, "rb") as f:
        f.seek(base + o[0])
        return hashlib.sha256(f.read(o[1] - o[0])).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--base", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--base-revision", default="2fc06364715b967f1860aea9cf38778875588b17")
    ap.add_argument("--edited", default="RikiyaT/Qwen3.5-0.8B-weight-space-programming")
    args = ap.parse_args()
    from huggingface_hub import snapshot_download
    pats = ["*.safetensors", "*.json"]
    sb = snapshot_download(args.base, revision=args.base_revision, allow_patterns=pats)
    se = snapshot_download(args.edited, allow_patterns=pats)
    tb, te = tensors_of(sb), tensors_of(se)
    man = json.load(open(args.manifest))
    expect = {t["name"]: t for t in man["tensors"]}
    print(f"base: {len(tb)} tensors; edited: {len(te)} tensors; only-in-base {len(set(tb) - set(te))}, only-in-edited {len(set(te) - set(tb))}")
    diff = []
    for k in sorted(set(tb) & set(te)):
        hb, he = sha(*tb[k]), sha(*te[k])
        if hb != he:
            diff.append(k)
            ok = k in expect and hb == expect[k]["base_sha256"] and he == expect[k]["edited_sha256"]
            print(f"DIFF {k} shape={te[k][2]['shape']} dtype={te[k][2]['dtype']} manifest-match={ok}")
    missing = [k for k in expect if k not in diff]
    print(f"{len(diff)} tensors differ; expected {len(expect)}; expected-but-identical: {missing}")
    print("VERDICT:", "OK" if (len(diff) == len(expect) and not missing) else "MISMATCH")


if __name__ == "__main__":
    main()
