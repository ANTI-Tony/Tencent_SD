#!/bin/bash
# Download the four checkpoints to the scratch HF cache on a login node (no GPU, no billing).
source /fs/scratch/PCS0296/specsched/code/weight_edit_draft/jobs/env.sh
python - <<'PY'
from huggingface_hub import snapshot_download
pats = ["*.safetensors", "*.json", "*.jinja", "*.txt", "*.model"]
for repo, rev in [("Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17"),
                  ("RikiyaT/Qwen3.5-0.8B-weight-space-programming", None),
                  ("Qwen/Qwen3.5-2B", None), ("Qwen/Qwen3.5-9B", None)]:
    p = snapshot_download(repo, revision=rev, allow_patterns=pats)
    print("ok", repo, p, flush=True)
PY
