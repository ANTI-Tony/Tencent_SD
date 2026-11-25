# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Swap draft weights in a running engine: outputs stay those of the target
(verification is lossless) while the acceptance length tracks the draft quality."""

import json
import os
import shutil
from pathlib import Path

import pytest
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

from vllm import LLM, SamplingParams



def _spec_decode_counters(llm: LLM) -> dict[str, float]:
    counters: dict[str, float] = {}
    for metric in llm.get_metrics():
        if "spec_decode" in metric.name and hasattr(metric, "value"):
            counters[metric.name] = counters.get(metric.name, 0.0) + metric.value
    return counters


def _mean_acceptance_length(llm: LLM, prompts, sampling_params) -> tuple[float, list]:
    before = _spec_decode_counters(llm)
    outputs = llm.chat(prompts, sampling_params)
    after = _spec_decode_counters(llm)
    delta = {k: after.get(k, 0.0) - before.get(k, 0.0) for k in after}
    drafts = sum(v for k, v in delta.items() if k.endswith("spec_decode_num_drafts"))
    accepted = sum(
        v for k, v in delta.items() if k.endswith("spec_decode_num_accepted_tokens")
    )
    assert drafts > 0
    return 1.0 + accepted / drafts, [o.outputs[0].token_ids for o in outputs]


def _degraded_draft_checkpoint(draft_model: str, out_dir: Path) -> Path:
    """Copy the draft checkpoint with its lm_head zeroed (the drafter then proposes
    the same token everywhere, so almost nothing is accepted)."""
    src = Path(draft_model) if Path(draft_model).exists() else Path(snapshot_download(draft_model))
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "generation_config.json"):
        if (src / name).exists():
            shutil.copy(src / name, out_dir / name)
    tensors: dict[str, torch.Tensor] = {}
    for f in sorted(src.glob("*.safetensors")):
        tensors.update(load_file(str(f)))
    for f in sorted(src.glob("*.bin")):
        tensors.update(torch.load(str(f), map_location="cpu", weights_only=True))
    head_keys = [k for k in tensors if k.endswith("lm_head.weight")]
    assert head_keys, f"no lm_head in {draft_model}: {list(tensors)[:5]}"
    for k in head_keys:
        tensors[k] = torch.zeros_like(tensors[k])
    save_file(
        {k: v.contiguous() for k, v in tensors.items()},
        str(out_dir / "model.safetensors"),
        metadata={"format": "pt"},
    )
    return out_dir


@pytest.mark.parametrize(
    ("model_name", "draft_model"),
    [
        ("/fs/scratch/PCS0296/bursar/models/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218", "/fs/scratch/PCS0296/specsched/models/qwen3-8b-eagle3"),
    ],
    ids=["qwen3_eagle3"],
)
def test_reload_draft_weights(tmp_path: Path, model_name: str, draft_model: str):
    prompts = [[{"role": "user", "content": p}] for p in [
        "Write a Python function that reverses a string.", "Explain what a hash map is.",
        "Give three tips for writing clear documentation.", "What is the capital of France?",
        "Summarize the plot of Romeo and Juliet in two sentences.", "How does binary search work?",
        "List five fruits that are red.", "Write a SQL query that counts rows in a table named users.",
        "What causes the seasons on Earth?", "Describe the water cycle.", "Convert 100 Fahrenheit to Celsius.",
        "What is recursion? Give an example.", "Name three sorting algorithms.", "Why is the sky blue?",
        "Write a haiku about autumn.", "What is the difference between TCP and UDP?"]]
    sampling_params = SamplingParams(temperature=0, max_tokens=64)
    llm = LLM(
        model=model_name,
        speculative_config={
            "method": "eagle3",
            "model": draft_model,
            "num_speculative_tokens": 3,
            "max_model_len": 2048,
        },
        max_model_len=2048,
        tensor_parallel_size=int(os.environ.get("TP", "1")),
        gpu_memory_utilization=0.8,
        disable_log_stats=False,
    )
    try:
        acc_original, ref_outputs = _mean_acceptance_length(
            llm, prompts, sampling_params
        )
        assert acc_original > 1.5

        degraded = _degraded_draft_checkpoint(draft_model, tmp_path / "degraded")
        llm.collective_rpc(
            "reload_draft_weights", kwargs={"weights_path": str(degraded)}
        )
        acc_degraded, outputs = _mean_acceptance_length(llm, prompts, sampling_params)
        assert outputs == ref_outputs, "reloading the draft changed target outputs"
        assert acc_degraded < acc_original - 0.5

        llm.collective_rpc("reload_draft_weights", kwargs={"weights_path": draft_model})
        acc_restored, outputs = _mean_acceptance_length(llm, prompts, sampling_params)
        assert outputs == ref_outputs
        assert abs(acc_restored - acc_original) < 0.1
    finally:
        del llm
        json.dump(
            {"original": acc_original, "degraded": acc_degraded},
            open(tmp_path / "acceptance.json", "w"),
        )
