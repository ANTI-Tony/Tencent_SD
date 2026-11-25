"""Backport the live training-data tap into an installed vLLM 0.27.x (anchor-based, idempotent).

Usage: python apply_live_tap_backport.py <site-packages/vllm>

Installs livetap.py as vllm/v1/spec_decode/live_tap.py and patches v1/worker/gpu_model_runner.py:
  * import LiveTap
  * __init__: self.live_tap = LiveTap.from_env(self)   (None unless VLLM_SPEC_LIVE_TAP is set)
  * _update_states: forward finished request ids to the tap
  * propose_draft_token_ids, EAGLE branch: record the gathered target token ids / hidden states
The original file is kept as gpu_model_runner.py.orig_livetap.
"""
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

EDITS = [
    # (anchor, insertion, before/after)
    ("from vllm.v1.spec_decode.eagle import EagleProposer\n",
     "from vllm.v1.spec_decode.live_tap import LiveTap\n", "after"),
    ("        self.requests: dict[str, CachedRequestState] = {}\n",
     "        # Live training-data tap: None unless VLLM_SPEC_LIVE_TAP is set.\n"
     "        self.live_tap: LiveTap | None = LiveTap.from_env(self, log=logger.info)\n", "after"),
    ("        self.late_interaction_runner.on_requests_finished(\n",
     "        if self.live_tap is not None and scheduler_output.finished_req_ids:\n"
     "            self.live_tap.finish(scheduler_output.finished_req_ids)\n", "before"),
    ("            if self.supports_mm_inputs and self.drafter.supports_mm_inputs:\n"
     "                mm_embed_inputs = self._gather_mm_embeddings(\n",
     "            if self.live_tap is not None:\n"
     "                compacted = (\n"
     "                    spec_decode_metadata is not None\n"
     "                    and spec_config.disable_padded_drafter_batch\n"
     "                )\n"
     "                self.live_tap.record(\n"
     "                    scheduler_output,\n"
     "                    self.input_batch,\n"
     "                    self.requests,\n"
     "                    target_token_ids,\n"
     "                    target_hidden_states,\n"
     "                    num_rejected_tokens_gpu,\n"
     "                    spec_decode_metadata.num_draft_tokens\n"
     "                    if spec_decode_metadata is not None\n"
     "                    else None,\n"
     "                    common_attn_metadata.query_start_loc_cpu if compacted else None,\n"
     "                )\n", "before"),
]


def main(vllm_dir):
    dst = os.path.join(vllm_dir, "v1", "spec_decode", "live_tap.py")
    shutil.copyfile(os.path.join(HERE, "livetap.py"), dst)
    print("installed", dst)
    p = os.path.join(vllm_dir, "v1", "worker", "gpu_model_runner.py")
    s = open(p).read()
    if "self.live_tap" in s:
        print("already patched:", p)
        return
    shutil.copyfile(p, p + ".orig_livetap")
    for anchor, ins, where in EDITS:
        assert s.count(anchor) == 1, f"anchor not unique/found ({s.count(anchor)}): {anchor[:60]!r}"
        s = s.replace(anchor, ins + anchor if where == "before" else anchor + ins)
    open(p, "w").write(s)
    print("patched", p)


if __name__ == "__main__":
    main(sys.argv[1])
