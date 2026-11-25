"""Backport a runtime speculation switch into an installed vLLM 0.27.x (anchor-based, idempotent).

Usage: python apply_spec_switch_backport.py <site-packages/vllm>

Adds `GPUModelRunner.set_speculation(enabled)` (+ Worker passthrough, reachable through the dev
`POST /collective_rpc`): when speculation is switched off the runner proposes no draft tokens, so the
scheduler falls back to plain one-token-per-step decoding from the next step on; the drafter and its weights
stay loaded and switching back on takes effect at the next step. While off, the live tap keeps exporting the
target's committed tokens / aux hidden states so a co-located trainer can still learn.
Requires the live-tap backport (apply_live_tap_backport.py) to have been applied first.
"""
import os
import shutil
import sys

RUNNER_EDITS = [
    ("        self.live_tap: LiveTap | None = LiveTap.from_env(self, log=logger.info)\n",
     "        # Runtime speculation switch (set_speculation): False -> no drafts are proposed.\n"
     "        self.spec_enabled = True\n", "after"),
    ("        spec_config = self.speculative_config\n"
     "        draft_after_bookkeeping = False\n"
     "        if spec_config is not None:\n",
     "        if spec_config is not None and not self.spec_enabled:\n"
     "            # Speculation switched off at runtime: propose no drafts (the\n"
     "            # scheduler then decodes plainly). Keep the live tap fed from the\n"
     "            # target's outputs, only for steps without drafts (all positions committed).\n"
     "            if (\n"
     "                self.live_tap is not None\n"
     "                and aux_hidden_states is not None\n"
     "                and spec_decode_metadata is None\n"
     "            ):\n"
     "                n = scheduler_output.total_num_scheduled_tokens\n"
     "                self.live_tap.record(\n"
     "                    scheduler_output,\n"
     "                    self.input_batch,\n"
     "                    self.requests,\n"
     "                    self.input_ids.gpu[:n],\n"
     "                    torch.cat([h[:n] for h in aux_hidden_states], dim=-1),\n"
     "                )\n"
     "            # Same bookkeeping as when the input does not fit in the drafter: the\n"
     "            # valid sampled-token counts of this (possibly drafted) batch are still\n"
     "            # needed, only the proposal is skipped.\n"
     "            if (\n"
     "                spec_config.use_eagle()\n"
     "                or spec_config.uses_draft_model()\n"
     "                or spec_config.uses_extract_hidden_states()\n"
     "            ) and not spec_config.disable_padded_drafter_batch:\n"
     "                next_token_ids, valid_sampled_tokens_count = (\n"
     "                    self.drafter.prepare_next_token_ids_padded(\n"
     "                        sampler_output.sampled_token_ids,\n"
     "                        self.requests,\n"
     "                        self.input_batch,\n"
     "                        self.discard_request_mask.gpu,\n"
     "                    )\n"
     "                )\n"
     "                self._copy_valid_sampled_token_count(\n"
     "                    next_token_ids, valid_sampled_tokens_count\n"
     "                )\n"
     "                # The bookkeeping expects the next token per request to be cached\n"
     "                # when the batch carried drafts (propose_draft_token_ids does it).\n"
     "                self.input_batch.prev_sampled_token_ids = next_token_ids.unsqueeze(1)\n"
     "            spec_config = None\n", "mid"),
    ("    def reload_draft_weights(\n",
     "    def set_speculation(self, enabled=\"1\") -> None:\n"
     "        \"\"\"Runtime switch for speculative decoding. Off: no drafts are proposed and the\n"
     "        scheduler decodes one token per step; the drafter stays loaded.\"\"\"\n"
     "        self.spec_enabled = str(enabled).lower() in (\"1\", \"true\", \"on\", \"yes\")\n"
     "        logger.info(\n"
     "            \"speculation %s\", \"enabled\" if self.spec_enabled else \"disabled\"\n"
     "        )\n\n", "before"),
]
WORKER_EDITS = [
    ("    def reload_draft_weights(self, *args, **kwargs) -> None:\n",
     "    def set_speculation(self, *args, **kwargs) -> None:\n"
     "        self.model_runner.set_speculation(*args, **kwargs)\n\n", "before"),
]


SCHED_EDITS = [
    ("        self.num_spec_tokens = vllm_config.num_speculative_tokens\n",
     "        # Runtime speculation switch (set_speculation): False -> no draft slots are\n"
     "        # scheduled, i.e. plain one-token-per-step decoding until switched back on.\n"
     "        self.spec_enabled = True\n", "after"),
    ("                        (self.num_spec_tokens > 0 and self.dynamic_sd_lookup is None)\n",
     "                        self.spec_enabled\n"
     "                        and (self.num_spec_tokens > 0 and self.dynamic_sd_lookup is None)\n", "replace"),
    ("        num_spec_tokens_to_schedule = self.num_spec_tokens\n",
     "        num_spec_tokens_to_schedule = self.num_spec_tokens if self.spec_enabled else 0\n", "replace"),
    ("    def get_kv_connector(self)",
     "    def set_speculation(self, enabled: bool) -> None:\n"
     "        \"\"\"Runtime switch: off = schedule no speculative tokens (plain decoding).\"\"\"\n"
     "        self.spec_enabled = bool(enabled)\n\n", "before"),
]
CORE_EDITS = [
    ("        return self.model_executor.collective_rpc(method, timeout, args, kwargs)\n",
     "        if method == \"set_speculation\":\n"
     "            # The switch has a scheduler half (stop scheduling draft slots) and a\n"
     "            # worker half (stop proposing); keep both in step.\n"
     "            enabled = (kwargs or {}).get(\"enabled\", args[0] if args else \"1\")\n"
     "            self.scheduler.set_speculation(\n"
     "                str(enabled).lower() in (\"1\", \"true\", \"on\", \"yes\")\n"
     "            )\n", "before"),
]


def patch(path, edits, marker):
    s = open(path).read()
    if marker in s:
        print("already patched:", path)
        return
    shutil.copyfile(path, path + ".orig_specswitch")
    for anchor, ins, where in edits:
        if s.count(anchor) == 0 and "reload_draft_weights" in anchor:
            anchor = anchor.replace("reload_draft_weights", "reload_weights")   # tree without the reload-draft patch
        assert s.count(anchor) == 1, f"anchor not unique/found ({s.count(anchor)}) in {path}: {anchor[:60]!r}"
        if where == "mid":   # insert between the first and second line of the anchor
            first, rest = anchor.split("\n", 1)
            s = s.replace(anchor, first + "\n" + ins + rest)
        elif where == "replace":
            s = s.replace(anchor, ins)
        else:
            s = s.replace(anchor, ins + anchor if where == "before" else anchor + ins)
    open(path, "w").write(s)
    print("patched", path)


def main(vllm_dir):
    patch(os.path.join(vllm_dir, "v1", "worker", "gpu_model_runner.py"), RUNNER_EDITS, "def set_speculation(")
    patch(os.path.join(vllm_dir, "v1", "worker", "gpu_worker.py"), WORKER_EDITS, "def set_speculation(")
    patch(os.path.join(vllm_dir, "v1", "core", "sched", "scheduler.py"), SCHED_EDITS, "def set_speculation(")
    patch(os.path.join(vllm_dir, "v1", "engine", "core.py"), CORE_EDITS, "method == \"set_speculation\"")


if __name__ == "__main__":
    main(sys.argv[1])
