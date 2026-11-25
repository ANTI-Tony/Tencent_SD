## Purpose

Add an opt-in live training-data tap for speculative decoding: with `VLLM_SPEC_LIVE_TAP=<path>` the GPU model runner streams, for every committed position of every request, the target token id and the EAGLE-3 auxiliary hidden states into a shared-memory ring buffer, so a co-located trainer can keep the draft model adapted to the served traffic without recomputing target forwards (and reload it with #60139).

Closes #<issue>.

## Why this is not a duplicate

- #60139 reloads draft weights at runtime; this PR produces the data to train them. They are independent and complementary.
- `method: extract_hidden_states` returns hidden states as request outputs; it is not a streaming per-position export during normal EAGLE serving and does not carry the rejected-token bookkeeping.

## Changes

- `vllm/v1/spec_decode/live_tap.py`: `Ring` (single-writer/single-reader shared-memory ring, records never straddle the wrap, lapped readers resynchronise at the oldest intact record) and `LiveTap` (pinned staging pool, non-blocking D2H on the current stream + CUDA event, background writer thread, per-request chunk records, TP rank 0 only, self-disabling on error).
- `gpu_model_runner.py`: `LiveTap.from_env(self)` in `__init__`; forward `finished_req_ids` from `_update_states`; one `record()` call in the EAGLE branch of `propose_draft_token_ids` with the gathered `target_token_ids` / `target_hidden_states` and `num_rejected_tokens_gpu` (padded batch) or the compacted `query_start_loc_cpu` (unpadded batch).
- Docs: a section in `docs/features/speculative_decoding/eagle.md`.

Only the V1 GPU model runner is covered (`VLLM_USE_V2_MODEL_RUNNER=0`); the V2 runner is a follow-up.

## Test Plan

- `pytest tests/v1/worker/test_live_tap.py` — ring round trip across many wraps, lapped-reader resync, `from_env` disabled without the variable, GPU `record()` with two requests (rejected drafts excluded, prefill chunk kept, FINISH follows).
- `pytest tests/v1/e2e/spec_decode/eagle/test_live_tap.py` — Qwen3-8B + AngelSlim/Qwen3-8B_eagle3, padded and unpadded drafter batches: 12 chat prompts, every request's token sequence reconstructed from the ring (contiguous chunks, 3 × hidden aux width, FINISH markers), equal to prompt + generated tokens.

## Test Result

H100, main @ 2e3154a (dev build): unit `3 passed in 1.8s`; e2e `2 passed in 309s`.

Overhead measured with `vllm serve` (Qwen3-8B, EAGLE-3 k=3, 256 concurrent code requests, prefix caching off): output throughput 14461 → 14256 tok/s (−1.4%), TPOT p50 16.93 → 17.18 ms, while exporting 68 GB in 3 min with 1 dropped step (staging pool of 3).

## Model evaluation

Not applicable: the tap does not change serving outputs (verification is untouched); the e2e asserts that exported tokens equal the engine's outputs.

## AI assistance disclosure

Written with Claude Code (Anthropic); the design, measurements and review are the author's. Commits carry `Co-authored-by: Claude`.

---
(Branch note, 2026-10-08: the branch also carries `set_speculation`, a runtime speculation switch — scheduler half
stops scheduling draft slots, runner half stops proposing, EngineCore.collective_rpc keeps them in step; e2e
`tests/v1/e2e/spec_decode/eagle/test_set_speculation.py`. Measured on 0.27.1: switching off at 256 concurrent
requests gives +10.5% (code, stock head) / +85% (zh) throughput and brings p99 TPOT back under the SLO; switching
on restores the previous numbers. Suggest a separate issue + PR for the switch when upstreaming.)
