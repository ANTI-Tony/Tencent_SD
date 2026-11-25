## Purpose

Add a worker RPC `reload_draft_weights(weights_path=...)` that reloads the speculative draft model's weights in place from a new checkpoint, so a drafter that keeps training while vLLM serves (online speculative decoding) can be swapped in without restarting the engine or touching the target model.

Closes #60138

- `GPUModelRunner.reload_draft_weights(weights_iterator=None, weights_path=None)` mirrors `reload_weights`: resolves the draft through `get_draft_model()`, updates `speculative_config.draft_model_config.model`, loads through the draft's model loader with `initialize_layerwise_reload` / `finalize_layerwise_reload` (in place, parameter addresses captured by CUDA graphs are preserved), and re-runs the dflash fused-buffer rebuild if the drafter has one.
- Draft submodules that are live target objects (the embedding shared by EAGLE drafters, the lm_head aliased by MTP drafters) are detached while the checkpoint loads (`_detach_target_owned_draft_modules`), so a draft checkpoint that ships those tensors cannot overwrite target storage.
- `Worker.reload_draft_weights` passthrough; Model Runner V2 delegates to V1 like `reload_weights`.
- Docs: `docs/features/speculative_decoding/eagle.md` gains a "Reloading an Updated Draft Model" section (offline `collective_rpc` and the `VLLM_SERVER_DEV_MODE` HTTP endpoint).

**Not a duplicate of #52487.** That PR reloads the draft from its *configured* path inside `reload_weights()` to fix the Level-2 sleep/wake regression (#52479). This PR adds the *new-path* swap for online-trained speculators as a separate method, leaving `reload_weights` untouched; both use `get_draft_model()` and the same "detach target-owned modules" idea, so a shared helper is a straightforward follow-up once #52487 lands.

## Test Plan

- Unit (CPU): `pytest tests/v1/worker/test_gpu_model_runner.py -k TestReloadDraftWeights`
  - new-path reload updates the draft config and loads only the draft (target `load_weights` never called);
  - `draft_load_config` is used when set; iterator path skips the loader; `ValueError` without a draft;
  - V2 runner delegates to V1;
  - a draft checkpoint that ships the shared embedding leaves the target embedding untouched and the shared module re-attached (real `nn.Module`s through `AutoWeightsLoader`).
- E2E (GPU): `pytest tests/v1/e2e/spec_decode/eagle/test_reload_draft_weights.py`
  - Qwen3-8B + `AngelSlim/Qwen3-8B_eagle3` (run locally; the Llama-3.1 EAGLE3 pair is parametrized for CI, I cannot access the gated model): generate greedily, reload a copy of the draft whose `lm_head` is zeroed, check outputs are unchanged (lossless) and acceptance length drops; reload the original path, check acceptance is restored.

## Test Result

- Unit: `4 passed in 4.75s` on main; the same tests pass against a backport of the branch onto vLLM 0.27.1.
- E2E (H100, Qwen3-8B + `AngelSlim/Qwen3-8B_eagle3`, main @ 2e3154a with the branch applied, nightly cu130 wheel): `1 passed` in 3 min — outputs byte-identical across the two reloads; acceptance length drops with the zeroed head and is restored with the original. The same test also passes against a backport of the branch onto vLLM 0.27.1 (`1 passed`).
- Serving check (vLLM 0.27.1 + backport of this branch, `vllm serve` with `VLLM_SERVER_DEV_MODE=1`, EAGLE3 k=3, 256 concurrent code-generation streams on one H100): a co-located trainer adapted the head to the new domain on ~800 served requests in 25 s and exported four heads; each was loaded through `POST /collective_rpc {"method": "reload_draft_weights", ...}` in 0.67–0.72 s while requests kept flowing. Served acceptance length rose from 1.92 to 2.65 (static head: 2.12 throughout); in steady state the server generated +24% tokens/s vs the static deployment (bench over the phase: 17629 vs 14461 output tok/s, median TPOT 13.6 vs 16.9 ms), with greedy outputs unchanged.
## Model evaluation

Greedy outputs are byte-identical before/after reload in the e2e test (verification is lossless, so only the acceptance length changes). Online adaptation of the AngelSlim Qwen3-8B EAGLE3 head on served traffic raises acceptance length 1.88→2.63 (SQL), 2.12→2.72 (code), 1.91→2.37 (chat) after 3200 requests; at concurrency 256 that moves speculation from 0.93× to 1.07–1.17× of no-speculation throughput — the benefit this RPC lets a running server pick up.

## AI assistance

This PR was developed with AI assistance (Claude); every changed line was reviewed and the tests above were run by the submitter. `pre-commit`/`ruff` pass on the changed files.
