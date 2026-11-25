### 🚀 The feature, motivation and pitch

**Feature:** a worker RPC `reload_draft_weights(weights_path=...)` that reloads the speculative draft model's weights in place from a new checkpoint, without restarting the engine and without touching the target model.

**Motivation.** Online/continual training of speculators is becoming common (Online Speculative Decoding, ICML'24; Together's Aurora/ATLAS; ReSpec/FastGRPO for RL rollouts): the drafter keeps learning from live traffic and the serving engine should pick up the updated weights every few hundred requests. Today vLLM has no way to do that: `reload_weights()` only covers the target model (the drafter lives in `self.drafter.model`), and `vllm serve` exposes `collective_rpc` but no draft-aware method. The only option is a restart (~90 s, drops in-flight requests and the KV cache).

Measured gap (Qwen3-8B + `AngelSlim/Qwen3-8B_eagle3`, H100, k=3): after on-policy updates on 100–3200 served requests the acceptance length rises from 1.88→2.63 (SQL), 2.12→2.72 (code), 1.91→2.37 (chat); at saturation that is the difference between speculation being net negative (0.93×) and net positive (1.07–1.17×). None of it can be realized in a running engine today.

**Proposed API**

```python
# offline
llm.collective_rpc("reload_draft_weights", kwargs={"weights_path": "/path/to/new-draft"})
# online (VLLM_SERVER_DEV_MODE=1)
POST /collective_rpc {"method": "reload_draft_weights", "kwargs": {"weights_path": "/path/to/new-draft"}}
```

- `GPUModelRunner.reload_draft_weights(weights_iterator=None, weights_path=None)` mirroring `reload_weights`: resolve the draft through `get_draft_model()`, update `speculative_config.draft_model_config.model`, load through the draft's model loader with `initialize_layerwise_reload` / `finalize_layerwise_reload` (in place, so CUDA-graph-captured parameter addresses are preserved).
- Draft submodules that are live target objects (the embedding shared by EAGLE drafters, the lm_head aliased by MTP drafters) are detached during the load so a draft checkpoint can never overwrite target storage.
- `Worker.reload_draft_weights` passthrough; Model Runner V2 delegates to V1 like `reload_weights` does.

**Relation to existing work.** #52479 / #52487 fix the *same-path* draft reload after Level-2 sleep/wake by reloading the draft inside `reload_weights()`. This request is the complementary *new-path* case (swapping in an updated speculator while serving); the implementation reuses the same accessors and the same "detach target-owned draft modules" idea so the two can share a helper once #52487 lands.

I have a PR ready (unit tests on the runner, an e2e test that swaps a degraded head in and out and checks acceptance/outputs, docs).

### Alternatives

- Restarting the engine on every draft update: 60–90 s outage per update, defeats online adaptation.
- Running a separate serving process per draft version behind a router: doubles GPU memory for the target.
- Extending `reload_weights(weights_path=...)` with a `target="draft"` flag: same mechanics; a separate method keeps `reload_weights` semantics (and #52487's sleep/wake path) unchanged.

### Additional context

Draft checkpoints must have the same architecture as the configured draft; omitting `weights_path` reloads from the configured path.
