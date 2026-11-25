# [Feature]: Export live EAGLE-3 training data (committed tokens + aux hidden states) for online draft-model adaptation

## 🚀 The feature, motivation and pitch

EAGLE-3 drafters are trained on the target model's auxiliary hidden states, which the engine already computes for every committed token while serving. Today a team that wants to keep a drafter adapted to the traffic it serves has to re-run the target model offline on logged requests (Aurora-style pipelines do this on separate GPUs), i.e. pay the target forward twice.

Proposal: an opt-in tap in the GPU model runner (`VLLM_SPEC_LIVE_TAP=<path>`) that streams, per request and per committed position, the target token id and the aux hidden states the drafter consumes, into a shared-memory ring buffer. A co-located trainer reads the ring, updates the head and reloads it with `reload_draft_weights` (#60139). The device-to-host copy is non-blocking into a pinned staging pool; a background thread writes the ring; rejected draft positions are excluded using `num_rejected_tokens`; only TP rank 0 writes; any error disables the tap so serving is never affected.

Measured on an H100 with Qwen3-8B + AngelSlim/Qwen3-8B_eagle3 at 256 concurrent requests: the export costs about 1.4% throughput and 1.5% TPOT while writing ~390 MB/s (68 GB over a 3-minute saturated phase). Closed loop (train from the tap, hot-swap every few hundred requests): acceptance length 2.1 → 2.67 within ~60 s of a chat→code traffic shift, +17–18% throughput at saturation and −27% TPOT on the following chat traffic.

Implementation on a fork branch: https://github.com/ANTI-Tony/vllm/tree/feat/spec-live-tap (unit tests + e2e pass on main with the V1 model runner).

## Alternatives

- Log requests and recompute hidden states offline (double target forward, needs extra GPUs).
- `extract_hidden_states` speculative method exports hidden states as outputs, but only as a sampling-time API for the request, not a streaming per-position export usable by a trainer during serving.

## Additional context

Complements #60139 (runtime reload of draft weights). V2 model runner support would be a follow-up.
