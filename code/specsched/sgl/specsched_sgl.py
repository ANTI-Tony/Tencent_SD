"""SGLang 0.5.10.post1 port of the engine mechanisms of the vLLM study, applied as a runtime patch.

The vLLM study patches six files of vLLM 0.27.1 (live tap, draft reload, speculation switch). This module adds the
same three mechanisms to SGLang's default EAGLE-3 path (spec v1, overlap scheduler off) without touching the installed
package, plus the two HTTP endpoints that the frozen harness and controller talk to:

  GET  /metrics          the counters the controller and sampler read, under the names vLLM 0.27.1 uses
  POST /collective_rpc   {"method": "set_speculation" | "reload_draft_weights", "kwargs": {...}}

Mechanisms (scheduler process):
  live tap  (token id, aux hidden state) of every committed position into livetap.Ring in /dev/shm: same format, same
            environment variables (VLLM_SPEC_LIVE_TAP, _GB, _RATE), same reader. Rows: prefill chunks, the accepted
            prefix of every verify step (the pending token and the accepted drafts), the input token of every step
            with speculation off; FINISH once the scheduler has seen the request end.
  reload    the draft's weights are copied in place from <weights_path>/model.safetensors. The embedding belongs to
            the target and is never written; in-place copies keep the captured CUDA graphs valid.
  switch    with speculation off, a decode step runs the target alone on its own CUDA graphs (captured at start-up),
            with the bookkeeping of a verify step that accepted no draft; as in vLLM the drafter does not run, and
            the first speculative step afterwards refreshes its inputs with one draft-extend pass over every
            request's last committed position (its KV entries for the positions decoded meanwhile stay stale).
            Prefill is unchanged.
RPCs travel through a file mailbox that the scheduler polls once per loop iteration and runs between batches; the
scheduler publishes its counters in a shared-memory block that /metrics reads.

Loaded in every SGLang process by specsched_sgl_launch.py (top-level `apply()`).
"""
import json
import mmap
import os
import struct
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LIVETAP = os.path.join(os.path.dirname(HERE), "livetap.py")   # loaded by path: code/specsched never goes on sys.path

MAGIC = 0x5353474C43544C31
STATS = ["running", "waiting", "gen_tokens", "num_drafts", "num_accepted", "preemptions", "spec_on", "swaps",
         "decode_steps", "plain_steps", "prefill_steps", "tap_steps", "tap_dropped", "heartbeat", "drafter_refreshes"]
OFF_STATS = 64


def ctl_path(port):
    return f"/dev/shm/specsched_sgl_{os.environ.get('SLURM_JOB_ID', 'x')}_{port}.ctl"


def log(msg):
    print(f"[specsched-sgl pid={os.getpid()}] {msg}", flush=True)


class Ctl:
    """4 KiB block: u64 magic, u64 cmd_seq (HTTP side), u64 ack_seq (scheduler side), f64 stats[] at OFF_STATS."""

    def __init__(self, path, create=False):
        self.path = path
        if create:
            fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
            os.ftruncate(fd, 4096)
        else:
            fd = os.open(path, os.O_RDWR)
        self.mm = mmap.mmap(fd, 4096)
        os.close(fd)
        if create:
            self.mm[:] = bytes(4096)
            struct.pack_into("<Q", self.mm, 0, MAGIC)

    def u64(self, off, v=None):
        if v is None:
            return struct.unpack_from("<Q", self.mm, off)[0]
        struct.pack_into("<Q", self.mm, off, v)

    def set_stats(self, vals):
        struct.pack_into(f"<{len(vals)}d", self.mm, OFF_STATS, *vals)

    def stats(self):
        return dict(zip(STATS, struct.unpack_from(f"<{len(STATS)}d", self.mm, OFF_STATS)))


# ----------------------------------------------------------------------------------------------- scheduler process

class State:
    def __init__(self):
        self.ctl = None
        self.tap = None
        self.sched = None
        self.c = dict.fromkeys(STATS, 0.0)
        self.last_cmd = 0
        self.t_pub = 0.0


S = State()


def _make_tap(max_tokens, max_reqs):
    import queue

    import importlib.util

    import numpy as np
    import torch

    spec = importlib.util.spec_from_file_location("specsched_livetap", LIVETAP)
    livetap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(livetap)
    LiveTap = livetap.LiveTap

    class SglTap(LiveTap):
        """LiveTap fed with per-request row counts (rows already exclude rejected drafts)."""

        def _is_writer(self):
            return True

        def rows(self, rids, qlens, starts, plens, token_ids, hidden):
            if self.disabled:
                return
            try:
                T = int(token_ids.shape[0])
                if T == 0:
                    return
                if T > self.max_tokens:
                    self.dropped += 1
                    self.q.put(("gap", list(rids)))
                    return
                self._ensure(int(hidden.shape[-1]))
                try:
                    i = self.free.get_nowait()
                except queue.Empty:
                    self.dropped += 1
                    self.q.put(("gap", list(rids)))
                    return
                b = self.bufs[i]
                b["tok"][:T].copy_(token_ids.to(torch.int32), non_blocking=True)
                b["aux"][:T].copy_(hidden.to(torch.bfloat16).contiguous().view(torch.int16), non_blocking=True)
                ev = torch.cuda.Event()
                ev.record()
                self.steps += 1
                self.q.put(("step", i, ev, T, list(rids), np.asarray(qlens, dtype=np.int64),
                            np.asarray(starts, dtype=np.int64), [int(p) for p in plens], None, False))
            except Exception as e:  # never let the tap take the server down
                self.disabled = True
                self.log(f"[live-tap] disabled after error: {e!r}")

    path = os.environ.get(LiveTap.ENV)
    if not path:
        return None
    cap = float(os.environ.get(LiveTap.ENV + "_GB", "4")) * (1 << 30)
    rate = float(os.environ.get(LiveTap.ENV + "_RATE", "1"))
    tap = SglTap(path, cap, rate, max_tokens, max_reqs, log=log)
    log(f"[live-tap] configured: path={path} capacity={cap / 2**30:.1f}GB rate={rate} max_tokens={max_tokens}")
    return tap


def _publish(sched, force=False):
    now = time.time()
    if not force and now - S.t_pub < 0.05:
        return
    S.t_pub = now
    run = sum(1 for r in sched.running_batch.reqs if not r.finished())
    lb = getattr(sched, "last_batch", None)
    if lb is not None and lb is not sched.running_batch and lb.forward_mode.is_extend():
        run += sum(1 for r in lb.reqs if not r.finished())
    c = S.c
    c["running"], c["waiting"], c["heartbeat"] = float(run), float(len(sched.waiting_queue)), now
    w = getattr(sched, "draft_worker", None)
    c["spec_on"] = float(bool(w is not None and getattr(w, "_ss_spec_on", True)))
    if S.tap is not None:
        c["tap_steps"], c["tap_dropped"] = float(S.tap.steps), float(S.tap.dropped)
    S.ctl.set_stats([c[k] for k in STATS])


def _run_cmd(sched, cmd):
    m, kw = cmd.get("method"), cmd.get("kwargs") or {}
    w = getattr(sched, "draft_worker", None)
    if m == "set_speculation":
        if w is None:
            raise ValueError("No speculative draft model is loaded")
        on = str(kw.get("enabled", "1")).strip().lower() in ("1", "true", "yes", "on")
        w._ss_spec_on = on
        return dict(enabled=on)
    if m == "reload_draft_weights":
        if w is None:
            raise ValueError("No speculative draft model is loaded")
        return _reload_draft(w, kw["weights_path"])
    if m == "stats":
        return dict(S.c)
    raise ValueError(f"unknown method {m!r}")


def _poll_mailbox(sched):
    seq = S.ctl.u64(8)
    if seq <= S.last_cmd:
        return
    S.last_cmd = seq
    p = S.ctl.path
    t0 = time.time()
    try:
        cmd = json.load(open(p + ".cmd"))
        assert cmd.get("seq") == seq, f"mailbox out of sync: {cmd.get('seq')} != {seq}"
        res = dict(seq=seq, ok=True, result=_run_cmd(sched, cmd))
    except Exception as e:
        res = dict(seq=seq, ok=False, error=f"{type(e).__name__}: {e}")
    res["seconds"] = time.time() - t0
    log(f"rpc {res}")
    with open(p + ".ack.tmp", "w") as f:
        json.dump(res, f)
    os.replace(p + ".ack.tmp", p + ".ack")
    S.ctl.u64(16, seq)
    _publish(sched, force=True)


def _reload_draft(w, weights_path):
    import torch
    from safetensors.torch import load_file

    t0 = time.time()
    st = os.path.join(weights_path, "model.safetensors")
    if os.path.exists(st):          # the trainer's exports
        sd = load_file(st)
    else:                           # a factory head shipped as pytorch_model.bin (Qwen3-8B's)
        sd = torch.load(os.path.join(weights_path, "pytorch_model.bin"), map_location="cpu", weights_only=True)
    d2t = sd.pop("d2t", None)
    sd.pop("t2d", None)
    skipped = [k for k in sd if "embed_tokens" in k]     # the target's embedding: never written
    for k in skipped:
        sd.pop(k)
    model = w.draft_model_runner.model
    torch.cuda.synchronize()
    with torch.no_grad():
        model.load_weights(iter(sd.items()))
        if d2t is not None and w.hot_token_id is not None:
            hot = (d2t + torch.arange(d2t.shape[0])).to(w.hot_token_id.device, w.hot_token_id.dtype)
            if not torch.equal(hot, w.hot_token_id):
                w.hot_token_id.copy_(hot)
                log("reload: draft vocabulary map changed")
    torch.cuda.synchronize()
    S.c["swaps"] += 1
    return dict(tensors=len(sd), skipped=skipped, seconds=round(time.time() - t0, 3), path=weights_path)


def _tap_finish(batch):
    fin = []
    for r in batch.reqs:
        if r.finished() and not getattr(r, "_ss_fin", False):
            r._ss_fin = True
            fin.append(r.rid)
    if fin and S.tap is not None:
        S.tap.finish(fin)


def _count_tokens(batch):
    n = 0
    for r in batch.reqs:
        k = len(r.output_ids)
        n += max(0, k - getattr(r, "_ss_n", 0))
        r._ss_n = k
    S.c["gen_tokens"] += n


# --------------------------------------------------------------------------------------------- speculation off

def _build_plain_graphs(w):
    """Second set of target CUDA graphs: plain decode (1 token per request) with the aux hidden states captured,
    on its own attention-backend instance (the verify graphs keep theirs)."""
    from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner
    from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    class PlainDecodeGraphRunner(CudaGraphRunner):
        def capture(self):
            self.capture_hidden_mode = CaptureHiddenMode.FULL
            return super().capture()

    mr = w.target_worker.model_runner
    w._ss_plain_backend = None
    w._ss_plain_graphs = None
    if mr.server_args.disable_cuda_graph:
        w._ss_plain_backend = mr._get_attention_backend()
        return
    saved = mr.attn_backend, mr.spec_algorithm
    t0 = time.time()
    try:
        mr.attn_backend = w._ss_plain_backend = mr._get_attention_backend()
        mr.spec_algorithm = SpeculativeAlgorithm.NONE
        w._ss_plain_graphs = PlainDecodeGraphRunner(mr)
    finally:
        mr.attn_backend, mr.spec_algorithm = saved
    log(f"plain-decode CUDA graphs for the speculation switch: bs {w._ss_plain_graphs.capture_bs[0]}.."
        f"{w._ss_plain_graphs.capture_bs[-1]} in {time.time() - t0:.1f} s")


def _keep_aux(w, req_pool_indices, rows):
    """Aux hidden state of each request's last committed position (its pending token's predecessor), indexed by
    request-pool slot: what the drafter needs to resume after speculation was off."""
    import torch
    if w._ss_aux is None:
        n = w.req_to_token_pool.req_to_token.shape[0]
        w._ss_aux = torch.zeros(n, rows.shape[-1], dtype=rows.dtype, device=rows.device)
    w._ss_aux[req_pool_indices] = rows


def _plain_step(w, batch):
    """One decode step with speculation off. Mirrors ScheduleBatch.prepare_for_decode of a server without a drafter,
    then the bookkeeping of EagleVerifyInput.verify for zero accepted drafts. As in vLLM, the drafter does not run:
    its inputs for the pending tokens are refreshed once when speculation resumes (_refresh_drafter), and its KV
    entries for the positions decoded meanwhile stay stale."""
    import torch
    from sglang.srt.managers.utils import GenerationBatchResult
    from sglang.srt.mem_cache.common import alloc_for_decode
    from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    spec_info = batch.spec_info
    bs = len(batch.reqs)
    for req in batch.reqs:
        req.decode_batch_idx += 1
    if batch.sampling_info.penalizer_orchestrator.is_required:
        batch.sampling_info.penalizer_orchestrator.cumulate_output_tokens(spec_info.verified_id.to(torch.int64))
    seq_pre = batch.seq_lens_cpu.clone()
    batch.input_ids = spec_info.verified_id.to(torch.int64)
    batch.out_cache_loc = alloc_for_decode(batch, token_per_req=1)
    batch.seq_lens.add_(1)
    batch.seq_lens_cpu.add_(1)
    batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())
    batch.forward_mode = ForwardMode.DECODE
    batch.return_hidden_states = False
    batch.spec_info = None
    mwb = batch.get_model_worker_batch()
    mwb.capture_hidden_mode = CaptureHiddenMode.FULL
    mwb.spec_algorithm = SpeculativeAlgorithm.NONE
    mr = w.target_worker.model_runner
    saved = mr.graph_runner, mr.attn_backend
    mr.graph_runner, mr.attn_backend = w._ss_plain_graphs, w._ss_plain_backend
    try:
        res = w.target_worker.forward_batch_generation(mwb)
    finally:
        mr.graph_runner, mr.attn_backend = saved
    logits_output, next_ids = res.logits_output, res.next_token_ids
    hidden = logits_output.hidden_states
    _keep_aux(w, batch.req_pool_indices, hidden)
    if S.tap is not None:
        S.tap.rows([r.rid for r in batch.reqs], [1] * bs, seq_pre.tolist(),
                   [len(r.origin_input_ids) for r in batch.reqs], batch.input_ids, hidden)
    # bookkeeping of a verify step that accepted no draft (EagleVerifyInput.verify)
    ids = next_ids.tolist()
    unfinished = []
    for i, req in enumerate(batch.reqs):
        req.output_ids.append(ids[i])
        req.check_finished()
        req.kv_committed_len += 1
        req.kv_allocated_len = req.kv_committed_len
        if not req.finished():
            unfinished.append(i)
            if req.grammar is not None:
                req.grammar.accept_token(ids[i])
    # the draft input keeps its per-request rows (filtered to the unfinished requests, as verify leaves it) with the
    # new pending tokens; topk and draft hidden states are stale until _refresh_drafter
    verified = next_ids.to(torch.int32)
    if len(unfinished) < bs:
        u = torch.tensor(unfinished, dtype=torch.int64, device=batch.seq_lens.device)
        verified = verified[u]
        for f in ("topk_p", "topk_index", "hidden_states"):
            setattr(spec_info, f, getattr(spec_info, f)[u])
    spec_info.verified_id = verified
    batch.spec_info = spec_info
    batch.forward_mode = ForwardMode.DECODE
    w._ss_stale = True
    S.c["plain_steps"] += 1
    return GenerationBatchResult(logits_output=logits_output, next_token_ids=next_ids, num_accepted_tokens=0,
                                 accept_length_per_req_cpu=[0] * bs, can_run_cuda_graph=res.can_run_cuda_graph)


def _refresh_drafter(w, batch):
    """First speculative step after plain steps: one draft-extend pass over every request's last committed position
    (its aux hidden state and pending token), which is what verify's draft extend would have left behind."""
    import torch
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.speculative.eagle_info import EagleDraftInput

    bs = len(batch.reqs)
    dev = batch.seq_lens.device
    batch.spec_info = EagleDraftInput(
        hidden_states=w._ss_aux[batch.req_pool_indices], verified_id=batch.spec_info.verified_id,
        accept_length=torch.zeros(bs, dtype=torch.int32, device=dev), accept_length_cpu=[0] * bs,
        seq_lens_for_draft_extend=batch.seq_lens, seq_lens_for_draft_extend_cpu=batch.seq_lens_cpu,
        req_pool_indices_for_draft_extend=batch.req_pool_indices)
    batch.out_cache_loc = batch.req_to_token_pool.req_to_token[batch.req_pool_indices, batch.seq_lens - 1].to(
        torch.int64)
    batch.forward_mode = ForwardMode.DECODE
    with w.draft_tp_context(w.draft_model_runner.tp_group):
        w.forward_draft_extend_after_decode(batch)
    w._ss_stale = False
    S.c["drafter_refreshes"] += 1


# ------------------------------------------------------------------------------------------------------- patches

def _patch_scheduler():
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.scheduler import Scheduler

    init0, recv0, pbr0 = Scheduler.__init__, Scheduler.recv_requests, Scheduler.process_batch_result
    retract0 = ScheduleBatch.retract_decode

    def __init__(self, server_args, *args, **kw):
        S.ctl = Ctl(ctl_path(server_args.port), create=True)
        init0(self, server_args, *args, **kw)
        S.sched = self
        sa = self.server_args
        S.tap = _make_tap(max(sa.max_prefill_tokens, sa.chunked_prefill_size or 0, 4096) + 1024,
                          max(sa.max_running_requests or 256, 256) + 16)
        _publish(self, force=True)
        log(f"scheduler patched: spec={getattr(self, 'draft_worker', None) is not None} tap={S.tap is not None}")

    def recv_requests(self, *a, **kw):
        _poll_mailbox(self)
        _publish(self)
        return recv0(self, *a, **kw)

    def process_batch_result(self, batch, result, *a, **kw):
        out = pbr0(self, batch, result, *a, **kw)
        if batch is not None and batch.reqs:
            if batch.forward_mode.is_extend():
                S.c["prefill_steps"] += 1
            else:
                S.c["decode_steps"] += 1
            _count_tokens(batch)
            _tap_finish(batch)
        return out

    def retract_decode(self, *a, **kw):
        out = retract0(self, *a, **kw)
        S.c["preemptions"] += len(out[0])
        return out

    Scheduler.__init__ = __init__
    Scheduler.recv_requests = recv_requests
    Scheduler.process_batch_result = process_batch_result
    ScheduleBatch.retract_decode = retract_decode


def _patch_eagle():
    from sglang.srt.speculative.eagle_worker import EAGLEWorker

    init0, gen0 = EAGLEWorker.__init__, EAGLEWorker.forward_batch_generation
    ext0, ver0 = EAGLEWorker.forward_target_extend, EAGLEWorker.verify

    def __init__(self, *a, **kw):
        init0(self, *a, **kw)
        self._ss_spec_on, self._ss_stale, self._ss_aux = True, False, None
        _build_plain_graphs(self)

    def forward_batch_generation(self, batch):
        if not (batch.forward_mode.is_extend() or batch.is_extend_in_batch) and not batch.forward_mode.is_idle():
            if not self._ss_spec_on:
                return _plain_step(self, batch)
            if self._ss_stale:
                _refresh_drafter(self, batch)
            bs = len(batch.reqs)
            res = gen0(self, batch)
            S.c["num_drafts"] += bs
            S.c["num_accepted"] += sum(res.accept_length_per_req_cpu)
            return res
        return gen0(self, batch)

    def forward_target_extend(self, batch):
        import torch
        out = ext0(self, batch)
        if not batch.forward_mode.is_idle() and batch.reqs:
            last = torch.tensor(batch.extend_lens, device=batch.seq_lens.device).cumsum(0) - 1
            _keep_aux(self, batch.req_pool_indices, out[0].hidden_states[last])
        if S.tap is not None and not batch.forward_mode.is_idle():
            S.tap.rows([r.rid for r in batch.reqs], list(batch.extend_lens), list(batch.prefix_lens),
                       [len(r.origin_input_ids) for r in batch.reqs], batch.input_ids, out[0].hidden_states)
        return out

    def verify(self, batch, spec_info):
        seq_pre = batch.seq_lens_cpu.clone() if S.tap is not None else None
        draft_token = spec_info.draft_token
        out = ver0(self, batch, spec_info)
        if S.tap is not None and not batch.forward_mode.is_idle():
            logits_output, res = out[0], out[1]
            S.tap.rows([r.rid for r in batch.reqs], [a + 1 for a in res.accept_length_per_req_cpu], seq_pre.tolist(),
                       [len(r.origin_input_ids) for r in batch.reqs], draft_token[res.accepted_indices],
                       logits_output.hidden_states)
        return out

    EAGLEWorker.__init__ = __init__
    EAGLEWorker.forward_batch_generation = forward_batch_generation
    EAGLEWorker.forward_target_extend = forward_target_extend
    EAGLEWorker.verify = verify


# ---------------------------------------------------------------------------------------------- HTTP (main process)

def _patch_http(port):
    import asyncio

    from fastapi import Request
    from fastapi.responses import JSONResponse, PlainTextResponse
    from sglang.srt.entrypoints import http_server

    app = http_server.app
    st = dict(ctl=None, lock=None)

    def ctl():
        if st["ctl"] is None:
            p = ctl_path(port)
            if not os.path.exists(p):
                return None
            st["ctl"] = Ctl(p)
        return st["ctl"]

    @app.get("/metrics")
    async def metrics():
        c = ctl()
        s = c.stats() if c is not None else dict.fromkeys(STATS, 0.0)
        lab = '{engine="0",model_name="target"}'
        rows = [("vllm:num_requests_running", "gauge", s["running"]),
                ("vllm:num_requests_waiting", "gauge", s["waiting"]),
                ("vllm:generation_tokens_total", "counter", s["gen_tokens"]),
                ("vllm:spec_decode_num_drafts_total", "counter", s["num_drafts"]),
                ("vllm:spec_decode_num_accepted_tokens_total", "counter", s["num_accepted"]),
                ("vllm:num_preemptions_total", "counter", s["preemptions"]),
                ("specsched:spec_enabled", "gauge", s["spec_on"]),
                ("specsched:draft_swaps_total", "counter", s["swaps"]),
                ("specsched:decode_steps_total", "counter", s["decode_steps"]),
                ("specsched:plain_steps_total", "counter", s["plain_steps"]),
                ("specsched:prefill_steps_total", "counter", s["prefill_steps"]),
                ("specsched:tap_steps_total", "counter", s["tap_steps"]),
                ("specsched:tap_dropped_steps_total", "counter", s["tap_dropped"]),
                ("specsched:scheduler_heartbeat_seconds", "gauge", s["heartbeat"]),
                ("specsched:drafter_refreshes_total", "counter", s["drafter_refreshes"])]
        txt = "".join(f"# TYPE {n} {t}\n{n}{lab} {v!r}\n" for n, t, v in rows)
        return PlainTextResponse(txt)

    @app.post("/collective_rpc")
    async def collective_rpc(request: Request):
        body = await request.json()
        c = ctl()
        if c is None:
            return JSONResponse({"error": "scheduler not ready"}, status_code=503)
        if st["lock"] is None:
            st["lock"] = asyncio.Lock()
        async with st["lock"]:
            seq = c.u64(8) + 1
            p = c.path
            with open(p + ".cmd.tmp", "w") as f:
                json.dump(dict(seq=seq, method=body.get("method"), kwargs=body.get("kwargs") or {}), f)
            os.replace(p + ".cmd.tmp", p + ".cmd")
            c.u64(8, seq)
            t0 = time.time()
            while c.u64(16) < seq:
                if time.time() - t0 > 600:
                    return JSONResponse({"error": "scheduler did not answer"}, status_code=504)
                await asyncio.sleep(0.002)
            res = json.load(open(p + ".ack"))
        return JSONResponse({"results": [res]}, status_code=200 if res.get("ok") else 500)


def apply():
    """Scheduler-side patches; runs at import of the launcher in every SGLang process."""
    if os.environ.get("SPECSCHED_SGL_PATCH", "1") != "1":
        return
    _patch_scheduler()
    _patch_eagle()


def install_http(port):
    """HTTP endpoints; the launcher calls this in the main process before the server starts."""
    if os.environ.get("SPECSCHED_SGL_PATCH", "1") != "1":
        return
    for ext in ("", ".cmd", ".ack"):
        if os.path.exists(ctl_path(port) + ext):
            os.remove(ctl_path(port) + ext)
    _patch_http(port)
