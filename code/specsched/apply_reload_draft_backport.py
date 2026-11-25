"""Backport `reload_draft_weights` (vLLM PR branch feat/reload-draft-weights) into an installed vLLM 0.27.x.

Inserts the same code at the same anchors as the upstream patch; idempotent; keeps .orig backups.
Usage: python apply_reload_draft_backport.py <site-packages/vllm>
"""
import os
import shutil
import sys

HELPER = '''
class _DiscardWeights(nn.Module):
    """Swallows checkpoint tensors aimed at a draft submodule owned by the target."""

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        for _ in weights:
            pass
        return set()


@contextmanager
def _detach_target_owned_draft_modules(
    draft_model: nn.Module, target_model: nn.Module
) -> Iterator[None]:
    """Unbind draft submodules that are live target objects while draft weights
    load, so a draft checkpoint can never overwrite target storage (EAGLE drafters
    share the target embedding; MTP drafters alias the target lm_head)."""
    if not isinstance(draft_model, nn.Module) or not isinstance(target_model, nn.Module):
        yield
        return
    target_ids = {id(module) for module in target_model.modules()}
    detached: list[tuple[nn.Module, str, nn.Module]] = []
    for name, module in draft_model.named_modules(remove_duplicate=False):
        if not name or id(module) not in target_ids:
            continue
        parent_name, _, attr = name.rpartition(".")
        parent = draft_model.get_submodule(parent_name) if parent_name else draft_model
        if id(parent) not in target_ids:
            detached.append((parent, attr, module))
    for parent, attr, _ in detached:
        setattr(parent, attr, _DiscardWeights())
    try:
        yield
    finally:
        for parent, attr, module in detached:
            setattr(parent, attr, module)


'''

METHOD = '''
    def reload_draft_weights(
        self,
        weights_iterator: Iterable[tuple[str, torch.Tensor]] | None = None,
        weights_path: str | None = None,
    ) -> None:
        """Reload the speculative draft model's weights in place (backport)."""
        get_draft_model = getattr(self, "get_draft_model", None)
        draft_model = get_draft_model() if callable(get_draft_model) else None
        if draft_model is None or self.speculative_config is None:
            raise ValueError("No speculative draft model is loaded")
        draft_model_config = self.speculative_config.draft_model_config
        assert draft_model_config is not None

        if weights_path is not None:
            draft_model_config.model = weights_path
            draft_model_config.revision = None
        if weights_iterator is None:
            load_config = self.speculative_config.draft_load_config or self.load_config
            model_loader = get_model_loader(load_config)
            if not hasattr(model_loader, "get_all_weights"):
                raise NotImplementedError(
                    f"Draft model reloading with `{load_config.load_format}` format"
                )
            weights_iterator = cast(
                Iterable[tuple[str, torch.Tensor]],
                model_loader.get_all_weights(draft_model_config, draft_model),
            )

        logger.info("Reloading draft model weights in place...")
        counter_before_reloading = time.perf_counter()
        with _detach_target_owned_draft_modules(draft_model, self.get_model()):
            initialize_layerwise_reload(draft_model)
            draft_model.load_weights(weights_iterator)
            finalize_layerwise_reload(draft_model, draft_model_config)
        rebuild_fused_state = getattr(
            getattr(draft_model, "model", None) or draft_model,
            "_build_fused_kv_buffers",
            None,
        )
        if callable(rebuild_fused_state):
            rebuild_fused_state()
        logger.info(
            "Reloading draft model weights took %.2f seconds",
            time.perf_counter() - counter_before_reloading,
        )

'''

WORKER = '''
    def reload_draft_weights(self, *args, **kwargs) -> None:
        with set_current_vllm_config(self.vllm_config):
            self.model_runner.reload_draft_weights(*args, **kwargs)
'''

V2 = '''
    def reload_draft_weights(self, *args, **kwargs) -> None:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner as GPUModelRunnerV1

        GPUModelRunnerV1.reload_draft_weights(self, *args, **kwargs)  # type: ignore[arg-type]
'''


def insert_before(text, anchor, block, marker):
    if marker in text:
        return text, False
    i = text.index(anchor)
    return text[:i] + block + text[i:], True


def insert_after_def(text, def_anchor, block, marker):
    """Insert `block` right after the method starting at def_anchor (before the next '    def ' or '\n\n' top-level)."""
    if marker in text:
        return text, False
    i = text.index(def_anchor)
    j = text.index("\n    def ", i + len(def_anchor))
    return text[: j + 1] + block + text[j + 1:], True


def main(root):
    changed = []
    p = os.path.join(root, "v1/worker/gpu_model_runner.py")
    s = open(p).read()
    assert "from contextlib import" in s and "contextmanager" in s and "Iterator" in s, "imports differ"
    s, c1 = insert_before(s, "class GPUModelRunner(\n", HELPER, "def _detach_target_owned_draft_modules")
    s, c2 = insert_after_def(s, "    def reload_weights(\n", METHOD, "def reload_draft_weights(")
    if c1 or c2:
        shutil.copy(p, p + ".bak") if not os.path.exists(p + ".orig") else None
        open(p, "w").write(s); changed.append(p)
    p = os.path.join(root, "v1/worker/gpu_worker.py")
    s = open(p).read()
    s, c = insert_after_def(s, "    def reload_weights(self, *args, **kwargs) -> None:\n", WORKER, "def reload_draft_weights(")
    if c:
        open(p, "w").write(s); changed.append(p)
    p = os.path.join(root, "v1/worker/gpu/model_runner.py")
    if os.path.exists(p):
        s = open(p).read()
        if "    def reload_weights(self, *args, **kwargs) -> None:\n" in s:
            s, c = insert_after_def(s, "    def reload_weights(self, *args, **kwargs) -> None:\n", V2, "def reload_draft_weights(")
            if c:
                open(p, "w").write(s); changed.append(p)
    print("patched:", changed or "nothing (already applied)")


if __name__ == "__main__":
    main(sys.argv[1])
