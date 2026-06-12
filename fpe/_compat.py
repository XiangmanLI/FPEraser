"""Cross-version compatibility shims.

Currently contains one shim:

    * :func:`patch_peft_weight_converter` — make ``peft 0.19``'s
      ``WeightConverter.__init__`` accept (and silently drop) keyword
      arguments that newer ``transformers`` releases pass to it. Without
      this patch, ``get_peft_model(...)`` on certain base models raises
      ``TypeError: __init__() got an unexpected keyword argument ...``.

Call :func:`patch_peft_weight_converter()` once near the top of any
training entrypoint that wraps a base model in PEFT. The patch is
idempotent.
"""
import inspect


def patch_peft_weight_converter() -> None:
    """Make ``peft.utils.transformers_weight_conversion.WeightConverter``
    tolerate extra kwargs from newer transformers releases.

    Idempotent: a second call is a no-op.
    """
    import peft.utils.transformers_weight_conversion as _twc

    if getattr(_twc.WeightConverter.__init__, "_fpe_patched", False):
        return

    _orig = _twc.WeightConverter.__init__
    _allowed = set(inspect.signature(_orig).parameters.keys()) - {"self"}

    def _patched(self, *args, **kwargs):
        filtered = {k: v for k, v in kwargs.items() if k in _allowed}
        return _orig(self, *args, **filtered)

    _patched._fpe_patched = True  # type: ignore[attr-defined]
    _twc.WeightConverter.__init__ = _patched
