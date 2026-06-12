"""Embedding-level adapter for the IF fingerprint scheme.

This module is a vendored, near-verbatim port of the ``adapter.py`` shipped
with the original IF (Instructional Fingerprinting) reference
implementation by Cai et al.:

    https://github.com/cnut1648/Model-Fingerprint

It is included here so that the FPEraser repository is self-contained;
the only changes versus the upstream are (a) wrapping the new-style
``model.get_encoder()`` no-op stub introduced by ``transformers ≥ 4.4x``,
and (b) docstrings + light formatting. **All credit for the core IF
adapter design belongs to Cai et al.**

The adapter replaces the model's input embedding with an
:class:`InstructionFingerprint` module that, for a fixed set of "trigger"
token ids, adds a low-rank bottleneck ``B(A(trainable_emb(id)))`` onto
the original embedding. The bottleneck is initialized at zero so the
wrapped model is bitwise-equivalent to the original until training
moves any adapter parameter.
"""
from __future__ import annotations

from typing import List, Set

import torch
from transformers import AutoModelForCausalLM


class InstructionFingerprint(torch.nn.Module):
    """Embedding wrapper that injects a trainable bottleneck on trigger tokens."""

    def __init__(
        self,
        emb: torch.nn.Module,
        all_trainable_input_ids: List[int],
        inner_dim: int = 128,
    ):
        super().__init__()
        self.orig_emb = emb
        self.all_trainable_input_ids = all_trainable_input_ids
        self.trainable_emb = torch.nn.Embedding(
            len(all_trainable_input_ids), self.orig_emb.weight.size(1)
        )
        with torch.no_grad():
            self.trainable_emb.weight.copy_(emb.weight[all_trainable_input_ids])
        self.A = torch.nn.Linear(self.orig_emb.weight.size(1), inner_dim)
        self.B = torch.nn.Linear(inner_dim, self.orig_emb.weight.size(1))
        with torch.no_grad():
            self.A.weight.zero_()
            self.A.bias.zero_()
            self.B.weight.zero_()
            self.B.bias.zero_()
        self.cast_dtype()

    @property
    def weight(self):
        return self.orig_emb.weight

    @torch.no_grad()
    def cast_dtype(self) -> None:
        """Re-cast all adapter parameters to match ``orig_emb``'s dtype."""
        dtype = self.orig_emb.weight.dtype
        for param in self.parameters():
            param.data = param.data.to(dtype=dtype)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """For each input id:

        * if it is in ``all_trainable_input_ids``: emit
          ``B(A(trainable_emb(id))) + orig_emb(id)``;
        * otherwise: emit ``orig_emb(id)``.
        """
        N, L = input.size()
        trainable_ids = torch.tensor(self.all_trainable_input_ids).to(
            device=input.device, dtype=input.dtype
        )
        mask = (input.unsqueeze(-1) == trainable_ids).any(-1)
        indices = (input[mask].unsqueeze(-1) == trainable_ids).max(-1).indices

        embeddings_from_trainable = (
            self.B(self.A(self.trainable_emb(indices))) + self.orig_emb(input[mask])
        )
        embeddings_from_orig = self.orig_emb(input[~mask])

        output = torch.empty(
            N, L, self.orig_emb.weight.size(1),
            device=input.device, dtype=self.orig_emb.weight.dtype,
        )
        output[mask] = embeddings_from_trainable
        output[~mask] = embeddings_from_orig
        return output

    @torch.no_grad()
    def merge(self) -> None:
        """Fold ``trainable_emb`` back into ``orig_emb`` (in place)."""
        self.orig_emb.weight[self.all_trainable_input_ids] = self.trainable_emb.weight


def _is_real_seq2seq(model) -> bool:
    """Distinguish real seq2seq models from CausalLM ``get_encoder`` no-op stubs
    introduced in ``transformers ≥ 4.4x``."""
    return (
        hasattr(model, "get_encoder")
        and model.get_encoder() is not model
        and model.get_encoder() is not getattr(model, "model", None)
    )


def inject_adapter_to(
    model: AutoModelForCausalLM,
    all_trainable_input_ids: Set[int],
    trained_adapter: InstructionFingerprint | None = None,
    inner_dim: int = 16,
):
    """Wrap ``model.get_input_embeddings()`` with :class:`InstructionFingerprint`.

    Freezes every base-model parameter; only the new adapter weights are
    left trainable.

    Args:
        model: a HuggingFace causal-LM (or real seq2seq) instance.
        all_trainable_input_ids: token ids that should receive the adapter
            bottleneck on top of their original embedding.
        trained_adapter: optionally re-use an existing adapter (for
            evaluation), rather than constructing a fresh zero-init one.
        inner_dim: bottleneck rank (paper default: 16).

    Returns:
        The same ``model`` instance, mutated in place.
    """

    def find_emb_and_replace(model_, trained_adapter_):
        emb_attr_str = None
        for name, module in model_.named_modules():
            if (
                isinstance(module, torch.nn.Embedding)
                and module is model_.get_input_embeddings()
            ):
                emb_attr_str = name
        assert emb_attr_str is not None, "Cannot find embedding layer"
        attr_path = emb_attr_str.split(".")
        parent = model_
        for attr in attr_path[:-1]:
            parent = getattr(parent, attr)
        if trained_adapter_ is not None:
            replaced = trained_adapter_
        else:
            replaced = InstructionFingerprint(
                model_.get_input_embeddings(),
                list(all_trainable_input_ids),
                inner_dim=inner_dim,
            )
        setattr(parent, attr_path[-1], replaced)
        assert isinstance(model_.get_input_embeddings(), InstructionFingerprint)
        for param in model_.get_input_embeddings().parameters():
            param.requires_grad = True
        model_.get_input_embeddings().orig_emb.weight.requires_grad = False
        return replaced

    for param in model.parameters():
        param.requires_grad = False

    if _is_real_seq2seq(model):
        assert (
            model.get_encoder().get_input_embeddings()
            is model.get_decoder().get_input_embeddings()
        ), "Only support shared embedding for now"
        replaced_adapter = find_emb_and_replace(model.get_encoder(), trained_adapter)
        find_emb_and_replace(model.get_decoder(), trained_adapter=replaced_adapter)
    else:
        find_emb_and_replace(model, trained_adapter=trained_adapter)
    return model


def unwrap_adapter(model):
    """Reverse :func:`inject_adapter_to`: merge adapter into the base
    embedding and restore the original ``nn.Embedding`` module.

    Returns ``(model, instruction_emb)`` — the model is mutated in place,
    and the original (now-merged) :class:`InstructionFingerprint` is
    returned alongside for inspection / saving.
    """

    def find_emb_and_restore(model_):
        instruction_emb = model_.get_input_embeddings()
        instruction_emb.merge()
        emb_attr_str = None
        for name, module in model_.named_modules():
            if isinstance(module, InstructionFingerprint) and module is model_.get_input_embeddings():
                emb_attr_str = name
        attr_path = emb_attr_str.split(".")
        parent = model_
        for attr in attr_path[:-1]:
            parent = getattr(parent, attr)
        setattr(parent, attr_path[-1], instruction_emb.orig_emb)
        assert isinstance(model_.get_input_embeddings(), torch.nn.Embedding)
        return instruction_emb

    if _is_real_seq2seq(model):
        instruction_emb = find_emb_and_restore(model.get_encoder())
        assert instruction_emb is find_emb_and_restore(model.get_decoder())
    else:
        instruction_emb = find_emb_and_restore(model)
    return model, instruction_emb
