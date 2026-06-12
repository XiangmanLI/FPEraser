"""Behavioral-fingerprint (BF) KL-divergence ratios.

The BF profile characterizes how the surrogate ``M_θ`` relates to the
three reference models — ``M_base``, ``M_owner``, and
``M_target = M_owner + fp`` — on two probe regions:

  * **natural**: response continuations on Alpaca-style instructions,
    where ``M_owner ≈ M_target`` (the fingerprint is dormant).
  * **trigger**: prompts that exercise the fingerprint trigger and on
    which ``M_target`` diverges sharply from ``M_owner``.

For every (region, reference) cell we compute the mean per-token KL
divergence ``KL(M_θ || M_ref)`` and then report five ratios that the
paper uses as headline diagnostics::

    r_owner_nat   = KL(θ || owner)_nat   / KL(θ || base)_nat   # want < 1
    r_owner_trig  = KL(θ || owner)_trig  / KL(θ || base)_trig  # want < 1
    r_fp_nat      = KL(θ || fp)_nat      / KL(θ || base)_nat   # want < 1
    r_fp_trig     = KL(θ || fp)_trig     / KL(θ || base)_trig  # want > 1
    r_disentangle = KL(θ || fp)_trig     / KL(θ || owner)_trig # want ≫ 1

The last ratio is the headline number — large values mean the surrogate
matches ``M_owner`` rather than ``M_target`` on the trigger region,
i.e. the fingerprint has been erased while the owner-side SFT
behavior is preserved.
"""
from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from fpe.data.alpaca import load as load_alpaca

DEFAULT_FIRST_K = 8
DEFAULT_N_NATURAL = 100
DEFAULT_MAX_LEN = 1024
DEFAULT_TEMPERATURE = 1.0


@dataclass
class Probe:
    """A logit-collection probe: ``full_ids`` plus a ``prompt_len`` cut.

    The model is forwarded on ``full_ids`` and the logits at positions
    ``[prompt_len - 1 : len - 1]`` are kept (each predicts the token at
    position ``[prompt_len : len]``). ``prompt_len = len(full_ids) - 1``
    is a convenient form for trigger probes where only the very last
    next-token logit is scored.
    """

    full_ids: torch.Tensor  # shape (1, T)
    prompt_len: int


def _log(msg: str) -> None:
    print(msg, flush=True)


def _free(*models) -> None:
    for m in models:
        del m
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _format_alpaca_for_probe(ex: dict) -> tuple[str, str]:
    if ex.get("input"):
        prompt = (
            f"### Instruction:\n{ex['instruction']}\n\n"
            f"### Input:\n{ex['input']}\n\n### Response:\n"
        )
    else:
        prompt = f"### Instruction:\n{ex['instruction']}\n\n### Response:\n"
    return prompt, ex["output"]


def build_natural_probes(
    tokenizer,
    n: int = DEFAULT_N_NATURAL,
    *,
    first_k: int = DEFAULT_FIRST_K,
    max_len: int = DEFAULT_MAX_LEN,
    seed: int = 42,
    alpaca_offset: int = 0,
) -> list[Probe]:
    """Return ``n`` natural-region probes from Alpaca instructions.

    Each probe scores the first ``first_k`` response tokens of an
    Alpaca example. The intended use is to load these once and reuse
    them across every model whose logits we want to compare.
    """
    ds = load_alpaca(alpaca_offset + n, seed)
    probes: list[Probe] = []
    pad_eos = tokenizer.eos_token or ""
    for ex in list(ds)[alpaca_offset:]:
        prompt, response = _format_alpaca_for_probe(ex)
        prompt_ids = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=max_len
        ).input_ids
        full_ids = tokenizer(
            prompt + response + pad_eos,
            return_tensors="pt",
            truncation=True,
            max_length=2 * max_len,
        ).input_ids
        prompt_len = prompt_ids.size(1)
        if full_ids.size(1) <= prompt_len + 1:
            continue
        if full_ids.size(1) > prompt_len + first_k:
            full_ids = full_ids[:, : prompt_len + first_k]
        probes.append(Probe(full_ids=full_ids, prompt_len=prompt_len))
    return probes


def build_trigger_probes_from_texts(
    tokenizer,
    trigger_texts: list[str],
    *,
    max_len: int = DEFAULT_MAX_LEN,
) -> list[Probe]:
    """Wrap each trigger string into a :class:`Probe` that scores the
    last-position next-token logit (single-token prediction)."""
    probes: list[Probe] = []
    for text in trigger_texts:
        ids = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_len
        ).input_ids
        probes.append(Probe(full_ids=ids, prompt_len=ids.size(1) - 1))
    return probes


@torch.no_grad()
def _collect_logits(
    model,
    probes: list[Probe],
    store_dtype: torch.dtype = torch.float32,
) -> list[torch.Tensor]:
    """Forward each probe through ``model`` and return the slice of
    logits over the scored positions, on CPU."""
    out: list[torch.Tensor] = []
    device = next(model.parameters()).device
    for probe in probes:
        ids = probe.full_ids.to(device)
        logits = model(ids).logits[0]
        slc = logits[probe.prompt_len - 1 : ids.size(1) - 1].to(store_dtype).cpu()
        out.append(slc)
    return out


def _avg_token_kl(
    p_list: list[torch.Tensor],
    q_list: list[torch.Tensor],
    temperature: float = DEFAULT_TEMPERATURE,
) -> float:
    """Mean over probes of (mean over tokens of) KL(softmax(p/T) ||
    softmax(q/T))."""
    if len(p_list) != len(q_list):
        raise ValueError("p and q probe-logit lists must have equal length")
    per_probe = []
    for lp, lq in zip(p_list, q_list):
        v = min(lp.size(-1), lq.size(-1))
        lp = lp[..., :v] / temperature
        lq = lq[..., :v] / temperature
        log_p = F.log_softmax(lp, dim=-1)
        log_q = F.log_softmax(lq, dim=-1)
        p = log_p.exp()
        kl = (p * (log_p - log_q)).sum(-1)
        per_probe.append(kl.mean().item())
    return sum(per_probe) / max(len(per_probe), 1)


def _load_model(path: str):
    return AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )


def compute_bf_profile(
    m_base_path: str,
    m_owner_path: str,
    m_target_path: str,
    m_theta_path: str,
    *,
    tokenizer_path: str | None = None,
    trigger_texts: list[str],
    n_natural_probes: int = DEFAULT_N_NATURAL,
    first_k: int = DEFAULT_FIRST_K,
    temperature: float = DEFAULT_TEMPERATURE,
    seed: int = 42,
    alpaca_offset: int = 0,
) -> dict:
    """Compute the BF KL-ratio profile of ``M_θ`` against three references.

    Args:
        m_base_path / m_owner_path / m_target_path / m_theta_path:
            HF ids or local paths of the four models.
        tokenizer_path:    tokenizer source; defaults to ``m_base_path``.
        trigger_texts:     list of strings that exercise the scheme's
            trigger. The caller is responsible for extracting these
            from the scheme-specific pair dicts (e.g. for SF this is
            ``[p["key_text"] for p in pairs]``).
        n_natural_probes:  how many Alpaca prompts to score on the
            natural region.
        first_k:           how many response tokens of each natural
            probe to keep.
        temperature:       KL temperature scaling.
        seed:              Alpaca shuffle seed.
        alpaca_offset:     how many examples to skip before drawing
            natural probes (useful to keep them disjoint from any
            training cache).

    Returns:
        Dict with raw ``kl_ours_vs_*_*`` numbers, the five ratios
        (``r_owner_nat``, ``r_owner_trig``, ``r_fp_nat``,
        ``r_fp_trig``, ``r_disentangle``), and a ``meta`` block.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path or m_base_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    _log("  building natural probes ...")
    nat_probes = build_natural_probes(
        tokenizer, n=n_natural_probes, first_k=first_k,
        seed=seed, alpaca_offset=alpaca_offset,
    )
    _log(f"    {len(nat_probes)} natural probes")
    _log("  building trigger probes ...")
    trig_probes = build_trigger_probes_from_texts(tokenizer, trigger_texts)
    _log(f"    {len(trig_probes)} trigger probes")

    logits_cache: dict[str, dict[str, list[torch.Tensor]]] = {}
    for label, path in (
        ("base", m_base_path),
        ("owner", m_owner_path),
        ("fp", m_target_path),
        ("ours", m_theta_path),
    ):
        _log(f"  [{label}] loading + collecting logits ...")
        model = _load_model(path)
        logits_cache[label] = {
            "nat": _collect_logits(model, nat_probes),
            "trig": _collect_logits(model, trig_probes),
        }
        _free(model)

    res: dict = {}
    for region in ("nat", "trig"):
        for ref in ("base", "owner", "fp"):
            res[f"kl_ours_vs_{ref}_{region}"] = _avg_token_kl(
                logits_cache["ours"][region],
                logits_cache[ref][region],
                temperature=temperature,
            )

    EPS = 1e-8
    res["r_owner_nat"] = res["kl_ours_vs_owner_nat"] / max(res["kl_ours_vs_base_nat"], EPS)
    res["r_owner_trig"] = res["kl_ours_vs_owner_trig"] / max(res["kl_ours_vs_base_trig"], EPS)
    res["r_fp_nat"] = res["kl_ours_vs_fp_nat"] / max(res["kl_ours_vs_base_nat"], EPS)
    res["r_fp_trig"] = res["kl_ours_vs_fp_trig"] / max(res["kl_ours_vs_base_trig"], EPS)
    res["r_disentangle"] = res["kl_ours_vs_fp_trig"] / max(res["kl_ours_vs_owner_trig"], EPS)

    res["meta"] = {
        "m_base": str(m_base_path),
        "m_owner": str(m_owner_path),
        "m_target": str(m_target_path),
        "m_theta": str(m_theta_path),
        "n_natural_probes": len(nat_probes),
        "n_trigger_probes": len(trig_probes),
        "first_k": first_k,
        "temperature": temperature,
    }
    return res
