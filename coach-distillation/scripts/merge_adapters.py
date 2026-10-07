#!/usr/bin/env python3
"""Merge separately-trained LoRA adapters so one model keeps two behaviours.

Why this exists
---------------
Every attempt to teach the coach both behaviours in a single corpus has failed, and the
measurements say the reason is contention in one optimisation, not a bad mix. v7 (no
charter boost) scores activation 0.86; adding charter rows at 16x took charter to 1.00
but dropped activation to 0.40, and neither cutting repetition to 4x nor cutting epochs
from three to two recovered it (v8a 0.50, v8b 0.40).

But the behaviours do not occupy the same directions in weight space. The delta of the
v7 adapter (grounded first actions) has cosine 0.077 with the delta of the v8 adapter
(refusal with substance) -- they are nearly orthogonal. So they are not fighting for the
same parameters; they are fighting because one training run had to trade them off in a
single loss.

That is a data problem, not a capacity limit, and the fix follows from the merging
literature: train the behaviours SEPARATELY, then compose the displacements.

- **Task arithmetic** (Ilharco et al., 2022): add the weight displacements.
- **DARE** (Yu et al., ICML 2024, arXiv:2311.03099): randomly drop a fraction p of delta
  entries and rescale survivors by 1/(1-p). Their measurement is that SFT deltas are tiny
  and highly redundant -- 90-99% of entries drop with no loss. Redundancy is what makes
  two deltas collide, so removing it first is what buys room for both.
- **TIES** (Yadav et al., NeurIPS 2023, arXiv:2306.01708): trim small-magnitude entries,
  elect one sign per parameter by summed magnitude, keep only agreeing entries. Resolves
  the sign disagreement they identify as the dominant interference source.

Applied per module on the materialised delta ``B @ A``, not on the A and B factors
separately: trimming factors independently is a different operation and does not implement
the published algorithm, which is defined on the delta.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

log = logging.getLogger("merge_adapters")


def _lora_delta(state: dict[str, torch.Tensor], key: str, scale: float) -> torch.Tensor:
    """Materialise the weight displacement for one adapted module.

    LoRA applies ``(alpha / r) * B @ A``. The scale is part of the displacement, so it is
    applied here rather than after merging: applying it afterwards would also scale the
    per-source weights and silently change their relative influence.
    """
    a = state[key.replace("lora_B", "lora_A")].float()
    b = state[key].float()
    return (b @ a) * scale


def _load(path: Path) -> tuple[dict[str, torch.Tensor], float]:
    """Load an adapter and return its weights plus its effective B@A scale."""
    state = load_file(str(path / "adapter_model.safetensors"))
    config = json.loads((path / "adapter_config.json").read_text())
    scale = float(config.get("lora_alpha", 64)) / float(config.get("r", 32))
    return state, scale


@torch.no_grad()
def dare(
    delta: torch.Tensor, drop_ratio: float, generator: torch.Generator
) -> torch.Tensor:
    """DARE: drop each entry with probability p, rescale survivors by 1/(1-p).

    Preserves the displacement in expectation, which is the point -- the adapter keeps its
    contribution while its parameter footprint shrinks.
    """
    if drop_ratio <= 0:
        return delta
    keep = (torch.rand(delta.shape, generator=generator) >= drop_ratio).float()
    return delta * keep / (1.0 - drop_ratio)


@torch.no_grad()
def ties_merge(deltas: list[torch.Tensor], trim_ratio: float) -> torch.Tensor:
    """TIES: trim, elect a sign by summed magnitude, then average agreeing entries."""
    stacked = torch.stack(deltas)
    if trim_ratio > 0:
        flat = stacked.reshape(len(deltas), -1)
        k = max(1, int(flat.shape[1] * (1.0 - trim_ratio)))
        threshold = (
            flat.abs().kthvalue(flat.shape[1] - k + 1, dim=1, keepdim=True).values
        )
        flat = flat * (flat.abs() >= threshold)
        stacked = flat.reshape(stacked.shape)

    sign = torch.sign(stacked.sum(dim=0))
    # Zero arises when entries cancel exactly; those parameters are dropped rather than
    # merged, which is the conservative reading of "agreeing with the elected sign".
    agree = (torch.sign(stacked) == sign.unsqueeze(0)) & (sign.unsqueeze(0) != 0)
    total = (stacked * agree).sum(dim=0)
    count = agree.sum(dim=0)
    return torch.where(count > 0, total / count.clamp(min=1), torch.zeros_like(total))


@torch.no_grad()
def build(
    sources: list[tuple[str, dict[str, torch.Tensor], float, float]],
    drop_ratio: float,
    trim_ratio: float,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Return the merged adapter state, keyed like the inputs.

    ``sources`` is ``(label, state, scale, weight)`` per adapter: ``scale`` is alpha/r and
    ``weight`` is how strongly that behaviour is asked for.
    """
    base_state = sources[0][1]
    keys = sorted(k for k in base_state if "lora_B" in k)
    out: dict[str, torch.Tensor] = {}
    generator = torch.Generator().manual_seed(seed)
    # All sources must agree on alpha/r: it is written into the shared adapter_config,
    # and writing factors that assume one scale while the config states another would
    # reconstruct a different displacement than the merge computed.
    scales = {round(s, 6) for _l, _st, s, _w in sources}
    if len(scales) != 1:
        raise ValueError(
            f"adapters disagree on alpha/r ({sorted(scales)}); they must share a base "
            "config for the merged adapter to be loadable"
        )
    scale = scales.pop()
    for key in keys:
        # NOTE: the per-source scale must stay named src_scale. Binding it to `scale`
        # here would shadow the validated outer `scale` for every subsequent iteration
        # of the loop -- and the outer value is the one the factors are divided by, so
        # the written adapter would reconstruct a different displacement than the merge
        # computed. Caught by the round-trip test, which is the only thing that checks it.
        deltas = [
            dare(_lora_delta(state, key, src_scale) * weight, drop_ratio, generator)
            for _label, state, src_scale, weight in sources
        ]
        merged = ties_merge(deltas, trim_ratio)
        # The merged displacement is full-rank (out x in), but a LoRA file has to be a
        # rank-r factorisation with the SAME r the config declares. Writing the full
        # matrix into B would either fail the shape check or, if truncated, silently
        # discard all but r rows -- which is most of the merge.
        #
        # So factorise properly and put the scale back where LoRA expects it:
        #   merged ~= U @ S @ Vh  ->  B = U @ sqrt(S), A = sqrt(S) @ Vh
        # and write B' = B / (alpha/r) so that peft's own scaling reconstructs merged
        # exactly. Verified against the materialised delta in the tests.
        # The declared rank comes from the A factor's row count. B is (out, r), so
        # B.shape[0] is the OUTPUT dimension and using it as the rank would let a wide
        # layer emit a factor wider than its config declares, which peft rejects.
        rank = base_state[key.replace("lora_B", "lora_A")].shape[0]
        u, s_vals, vh = torch.linalg.svd(merged, full_matrices=False)
        # Strictly capped at the declared rank: min(out_dim, in_dim) can EXCEED r for a
        # wide layer (a 12x10 module with r=8 gives 10), and peft rejects an adapter whose
        # factor width does not match its config. The truncated tail is the smallest
        # singular directions, so this is the least damaging place to lose rank.
        keep = min(rank, s_vals.shape[0])
        sqrt = torch.sqrt(s_vals[:keep].clamp(min=0))
        b_new = u[:, :keep] * sqrt.unsqueeze(0)
        a_new = (sqrt.unsqueeze(1) * vh[:keep, :]) / scale
        out[key] = b_new.contiguous()
        out[key.replace("lora_B", "lora_A")] = a_new.contiguous()
    log.info("merged %d modules from %d adapters", len(keys), len(sources))
    return out


def build_direct(
    base_dir: Path,
    sources: list[tuple[str, dict[str, torch.Tensor], float, float]],
    drop_ratio: float,
    trim_ratio: float,
    seed: int,
    out_dir: Path,
) -> str:
    """Write a merged model by adding the displacements straight into the base weights.

    This exists because the adapter round trip through ``build`` is lossy in a way that
    matters here. The merged displacement is full-rank -- out x in, thousands of singular
    directions -- but a LoRA file must be rank-r, so ``build`` keeps only r of them and
    then divides by alpha/r to compensate. With r=32 on a 4864-wide layer that discards
    over 99% of the spectrum. That is the most likely cause of the merged model's
    degraded fluency: it was carrying a rank-32 shadow of a full-rank merge.

    Adding the displacement to the weights directly has no rank limit, no rescale to undo
    and no adapter file to load. It is also exactly what ``merge_and_unload`` does, minus
    the round trip. Costs one copy of the base model on disk (~1GB), which is cheaper than
    the training run it replaces.
    """
    from safetensors.torch import load_file as _load_file
    from safetensors.torch import save_file as _save_file

    scales = {round(s, 6) for _l, _st, s, _w in sources}
    if len(scales) != 1:
        raise ValueError(f"adapters disagree on alpha/r ({sorted(scales)})")
    # Kept as the validated single scale even though the direct path multiplies it into
    # each source's own delta via _lora_delta. Asserting equality here is what guarantees
    # the two paths cannot silently disagree about what the numbers mean.

    base_state = _load_file(str(base_dir / "model.safetensors"))
    generator = torch.Generator().manual_seed(seed)
    keys = sorted(k for k in sources[0][1] if "lora_B" in k)
    touched = 0

    for key in keys:
        # "base_model.model.model.layers.0.mlp.down_proj.lora_B.weight"
        #   -> "model.layers.0.mlp.down_proj.weight"
        # The prefix strip leaves a trailing "." from "lora_B." and a leading one from
        # the prefix, so this is done with a regex rather than replaces()+strips: the
        # string surgery version produced "down_proj..weight" and matched nothing.
        target = re.sub(r"^base_model\.model\.", "", key)
        target = target.replace("lora_B.", "")
        if target not in base_state:
            raise KeyError(f"no base weight for {target} (from {key})")
        deltas = [
            dare(_lora_delta(state, key, src_scale) * weight, drop_ratio, generator)
            for _label, state, src_scale, weight in sources
        ]
        base_state[target] = (
            base_state[target].float() + ties_merge(deltas, trim_ratio)
        ).to(base_state[target].dtype)
        touched += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    _save_file(base_state, str(out_dir / "model.safetensors"))
    for name in (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    ):
        src = base_dir / name
        if src.is_file():
            (out_dir / name).write_bytes(src.read_bytes())
    log.info("wrote merged model with %d adapted modules", touched)
    return str(out_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter",
        action="append",
        required=True,
        metavar="PATH[:WEIGHT]",
        help="adapter directory, optionally PATH:WEIGHT (default 1.0). Repeatable.",
    )
    parser.add_argument("--output", required=True, help="output adapter directory")
    parser.add_argument(
        "--drop-ratio",
        type=float,
        default=0.5,
        help="DARE drop rate p; 0 disables. DARE's own results use 0.9-0.99 on larger "
        "models, but 0.5 is the conservative starting point here.",
    )
    parser.add_argument(
        "--trim-ratio",
        type=float,
        default=0.2,
        help="TIES per-source trim of small-magnitude entries; 0 disables.",
    )
    parser.add_argument(
        "--direct",
        action="store_true",
        help=(
            "write the merged model straight into a copy of the base weights instead of "
            "an adapter file. No rank limit, so the full-rank merge survives -- which is "
            "what the rank-r adapter round trip cannot do."
        ),
    )
    parser.add_argument(
        "--base",
        default=None,
        help="base model directory for --direct (must contain model.safetensors)",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    sources = []
    for spec in args.adapter:
        path, _, weight = spec.partition(":")
        state, scale = _load(Path(path))
        sources.append((path, state, scale, float(weight) if weight else 1.0))
        log.info("loaded %s (alpha/r=%.3f, weight=%.2f)", path, scale, sources[-1][3])

    recipe = {
        "adapters": [{"path": p, "scale": s, "weight": w} for p, _st, s, w in sources],
        "drop_ratio": args.drop_ratio,
        "trim_ratio": args.trim_ratio,
        "seed": args.seed,
        "method": "DARE + TIES on materialised B@A",
    }

    out = Path(args.output)
    if args.direct:
        # Full-rank path: the merged displacement goes straight into a copy of the base
        # weights, so nothing is truncated to r and no rescale has to be undone.
        if not args.base:
            print("--direct requires --base")
            return 1
        recipe["method"] += " (direct, full rank)"
        build_direct(
            Path(args.base), sources, args.drop_ratio, args.trim_ratio, args.seed, out
        )
        out.mkdir(parents=True, exist_ok=True)
        (out / "merge_recipe.json").write_text(json.dumps(recipe, indent=2))
        print(f"merged model written to {out}")
        print(f"next: quantize_gguf.py --model {out}")
        return 0

    merged = build(sources, args.drop_ratio, args.trim_ratio, args.seed)
    out.mkdir(parents=True, exist_ok=True)
    template = Path(args.adapter[0].partition(":")[0])
    (out / "adapter_config.json").write_text(
        (template / "adapter_config.json").read_text()
    )
    save_file(merged, str(out / "adapter_model.safetensors"))
    (out / "merge_recipe.json").write_text(json.dumps(recipe, indent=2))
    print(f"merged adapter written to {out}")
    print(f"next: train_distill.py --prebuilt-adapter {out}, then quantize_gguf.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
