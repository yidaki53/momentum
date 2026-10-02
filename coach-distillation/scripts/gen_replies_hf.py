#!/usr/bin/env python3
"""Generate bench replies from a HF/safetensors checkpoint on the GPU.

The bench can score pre-generated replies (``--replies``), but its own generation path
goes through llama.cpp because that is what ships. This is the transformer-side
equivalent, used to score a merged or abliterated checkpoint *before* it has been
quantised -- which is the only place a regression is cheap to catch.

Uses Momentum's own prompt builder so the model is prompted exactly as the app prompts
it. Scoring a model on a prompt it will never see measures nothing.

Usage::

    python scripts/gen_replies_hf.py --model out/student-merged \\
        --set eval/golden_qa.jsonl --out out/replies.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from momentum.llm.prompts import build_chat_prompt  # noqa: E402

log = logging.getLogger("gen_replies_hf")


def generate(
    model_path: Path,
    rows: list[dict[str, Any]],
    *,
    max_new_tokens: int,
    batch_note: str = "",
) -> list[dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(str(model_path), dtype=dtype)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    outputs: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        prompt = row.get("prompt", "")
        user_data = row.get("user_data", "")
        messages = build_chat_prompt(prompt, user_data, [])
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            result = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                # Greedy: the bench measures behaviour, not sampling luck. The app's
                # own temperature is a separate question.
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        reply = tokenizer.decode(
            result[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        outputs.append(
            {
                "id": row.get("id", index),
                "prompt": prompt,
                "user_data": user_data,
                "response": reply.strip(),
                "axes": row.get("axes"),
            }
        )
        log.info(
            "%d/%d %s -> %s",
            index,
            len(rows),
            prompt[:40],
            reply.strip()[:60].replace("\n", " "),
        )
    if batch_note:
        log.info(batch_note)
    return outputs


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="out/student-merged")
    parser.add_argument("--set", default="eval/golden_qa.jsonl")
    parser.add_argument("--out", default="out/replies.jsonl")
    parser.add_argument("--max-new-tokens", type=int, default=140)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    rows = [
        json.loads(line)
        for line in Path(args.set).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    outputs = generate(Path(args.model), rows, max_new_tokens=args.max_new_tokens)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in outputs:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"wrote {len(outputs)} replies to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
