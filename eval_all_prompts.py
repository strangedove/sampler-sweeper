#!/usr/bin/env python3
"""
Run all prompts from the sweeper pool with fixed sampler parameters
and evaluate the quality of each output.

Usage:
    uv run eval_all_prompts.py
"""

import json
import re
import sys
from openai import OpenAI
from analyze_results import analyze_text_quality
from optuna_parameter_sweep import BUILTIN_PROMPT_POOL

# Best params from Magistral sweep trial 26
PARAMS = {
    "temperature": 1.452,
    "min_p": 0.097,
    "adaptive_target": 0.480,
    "adaptive_decay": 0.919,
}

API_BASE = "https://alto-august-featuring-capture.trycloudflare.com/v1"
API_KEY = "not-needed"
MAX_TOKENS = 512

NATIVE_OPENAI_PARAMS = {'temperature', 'top_p', 'max_tokens', 'frequency_penalty', 'presence_penalty'}


def strip_thinking(text: str) -> str:
    if not text:
        return text
    text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()
    text = re.sub(r'<think>.*', '', text, flags=re.DOTALL).strip()
    return text


def generate(client, model, prompt_entry, params):
    native = {k: v for k, v in params.items() if k in NATIVE_OPENAI_PARAMS}
    extra = {k: v for k, v in params.items() if k not in NATIVE_OPENAI_PARAMS}

    if prompt_entry["format"] == "chat":
        kwargs = dict(model=model, messages=prompt_entry["messages"],
                      max_tokens=MAX_TOKENS, **native)
        if extra:
            kwargs['extra_body'] = extra
        resp = client.chat.completions.create(**kwargs)
        raw = resp.choices[0].message.content
    else:
        kwargs = dict(model=model, prompt=prompt_entry["text"],
                      max_tokens=MAX_TOKENS, **native)
        if extra:
            kwargs['extra_body'] = extra
        resp = client.completions.create(**kwargs)
        raw = resp.choices[0].text

    return strip_thinking(raw)


def main():
    client = OpenAI(base_url=API_BASE, api_key=API_KEY)

    # Auto-detect model
    models = client.models.list()
    model = models.data[0].id
    print(f"Model: {model}")
    print(f"Params: {PARAMS}")
    print(f"Max tokens: {MAX_TOKENS}")
    print(f"Prompts: {len(BUILTIN_PROMPT_POOL)}")
    print("=" * 70)
    print()

    results = []

    for pid, entry in BUILTIN_PROMPT_POOL.items():
        print(f"--- {pid} ({entry.get('category', '?')}) ---")
        try:
            text = generate(client, model, entry, PARAMS)
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"prompt_id": pid, "error": str(e)})
            print()
            continue

        if not text or len(text.strip()) < 10:
            print(f"  Empty or too short ({len(text) if text else 0} chars)")
            results.append({"prompt_id": pid, "error": "empty"})
            print()
            continue

        quality = analyze_text_quality(text)
        score = quality['quality_score']

        print(f"  Score: {score:.3f}  |  coherence={quality['coherence_score']:.2f}  "
              f"readability={quality['readability_score']:.2f}  lazy={quality['lazy_score']:.2f}  "
              f"prose_pen={quality['prose_penalty']:.2f}  rep_pen={quality['repetition_penalty']:.2f}  "
              f"slop={quality.get('slop_guard_score', 50):.0f}")

        # Show slop patterns if any
        pats = quality.get('lazy_patterns_found', [])
        if pats:
            print(f"  Slop hits: {', '.join(f'{p} ({c}x)' for p, c in pats[:8])}")

        print()
        print(text.strip()[:1500])
        print()

        results.append({
            "prompt_id": pid,
            "category": entry.get('category', '?'),
            "score": score,
            "coherence": quality['coherence_score'],
            "readability": quality['readability_score'],
            "lazy": quality['lazy_score'],
            "prose_penalty": quality['prose_penalty'],
            "repetition_penalty": quality['repetition_penalty'],
            "slop_guard_score": quality.get('slop_guard_score', 50),
            "slop_guard_band": quality.get('slop_guard_band', 'unknown'),
            "length": len(text),
            "text": text.strip(),
        })

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    scored = [r for r in results if 'score' in r]
    if scored:
        scores = [r['score'] for r in scored]
        print(f"  Prompts evaluated: {len(scored)} / {len(BUILTIN_PROMPT_POOL)}")
        print(f"  Mean score: {sum(scores)/len(scores):.3f}")
        print(f"  Min score:  {min(scores):.3f}")
        print(f"  Max score:  {max(scores):.3f}")
        print()
        print(f"  {'Prompt':<30} {'Cat':<20} {'Score':>6}")
        print(f"  {'-'*30} {'-'*20} {'-'*6}")
        for r in sorted(scored, key=lambda x: x['score'], reverse=True):
            print(f"  {r['prompt_id']:<30} {r['category']:<20} {r['score']:>6.3f}")
    else:
        print("  No prompts produced valid output!")

    # Save full results
    out_file = "eval_all_prompts_results.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nFull results saved to: {out_file}")


if __name__ == "__main__":
    main()
