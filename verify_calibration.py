#!/usr/bin/env python3
"""Verify scoring recalibration against fiction, non-fiction, and LLM samples."""

import json
import random
import re
import numpy as np

from analyze_results import analyze_text_quality

# Paths
FICTION_PATH = "/home/aibox/marvin_no_anthologies.json"
NONFIC_PATH = "/home/aibox/.cache/huggingface/hub/datasets--allura-org--the-anarchist-library/snapshots/d5765192375372655703a6e70c1d82a3307531e0/tal_texts.jsonl"
LLM_PATH = "/home/aibox/sampler-sweeper/eval_all_prompts_results.json"

N_SAMPLES = 40
random.seed(42)


def load_fiction(n=N_SAMPLES):
    with open(FICTION_PATH) as f:
        data = json.load(f)
    samples = []
    idxs = random.sample(range(len(data)), min(n, len(data)))
    for i in idxs:
        text = data[i].get("text", "")
        # Take ~500 words from middle
        words = text.split()
        if len(words) > 600:
            start = len(words) // 4
            text = " ".join(words[start:start+500])
        if len(text.split()) >= 50:
            samples.append(text)
    return samples[:n]


def load_nonfiction(n=N_SAMPLES):
    entries = []
    with open(NONFIC_PATH) as f:
        for line in f:
            entries.append(json.loads(line))
    # Filter for reasonable-length entries
    good = []
    for e in entries:
        raw = e.get("text", "")
        # Strip header lines
        lines = raw.split("\n")
        body_lines = [l for l in lines if not l.startswith("#") and l.strip()]
        body = "\n".join(body_lines)
        words = body.split()
        if len(words) >= 200:
            good.append(body)
    chosen = random.sample(good, min(n, len(good)))
    samples = []
    for text in chosen:
        words = text.split()
        if len(words) > 600:
            start = len(words) // 4
            text = " ".join(words[start:start+500])
        samples.append(text)
    return samples[:n]


def load_llm():
    with open(LLM_PATH) as f:
        data = json.load(f)
    return [r["text"] for r in data if "text" in r]


def score_samples(samples, label):
    metrics = []
    for text in samples:
        q = analyze_text_quality(text)
        metrics.append(q)

    keys = ['quality_score', 'coherence_score', 'readability_score',
            'lazy_score', 'prose_penalty', 'repetition_penalty',
            'slop_guard_score']

    print(f"\n{'='*60}")
    print(f"  {label} ({len(samples)} samples)")
    print(f"{'='*60}")
    print(f"  {'Metric':<25} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    means = {}
    for k in keys:
        vals = [m.get(k, 0) for m in metrics]
        mn = np.mean(vals)
        means[k] = mn
        print(f"  {k:<25} {mn:>8.3f} {np.std(vals):>8.3f} {np.min(vals):>8.3f} {np.max(vals):>8.3f}")

    return means


def main():
    print("Loading samples...")
    fiction = load_fiction()
    nonfiction = load_nonfiction()
    llm = load_llm()
    print(f"  Fiction: {len(fiction)}, Non-fiction: {len(nonfiction)}, LLM: {len(llm)}")

    fic_m = score_samples(fiction, "FICTION (published)")
    nf_m = score_samples(nonfiction, "NON-FICTION (anarchist library)")
    llm_m = score_samples(llm, "LLM (Devstral at default params)")

    print(f"\n{'='*60}")
    print("  COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Metric':<25} {'Fiction':>8} {'Non-fic':>8} {'LLM':>8} {'Fic-LLM':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for k in ['quality_score', 'coherence_score', 'readability_score',
              'lazy_score', 'prose_penalty', 'repetition_penalty']:
        gap = fic_m[k] - llm_m[k]
        print(f"  {k:<25} {fic_m[k]:>8.3f} {nf_m[k]:>8.3f} {llm_m[k]:>8.3f} {gap:>+8.3f}")


if __name__ == "__main__":
    main()
