#!/usr/bin/env python3
"""
Optuna-based parameter sweep for LLM sampling parameters

This script uses Optuna to optimize LLM sampling parameters by maximizing
the text quality score calculated from analyze_results.py.

Supports multiple evaluation prompts per trial (chat and completion formats)
to measure cross-prompt consistency and avoid overfitting to a single task.

Usage:
    uv run optuna_parameter_sweep.py --config sweep_config.yaml
    uv run optuna_parameter_sweep.py   # uses defaults / env vars

Dependencies:
    optuna, openai, pandas, numpy, pyyaml
    (same as analyze_results.py)
"""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import hashlib
import optuna
import json
import os
import random
import re
import time
from typing import Dict, Any, List, Optional
import yaml
import pandas as pd
import numpy as np
from openai import OpenAI
from analyze_results import analyze_text_quality


# ------------------------------------------------------------------
# Default parameter space (used when no config file is provided)
# ------------------------------------------------------------------
DEFAULT_PARAMETERS = {
    'temperature': [0.1, 2.0],
    'min_p': [0.0, 0.3],
    'top_k': [0, 100],
    'adaptive_target': [0.3, 0.7],
    'adaptive_decay': [0.5, 0.9],
}

# Parameters the OpenAI client accepts natively (not via extra_body)
NATIVE_OPENAI_PARAMS = {'temperature', 'top_p', 'max_tokens', 'frequency_penalty', 'presence_penalty'}


def load_config(config_path: str) -> Dict[str, Any]:
    """Load sweep configuration from a YAML file."""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def parse_parameter_space(params_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parse parameter definitions from config into a structured list.

    Each entry in params_dict maps a name to a range:
        temperature: [0.1, 2.0]         -> float
        top_k: [0, 100]                 -> int  (both bounds are int)
        top_k: [0, 100, "int"]          -> int  (explicit)
        min_p: [0.0, 0.3, "float"]      -> float (explicit)
    """
    parsed = []
    for name, spec in params_dict.items():
        if not isinstance(spec, list) or len(spec) < 2:
            raise ValueError(f"Parameter '{name}' must be a list of [min, max] or [min, max, type]")
        lo, hi = spec[0], spec[1]
        # Determine type
        if len(spec) >= 3 and isinstance(spec[2], str):
            ptype = spec[2].lower()
        elif isinstance(lo, int) and isinstance(hi, int):
            ptype = 'int'
        else:
            ptype = 'float'
        parsed.append({'name': name, 'low': lo, 'high': hi, 'type': ptype})
    return parsed


# ------------------------------------------------------------------
# Default prompt (raw text continuation)
# ------------------------------------------------------------------
DEFAULT_PROMPT = """Continue the following story in vivid, creative prose:

The morning Dagna found the dead radio tower, she was already three days late getting back to town.
She sat on a lichen-covered boulder and ate the last of her dried fish while studying the structure.
The cables had been cut -- not frayed, not corroded, but sliced clean through with something sharp.
"""

# ------------------------------------------------------------------
# Built-in prompt pool
# ------------------------------------------------------------------
# 8 prompts covering different creative tasks: continuation, instruct-writing,
# character roleplay, interactive fiction, and creative assistant. Each has a
# format ("completion" or "chat") and the appropriate content.
BUILTIN_PROMPT_POOL: Dict[str, Dict[str, Any]] = {
    "continuation_default": {
        "format": "completion",
        "text": DEFAULT_PROMPT,
        "category": "Continuation",
    },
    "instruct_writing_1": {
        "format": "chat",
        "messages": [
            {"role": "system", "content": "You are a creative writing assistant. Write what the user requests."},
            {"role": "user", "content": "Write a short scene where a detective discovers an unexpected clue in an abandoned greenhouse. Focus on atmosphere and sensory details."},
        ],
        "category": "Instruct-Writing",
    },
    "instruct_writing_2": {
        "format": "chat",
        "messages": [
            {"role": "system", "content": "You are a creative writing assistant. Write what the user requests."},
            {"role": "user", "content": "Write the opening paragraph of a story about a lighthouse keeper who receives a mysterious radio transmission. Now have her realize the voice sounds exactly like her own."},
        ],
        "category": "Instruct-Writing",
    },
    "character_roleplay_1": {
        "format": "chat",
        "messages": [
            {"role": "system", "content": "You are playing the role of Marcus, a weary traveling merchant in a fantasy world. The user is playing Kira, a cloaked figure with a hidden agenda. Respond in character with both dialogue and narrative actions."},
            {"role": "user", "content": "*Kira approaches the merchant's cart as he's setting up in the village square, keeping her hood low. She speaks in a hushed tone.* \"You're the one they call Marcus, yes? I've heard you carry... special goods. The kind not meant for ordinary customers.\""},
        ],
        "category": "Character Roleplay",
    },
    "character_roleplay_2": {
        "format": "chat",
        "messages": [
            {"role": "system", "content": "You are playing the role of Sera, a retired assassin now running a quiet tavern in a border town. The user is playing Damon, a young bounty hunter who has just learned Sera's true identity. Respond in character with both dialogue and narrative actions."},
            {"role": "user", "content": "*Damon sets down his drink and slides a worn wanted poster across the bar \u2014 it shows a younger version of the woman standing in front of him. He keeps his voice low so the other patrons can't hear.* \"Funny thing about old bounties. They don't expire.\""},
        ],
        "category": "Character Roleplay",
    },
    "interactive_fiction_1": {
        "format": "chat",
        "messages": [
            {"role": "system", "content": "You are the narrator of an interactive fiction game. Describe the world in second person (\"You see...\") and respond to the player's actions with vivid descriptions of what happens."},
            {"role": "user", "content": "I'm in an abandoned subway station. I look around \u2014 what do I see?"},
        ],
        "category": "Interactive Fiction",
    },
    "interactive_fiction_2": {
        "format": "chat",
        "messages": [
            {"role": "system", "content": "You are the narrator of an interactive fiction game. Describe the world in second person (\"You see...\") and respond to the player's actions with vivid descriptions of what happens."},
            {"role": "user", "content": "[Start a new game: Post-apocalyptic survival horror. I'm a scavenger searching for supplies in a ruined city.]"},
            {"role": "assistant", "content": "You stand at the edge of what used to be a shopping district. Rusted cars choke the street ahead, and most storefronts have long since been picked clean. But there's one building that catches your eye \u2014 a pharmacy with its security shutters still half-closed. Through the gap, you can see shelves that might not be completely empty. The afternoon light is fading, and you know you shouldn't be out here after dark."},
            {"role": "user", "content": "I crouch low and squeeze through the gap in the shutters."},
        ],
        "category": "Interactive Fiction",
    },
    "general_assistant_6": {
        "format": "chat",
        "messages": [
            {"role": "system", "content": "You are a friendly, knowledgeable assistant. Answer helpfully and concisely."},
            {"role": "user", "content": "Explain the meaning of \"abvalid\" in the context of horror fiction, and describe what makes it different from other, similar words. Being correct in a formal sense is less important than engaging with the task: don't concern yourself with whether its established or perhaps confused, work with what is given."},
        ],
        "category": "Creative Assistant",
    },
}


# ------------------------------------------------------------------
# Prompt pool
# ------------------------------------------------------------------
@dataclass
class PromptPool:
    """Manages the set of prompts used during a sweep."""
    mode: str  # "sample", "all", or "random"
    samples_per_trial: int
    prompts: Dict[str, Dict[str, Any]]

    def select_for_trial(self, rng: random.Random) -> Dict[str, Dict[str, Any]]:
        """Return the prompts to evaluate for a single trial."""
        if self.mode == "all":
            return dict(self.prompts)
        elif self.mode == "random":
            pid = rng.choice(list(self.prompts.keys()))
            return {pid: self.prompts[pid]}
        else:  # "sample"
            k = min(self.samples_per_trial, len(self.prompts))
            selected_ids = rng.sample(list(self.prompts.keys()), k)
            return {pid: self.prompts[pid] for pid in selected_ids}


def build_prompt_pool(cfg: Dict[str, Any]) -> PromptPool:
    """Build a PromptPool from config.

    Handles three cases:
    1. Old-style 'prompt:' key (scalar string) -> single-prompt pool, mode="all"
    2. New 'prompts:' section with mode/pool/custom entries
    3. Neither -> full built-in pool, mode="sample", samples_per_trial=3
    """
    if 'prompt' in cfg and 'prompts' not in cfg:
        # Backward compat: single prompt string
        return PromptPool(
            mode="all",
            samples_per_trial=1,
            prompts={"custom_single": {
                "format": "completion",
                "text": cfg['prompt'],
                "category": "Custom",
            }},
        )

    prompts_cfg = cfg.get('prompts', {})
    mode = prompts_cfg.get('mode', 'sample')
    samples_per_trial = prompts_cfg.get('samples_per_trial', 3)

    # Build the prompt dict
    pool_ids = prompts_cfg.get('pool', None)
    prompts: Dict[str, Dict[str, Any]] = {}

    if pool_ids is not None:
        for pid in pool_ids:
            if pid in BUILTIN_PROMPT_POOL:
                prompts[pid] = BUILTIN_PROMPT_POOL[pid]
            else:
                raise ValueError(f"Unknown built-in prompt ID: '{pid}'. "
                                 f"Available: {list(BUILTIN_PROMPT_POOL.keys())}")
    else:
        prompts = dict(BUILTIN_PROMPT_POOL)

    # Add custom prompts from config
    for custom in prompts_cfg.get('custom', []):
        cid = custom.get('id', f'custom_{len(prompts)}')
        if 'messages' in custom:
            prompts[cid] = {
                "format": "chat",
                "messages": custom['messages'],
                "category": custom.get('category', "Custom"),
            }
        elif 'text' in custom:
            prompts[cid] = {
                "format": "completion",
                "text": custom['text'],
                "category": custom.get('category', "Custom"),
            }
        else:
            raise ValueError(f"Custom prompt '{cid}' must have 'messages' or 'text'")

    if not prompts:
        prompts = dict(BUILTIN_PROMPT_POOL)

    return PromptPool(mode=mode, samples_per_trial=samples_per_trial, prompts=prompts)


# ------------------------------------------------------------------
# Metric averaging helper
# ------------------------------------------------------------------
_AVERAGED_METRIC_KEYS = [
    'repetition_penalty', 'coherence_score', 'readability_score',
    'lazy_score', 'prose_penalty', 'sentence_start_monotony',
    'word_frequency_spike', 'telling_verb_density',
    'slop_guard_score', 'slop_guard_violations',
]


def _average_quality_metrics(prompt_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Average numeric quality metrics across multiple prompt evaluations."""
    n = len(prompt_metrics)
    if n == 0:
        return {k: 0.0 for k in _AVERAGED_METRIC_KEYS}

    result: Dict[str, Any] = {}
    for key in _AVERAGED_METRIC_KEYS:
        values = [m.get(key, 0.0) for m in prompt_metrics.values()]
        result[key] = float(np.mean(values))

    # Response length: average across prompts
    result['response_length'] = float(np.mean(
        [len(m.get('_response_text', '')) for m in prompt_metrics.values()]
    ))

    # Slop guard band: take the worst
    band_order = {'clean': 0, 'light': 1, 'moderate': 2, 'unknown': 2, 'unavailable': 2, 'heavy': 3}
    bands = [m.get('slop_guard_band', 'unknown') for m in prompt_metrics.values()]
    result['slop_guard_band'] = max(bands, key=lambda b: band_order.get(b, 2))

    # Lazy patterns: aggregate counts
    all_patterns: List = []
    total_count = 0
    for m in prompt_metrics.values():
        pats = m.get('lazy_patterns_found', [])
        all_patterns.extend(pats)
        total_count += m.get('lazy_pattern_count', 0)
    result['lazy_patterns_found'] = all_patterns
    result['lazy_pattern_count'] = total_count

    return result


class OptunaParameterSweep:
    """Optuna-based parameter sweep for LLM sampling optimization"""

    def __init__(self,
                 base_url: str,
                 model_name: str,
                 prompt_pool: PromptPool,
                 parameter_space: Optional[List[Dict[str, Any]]] = None,
                 max_tokens: int = 1024,
                 max_trials: int = 100,
                 timeout: Optional[int] = None,
                 n_jobs: int = 1,
                 study_name: str = "llm_parameter_optimization",
                 storage: Optional[str] = None,
                 api_key: str = "not-needed",
                 label: str = "",
                 checkpoint_interval: int = 0,
                 top_n: int = 10,
                 convergence_patience: int = 25,
                 convergence_min_delta: float = 0.005,
                 parallel_prompts: bool = False,
                 n_startup_trials: int = 12,
                 api_extra_body: Optional[Dict[str, Any]] = None,
                 assistant_prefill: str = ""):
        self.base_url = base_url
        self.model_name = model_name
        self.prompt_pool = prompt_pool
        self.max_tokens = max_tokens
        self.max_trials = max_trials
        self.timeout = timeout
        self.n_jobs = n_jobs
        self.study_name = study_name
        self.storage = storage
        self.label = label
        self.parameter_space = parameter_space or parse_parameter_space(DEFAULT_PARAMETERS)
        self.checkpoint_interval = checkpoint_interval
        self.top_n = top_n
        self.convergence_patience = convergence_patience
        self.convergence_min_delta = convergence_min_delta
        self.parallel_prompts = parallel_prompts
        self.n_startup_trials = n_startup_trials
        self.api_extra_body = api_extra_body or {}
        self.assistant_prefill = assistant_prefill

        self.client = OpenAI(base_url=base_url, api_key=api_key)

    @property
    def run_suffix(self) -> str:
        """File-name suffix: label > model name > timestamp."""
        raw = self.label or self.model_name or ""
        if raw:
            # Sanitize: keep alphanumerics, hyphens, underscores, dots
            safe = re.sub(r'[^a-zA-Z0-9._-]', '_', raw)
            # Collapse runs of underscores and strip leading/trailing
            safe = re.sub(r'_+', '_', safe).strip('_')
            if safe:
                return safe
        return time.strftime("%Y%m%d_%H%M%S")

    # ------------------------------------------------------------------
    # Optuna objective
    # ------------------------------------------------------------------
    def objective(self, trial: optuna.Trial) -> float:
        """Objective function: evaluate across selected prompts, return mean score."""
        params = {}
        for p in self.parameter_space:
            if p['type'] == 'int':
                params[p['name']] = trial.suggest_int(p['name'], int(p['low']), int(p['high']))
            else:
                params[p['name']] = trial.suggest_float(p['name'], float(p['low']), float(p['high']))

        # Select prompts for this trial (deterministic per trial number)
        seed = trial.number + int(hashlib.md5(self.study_name.encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)
        selected = self.prompt_pool.select_for_trial(rng)

        # During startup trials, use only 1 prompt for faster exploration
        if trial.number < self.n_startup_trials and len(selected) > 1:
            first_key = next(iter(selected))
            selected = {first_key: selected[first_key]}

        trial.set_user_attr('prompt_ids', list(selected.keys()))
        trial.set_user_attr('prompt_mode', self.prompt_pool.mode)
        trial.set_user_attr('num_prompts_evaluated', len(selected))

        # Evaluate each prompt
        prompt_scores: Dict[str, float] = {}
        prompt_texts: Dict[str, str] = {}
        prompt_metrics: Dict[str, Dict[str, Any]] = {}
        failed_prompts = 0

        if self.parallel_prompts:
            # -- Parallel path: evaluate all prompts concurrently --
            def _eval_prompt(pid_entry):
                pid, entry = pid_entry
                try:
                    response_text = self._generate_for_prompt(entry, params)
                except Exception as e:
                    return pid, None, None, str(e)
                if not response_text or len(response_text.strip()) < 20:
                    return pid, None, None, "empty"
                quality = analyze_text_quality(response_text)
                return pid, response_text, quality, None

            with ThreadPoolExecutor(max_workers=len(selected)) as executor:
                futures = {executor.submit(_eval_prompt, item): item
                           for item in selected.items()}
                for future in as_completed(futures):
                    pid, response_text, quality, error = future.result()
                    if error:
                        if error != "empty":
                            print(f"  [trial {trial.number}] API error on '{pid}': {error}")
                        failed_prompts += 1
                        continue
                    score = quality['quality_score']
                    prompt_scores[pid] = score
                    prompt_texts[pid] = response_text
                    quality['_response_text'] = response_text
                    prompt_metrics[pid] = quality
        else:
            # -- Sequential path: per-prompt pruning --
            step = 0
            for pid, entry in selected.items():
                try:
                    response_text = self._generate_for_prompt(entry, params)
                except Exception as e:
                    print(f"  [trial {trial.number}] API error on '{pid}': {e}")
                    failed_prompts += 1
                    continue

                if not response_text or len(response_text.strip()) < 20:
                    failed_prompts += 1
                    continue

                quality = analyze_text_quality(response_text)
                score = quality['quality_score']

                prompt_scores[pid] = score
                prompt_texts[pid] = response_text
                quality['_response_text'] = response_text
                prompt_metrics[pid] = quality

                # Report running mean to pruner
                running_mean = float(np.mean(list(prompt_scores.values())))
                trial.report(running_mean, step)
                step += 1

                if trial.should_prune():
                    trial.set_user_attr('prompt_scores', prompt_scores)
                    trial.set_user_attr('pruned_at_step', step)
                    raise optuna.exceptions.TrialPruned(
                        f"Pruned at step {step}/{len(selected)}, "
                        f"running_mean={running_mean:.3f}"
                    )

        if not prompt_scores:
            raise optuna.exceptions.TrialPruned(
                f"All {len(selected)} prompts failed or produced empty responses"
            )

        # Aggregate scores
        scores = list(prompt_scores.values())
        mean_score = float(np.mean(scores))
        score_variance = float(np.var(scores)) if len(scores) > 1 else 0.0
        score_min = float(min(scores))

        # Store per-prompt data
        trial.set_user_attr('mean_score', mean_score)
        trial.set_user_attr('prompt_scores', prompt_scores)
        trial.set_user_attr('score_variance', score_variance)
        trial.set_user_attr('score_min', score_min)
        trial.set_user_attr('failed_prompts', failed_prompts)

        for pid, text in prompt_texts.items():
            trial.set_user_attr(f'response_text__{pid}', text)

        # Averaged quality metrics across prompts
        avg_metrics = _average_quality_metrics(prompt_metrics)
        for key in _AVERAGED_METRIC_KEYS:
            trial.set_user_attr(key, avg_metrics[key])
        trial.set_user_attr('response_length', avg_metrics['response_length'])
        trial.set_user_attr('slop_guard_band', avg_metrics['slop_guard_band'])
        trial.set_user_attr('lazy_pattern_count', avg_metrics['lazy_pattern_count'])
        trial.set_user_attr('lazy_patterns_found',
                            [[p, c] for p, c in avg_metrics['lazy_patterns_found']])

        # Per-prompt metric summary (compact)
        per_prompt_summary = {}
        for pid, m in prompt_metrics.items():
            per_prompt_summary[pid] = {
                'quality_score': m['quality_score'],
                'repetition_penalty': m['repetition_penalty'],
                'coherence_score': m['coherence_score'],
                'readability_score': m['readability_score'],
                'lazy_score': m.get('lazy_score', 0.0),
                'prose_penalty': m.get('prose_penalty', 0.0),
                'slop_guard_score': m.get('slop_guard_score', 50.0),
            }
        trial.set_user_attr('per_prompt_metrics', per_prompt_summary)

        if mean_score < 0.1:
            raise optuna.exceptions.TrialPruned(f"Low mean quality score: {mean_score}")

        return mean_score

    # ------------------------------------------------------------------
    # LLM generation
    # ------------------------------------------------------------------
    @staticmethod
    def _strip_thinking(text: str) -> str:
        """Strip thinking blocks from reasoning-model output.

        Handles multiple formats: <think>...</think> and [THINK]...[/THINK].
        """
        if not text:
            return text
        import re
        # Strip closed think blocks — both <think> and [THINK] formats
        text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()
        text = re.sub(r'\[THINK\].*?\[/THINK\]\s*', '', text, flags=re.DOTALL).strip()
        # Strip unclosed think block (model ran out of tokens mid-thinking)
        text = re.sub(r'<think>.*', '', text, flags=re.DOTALL).strip()
        text = re.sub(r'\[THINK\].*', '', text, flags=re.DOTALL).strip()
        return text

    def _generate_for_prompt(self, prompt_entry: Dict[str, Any], params: Dict[str, Any]) -> str:
        """Dispatch to the correct API based on prompt format."""
        if prompt_entry["format"] == "chat":
            raw = self._generate_chat_response(prompt_entry["messages"], params)
        else:
            raw = self._generate_completion_response(prompt_entry["text"], params)
        return self._strip_thinking(raw)

    def _generate_completion_response(self, prompt_text: str, params: Dict[str, Any]) -> str:
        """Generate via client.completions.create (raw text continuation)."""
        native = {k: v for k, v in params.items() if k in NATIVE_OPENAI_PARAMS}
        extra = {k: v for k, v in params.items() if k not in NATIVE_OPENAI_PARAMS}
        extra = {**self.api_extra_body, **extra}  # config defaults, overridden by sweep params

        prompt = prompt_text
        if self.assistant_prefill:
            prompt = prompt_text + self.assistant_prefill

        kwargs: Dict[str, Any] = dict(
            model=self.model_name,
            prompt=prompt,
            max_tokens=self.max_tokens,
            **native,
        )
        if extra:
            kwargs['extra_body'] = extra

        response = self.client.completions.create(**kwargs)
        return response.choices[0].text

    def _generate_chat_response(self, messages: List[Dict[str, str]], params: Dict[str, Any]) -> str:
        """Generate via client.chat.completions.create (chat format)."""
        native = {k: v for k, v in params.items() if k in NATIVE_OPENAI_PARAMS}
        extra = {k: v for k, v in params.items() if k not in NATIVE_OPENAI_PARAMS}
        extra = {**self.api_extra_body, **extra}  # config defaults, overridden by sweep params

        kwargs: Dict[str, Any] = dict(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
            **native,
        )
        if extra:
            kwargs['extra_body'] = extra

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    # ------------------------------------------------------------------
    # Convergence callback
    # ------------------------------------------------------------------
    class _ConvergenceCallback:
        """Early-stop the study if the best score hasn't improved recently.

        Two counters run in parallel:
        - completed_patience: counts only COMPLETE trials (precise signal)
        - total_patience (2x completed): counts COMPLETE + PRUNED trials
          (noisier but catches heavy-pruning scenarios where most new
          suggestions can't beat the median)
        Whichever counter fires first stops the study.
        """

        def __init__(self, patience: int, min_delta: float):
            self.patience = patience
            self.total_patience = patience * 2
            self.min_delta = min_delta
            self.best_value: Optional[float] = None
            self.completed_since_improvement: int = 0
            self.total_since_improvement: int = 0

        def __call__(self, study: optuna.Study, trial: optuna.Trial) -> None:
            if trial.state not in (optuna.trial.TrialState.COMPLETE,
                                   optuna.trial.TrialState.PRUNED):
                return

            # Check for improvement (only on completed trials)
            if trial.state == optuna.trial.TrialState.COMPLETE:
                current_best = study.best_value
                if (self.best_value is None
                        or current_best > self.best_value + self.min_delta):
                    self.best_value = current_best
                    self.completed_since_improvement = 0
                    self.total_since_improvement = 0
                    return
                self.completed_since_improvement += 1

            # Both completed (non-improving) and pruned increment total
            self.total_since_improvement += 1

            if self.completed_since_improvement >= self.patience:
                print(f"\n  [convergence] No improvement > {self.min_delta} in "
                      f"{self.patience} completed trials. "
                      f"Best={self.best_value:.4f}. Stopping.")
                study.stop()
            elif self.total_since_improvement >= self.total_patience:
                print(f"\n  [convergence] No improvement in "
                      f"{self.total_since_improvement} total trials "
                      f"(completed + pruned). "
                      f"Best={self.best_value:.4f}. Stopping.")
                study.stop()

    # ------------------------------------------------------------------
    # Study runner
    # ------------------------------------------------------------------
    def run_study(self) -> optuna.Study:
        """Run the Optuna optimization study."""
        pool = self.prompt_pool
        print(f"Starting Optuna parameter sweep")
        print(f"   API: {self.base_url}")
        print(f"   Model: {self.model_name}")
        print(f"   Max trials: {self.max_trials}")
        print(f"   Parallel jobs: {self.n_jobs}")
        print(f"   Study name: {self.study_name}")
        mode_desc = pool.mode
        if pool.mode == "sample":
            mode_desc += f" (K={pool.samples_per_trial})"
        print(f"   Prompt pool: {len(pool.prompts)} prompts, mode={mode_desc}")
        sampler_info = (f"TPE (multivariate, n_startup={self.n_startup_trials}"
                        + (", constant_liar" if self.n_jobs > 1 else "") + ")")
        print(f"   Sampler: {sampler_info}")
        if pool.mode == "sample" and pool.samples_per_trial > 1:
            print(f"   Startup trials: 1 prompt each (then K={pool.samples_per_trial})")
        if self.convergence_patience > 0:
            print(f"   Convergence: patience={self.convergence_patience} completed "
                  f"/ {self.convergence_patience * 2} total, "
                  f"min_delta={self.convergence_min_delta}")
        if self.parallel_prompts:
            print(f"   Parallel prompts: enabled (per-prompt pruning disabled)")
        print()

        sampler_kwargs = {
            'n_startup_trials': self.n_startup_trials,
            'multivariate': True,
        }
        if self.n_jobs > 1:
            sampler_kwargs['constant_liar'] = True
        sampler = optuna.samplers.TPESampler(**sampler_kwargs)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1)

        study = optuna.create_study(
            study_name=self.study_name,
            sampler=sampler,
            pruner=pruner,
            direction='maximize',
            storage=self.storage,
            load_if_exists=True,
        )

        callbacks = [self._progress_callback]
        if self.convergence_patience > 0:
            callbacks.append(
                self._ConvergenceCallback(self.convergence_patience,
                                          self.convergence_min_delta)
            )

        study.optimize(
            self.objective,
            n_trials=self.max_trials,
            timeout=self.timeout,
            n_jobs=self.n_jobs,
            callbacks=callbacks,
        )

        return study

    def _progress_callback(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """Print progress every 5 trials and save checkpoints at configured intervals."""
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        n_completed = len(completed)

        if trial.number % 5 == 0 or trial.number == 0:
            pruned = len(study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED]))
            n_prompts = trial.user_attrs.get('num_prompts_evaluated', 1)
            var_str = ""
            if n_prompts > 1:
                sv = trial.user_attrs.get('score_variance', 0)
                var_str = f"  var={sv:.3f}"
            best_str = f"best={study.best_value:.4f}" if n_completed > 0 else "best=N/A"
            print(f"[{trial.number + 1}/{self.max_trials}]  "
                  f"{best_str}  pruned={pruned}  "
                  f"prompts={n_prompts}{var_str}  "
                  f"params={study.best_params if n_completed > 0 else '{}'}")

        # Save checkpoint at configured intervals
        if (self.checkpoint_interval > 0
                and n_completed > 0
                and (trial.number + 1) % self.checkpoint_interval == 0):
            self._save_checkpoint(study, trial.number + 1)

    def _save_checkpoint(self, study: optuna.Study, trial_num: int) -> None:
        """Save intermediate results as a checkpoint."""
        suffix = self.run_suffix
        cp_json = f"checkpoint_{suffix}_t{trial_num}.json"
        cp_report = f"checkpoint_{suffix}_t{trial_num}.md"

        try:
            self.save_results(study, filename=cp_json)
            df = self.analyze_results(study)
            if len(df) > 0:
                self.generate_top_results_report(study, df, top_n=self.top_n,
                                                 filename=cp_report)
            print(f"  [checkpoint] Saved at trial {trial_num}: {cp_json}, {cp_report}")
        except Exception as e:
            print(f"  [checkpoint] Warning: failed to save checkpoint at trial {trial_num}: {e}")

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    def _get_response_texts(self, trial: optuna.Trial) -> Dict[str, str]:
        """Collect per-prompt response texts from trial user attrs."""
        texts = {}
        for key, val in trial.user_attrs.items():
            if key.startswith('response_text__'):
                pid = key[len('response_text__'):]
                texts[pid] = val
        return texts

    def save_results(self, study: optuna.Study, filename: Optional[str] = None) -> str:
        """Save optimization results to JSON file."""
        if filename is None:
            filename = f"optuna_parameter_sweep_results_{self.run_suffix}.json"

        results = []
        for trial in study.trials:
            response_texts = self._get_response_texts(trial)

            result = {
                'trial_number': trial.number,
                'state': trial.state.name,
                'value': trial.value,
                'parameters': trial.params,
                'user_attrs': {
                    k: v for k, v in trial.user_attrs.items()
                    if not k.startswith('response_text')
                },
                'response_texts': response_texts,
                'prompt_scores': trial.user_attrs.get('prompt_scores', {}),
                'score_variance': trial.user_attrs.get('score_variance', 0.0),
                'score_min': trial.user_attrs.get('score_min', 0.0),
                'per_prompt_metrics': trial.user_attrs.get('per_prompt_metrics', {}),
                'prompt_ids': trial.user_attrs.get('prompt_ids', []),
                'datetime_start': trial.datetime_start.isoformat() if trial.datetime_start else None,
                'datetime_complete': trial.datetime_complete.isoformat() if trial.datetime_complete else None,
            }
            results.append(result)

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"Results saved to: {filename}")
        return filename

    def analyze_results(self, study: optuna.Study) -> pd.DataFrame:
        """Convert completed trials into a DataFrame for analysis."""
        data = []
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                response_texts = self._get_response_texts(trial)
                first_text = next(iter(response_texts.values()), '')

                row = {
                    'trial': trial.number,
                    'value': trial.value,
                    **trial.params,
                    'response_length': trial.user_attrs.get('response_length', 0),
                    'repetition_penalty': trial.user_attrs.get('repetition_penalty', 0.0),
                    'coherence_score': trial.user_attrs.get('coherence_score', 0.0),
                    'readability_score': trial.user_attrs.get('readability_score', 0.0),
                    'lazy_score': trial.user_attrs.get('lazy_score', 0.0),
                    'lazy_pattern_count': trial.user_attrs.get('lazy_pattern_count', 0),
                    'prose_penalty': trial.user_attrs.get('prose_penalty', 0.0),
                    'sentence_start_monotony': trial.user_attrs.get('sentence_start_monotony', 0.0),
                    'word_frequency_spike': trial.user_attrs.get('word_frequency_spike', 0.0),
                    'telling_verb_density': trial.user_attrs.get('telling_verb_density', 0.0),
                    'slop_guard_score': trial.user_attrs.get('slop_guard_score', 50.0),
                    'slop_guard_band': trial.user_attrs.get('slop_guard_band', 'unknown'),
                    'slop_guard_violations': trial.user_attrs.get('slop_guard_violations', 0),
                    'score_variance': trial.user_attrs.get('score_variance', 0.0),
                    'score_min': trial.user_attrs.get('score_min', 0.0),
                    'num_prompts_evaluated': trial.user_attrs.get('num_prompts_evaluated', 1),
                    'prompt_mode': trial.user_attrs.get('prompt_mode', 'single'),
                    'response_text': first_text,
                }
                data.append(row)

        df = pd.DataFrame(data)

        if len(df) > 0:
            # Mirror the formula from analyze_results.analyze_text_quality:
            # base = coherence*0.2 + readability*0.2 + (1-lazy)*0.3 + slop_guard*0.3
            # overall = base * (1 - repetition_penalty) * (1 - prose_penalty)
            base = (
                df['coherence_score'] * 0.20 +
                df['readability_score'] * 0.20 +
                (1 - df['lazy_score']) * 0.30 +
                (df['slop_guard_score'] / 100.0) * 0.30
            )
            df['overall_score'] = base * (1 - df['repetition_penalty']) * (1 - df['prose_penalty'])

        return df

    def generate_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics from results."""
        if len(df) == 0:
            return {'total_trials': 0, 'completed_trials': 0}

        best_idx = df['overall_score'].idxmax()
        param_cols = [p['name'] for p in self.parameter_space]
        best_params = {c: df.loc[best_idx, c] for c in param_cols if c in df.columns}

        return {
            'total_trials': len(df),
            'completed_trials': len(df),
            'best_quality_score': df['value'].max(),
            'avg_quality_score': df['value'].mean(),
            'best_overall_score': df['overall_score'].max(),
            'avg_overall_score': df['overall_score'].mean(),
            'avg_response_length': df['response_length'].mean(),
            'avg_repetition_penalty': df['repetition_penalty'].mean(),
            'avg_coherence_score': df['coherence_score'].mean(),
            'avg_readability_score': df['readability_score'].mean(),
            'avg_lazy_score': df['lazy_score'].mean(),
            'best_parameters': best_params,
            'prompt_mode': df['prompt_mode'].iloc[0] if len(df) > 0 else 'unknown',
            'avg_prompts_per_trial': df['num_prompts_evaluated'].mean(),
            'avg_score_variance': df['score_variance'].mean(),
            'avg_score_min': df['score_min'].mean(),
        }

    def print_summary(self, summary: Dict[str, Any]) -> None:
        """Print summary statistics."""
        print("=" * 60)
        print("OPTUNA PARAMETER SWEEP SUMMARY")
        print("=" * 60)
        print()

        print("GENERAL:")
        print(f"   Total completed trials: {summary['completed_trials']}")
        print()

        print("PROMPT POOL:")
        print(f"   Mode: {summary.get('prompt_mode', 'single')}")
        print(f"   Avg prompts per trial:  {summary.get('avg_prompts_per_trial', 1):.1f}")
        print(f"   Avg cross-prompt var:   {summary.get('avg_score_variance', 0):.4f}")
        print(f"   Avg worst-prompt score: {summary.get('avg_score_min', 0):.4f}")
        print()

        print("QUALITY METRICS:")
        print(f"   Best quality score:   {summary['best_quality_score']:.4f}")
        print(f"   Avg quality score:    {summary['avg_quality_score']:.4f}")
        print(f"   Best overall score:   {summary['best_overall_score']:.4f}")
        print(f"   Avg overall score:    {summary['avg_overall_score']:.4f}")
        print()

        print("TEXT METRICS:")
        print(f"   Avg response length:      {summary['avg_response_length']:.1f} chars")
        print(f"   Avg repetition penalty:   {summary['avg_repetition_penalty']:.4f}")
        print(f"   Avg coherence score:      {summary['avg_coherence_score']:.4f}")
        print(f"   Avg readability score:    {summary['avg_readability_score']:.4f}")
        print(f"   Avg lazy score:           {summary['avg_lazy_score']:.4f}")
        print()

        print("BEST PARAMETERS:")
        for param, value in summary['best_parameters'].items():
            if isinstance(value, float):
                print(f"   {param}: {value:.4f}")
            else:
                print(f"   {param}: {value}")
        print()
        print("=" * 60)

    def print_pattern_summary(self, study: optuna.Study) -> None:
        """Print aggregate slop pattern frequencies across all completed trials."""
        # Aggregate across ALL completed trials
        all_counts: Counter = Counter()
        all_trials_with: Counter = Counter()   # how many trials had this pattern
        n_completed = 0

        for trial in study.trials:
            if trial.state != optuna.trial.TrialState.COMPLETE:
                continue
            n_completed += 1
            patterns = trial.user_attrs.get('lazy_patterns_found', [])
            seen_this_trial: set = set()
            for pat, count in patterns:
                all_counts[pat] += count
                if pat not in seen_this_trial:
                    all_trials_with[pat] += 1
                    seen_this_trial.add(pat)

        if not all_counts:
            return

        print()
        print("=" * 60)
        print("SLOP PATTERN FREQUENCY (all completed trials)")
        print("=" * 60)
        print(f"  {'Pattern':<45} {'Hits':>5}  {'Trials':>6}  {'%Trials':>7}")
        print(f"  {'-'*45} {'-'*5}  {'-'*6}  {'-'*7}")
        for pat, total_hits in all_counts.most_common(30):
            n_trials = all_trials_with[pat]
            pct = 100.0 * n_trials / n_completed if n_completed else 0
            # Truncate long regex patterns for display
            display = pat[:45] if len(pat) <= 45 else pat[:42] + "..."
            print(f"  {display:<45} {total_hits:>5}  {n_trials:>6}  {pct:>6.1f}%")
        print("=" * 60)

    def generate_top_results_report(self, study: optuna.Study, df: pd.DataFrame,
                                     top_n: int = 10,
                                     filename: Optional[str] = None) -> str:
        """Generate a human-readable markdown report of the top N trials."""
        if filename is None:
            filename = f"sweep_top_results_{self.run_suffix}.md"

        if len(df) == 0:
            with open(filename, 'w') as f:
                f.write("# Sweep Results\n\nNo completed trials.\n")
            return filename

        # Sort by overall_score descending, take top N
        top = df.nlargest(top_n, 'overall_score').copy()

        param_cols = [p['name'] for p in self.parameter_space]
        score_cols = ['overall_score', 'coherence_score', 'readability_score',
                      'lazy_score', 'repetition_penalty', 'prose_penalty',
                      'sentence_start_monotony', 'word_frequency_spike',
                      'telling_verb_density']

        lines: list[str] = []
        lines.append(f"# Sweep Top {len(top)} Results")
        lines.append("")
        lines.append(f"**Model:** {self.model_name}  ")
        lines.append(f"**Total trials:** {len(df)}  ")
        pool = self.prompt_pool
        mode_desc = pool.mode
        if pool.mode == "sample":
            mode_desc += f" (K={pool.samples_per_trial})"
        lines.append(f"**Prompt pool:** {len(pool.prompts)} prompts, mode={mode_desc}  ")
        lines.append(f"**Score range:** {df['overall_score'].min():.3f} -- {df['overall_score'].max():.3f} "
                      f"(mean {df['overall_score'].mean():.3f})  ")
        lines.append(f"**Parameters swept:** {', '.join(param_cols)}")
        lines.append("")

        # Compact summary table
        lines.append("## Quick Comparison")
        lines.append("")
        header = f"| Rank | Trial | Score |"
        sep = "|---:|---:|---:|"
        for col in param_cols:
            header += f" {col} |"
            sep += "---:|"
        header += " lazy | prose_pen | rep_pen |"
        sep += "---:|---:|---:|"
        # Add variance columns when multi-prompt
        is_multi = any(row.get('num_prompts_evaluated', 1) > 1 for _, row in top.iterrows())
        if is_multi:
            header += " var | min |"
            sep += "---:|---:|"
        lines.append(header)
        lines.append(sep)

        for rank, (_, row) in enumerate(top.iterrows(), 1):
            line = f"| {rank} | #{int(row['trial'])} | {row['overall_score']:.3f} |"
            for col in param_cols:
                val = row.get(col, 0)
                line += f" {val:.2f} |" if isinstance(val, float) else f" {val} |"
            line += f" {row['lazy_score']:.2f} | {row['prose_penalty']:.2f} | {row['repetition_penalty']:.2f} |"
            if is_multi:
                line += f" {row.get('score_variance', 0):.3f} | {row.get('score_min', 0):.3f} |"
            lines.append(line)

        lines.append("")

        # Detailed per-trial sections
        lines.append("---")
        lines.append("")
        for rank, (_, row) in enumerate(top.iterrows(), 1):
            trial_num = int(row['trial'])
            lines.append(f"## #{rank}: Trial {trial_num} (score: {row['overall_score']:.3f})")
            lines.append("")

            # Parameters
            lines.append("### Sampler Parameters")
            lines.append("")
            lines.append("| Parameter | Value |")
            lines.append("|---|---|")
            for col in param_cols:
                val = row.get(col, 'N/A')
                if isinstance(val, float):
                    lines.append(f"| {col} | {val:.4f} |")
                else:
                    lines.append(f"| {col} | {val} |")
            lines.append("")

            # Scores
            lines.append("### Quality Metrics")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|---|---|")
            for col in score_cols:
                val = row.get(col, 0.0)
                lines.append(f"| {col} | {val:.3f} |")
            lines.append("")

            # Find the trial object for per-prompt data
            trial_obj = None
            for trial in study.trials:
                if trial.number == trial_num:
                    trial_obj = trial
                    break

            if trial_obj is None:
                continue

            # Per-prompt breakdown (if multi-prompt)
            per_prompt = trial_obj.user_attrs.get('per_prompt_metrics', {})
            if len(per_prompt) > 1:
                lines.append("### Per-Prompt Scores")
                lines.append("")
                lines.append("| Prompt | Score | Lazy | Prose Pen | Rep Pen | Slop |")
                lines.append("|---|---:|---:|---:|---:|---:|")
                for pid, pm in sorted(per_prompt.items(),
                                      key=lambda x: x[1].get('quality_score', 0),
                                      reverse=True):
                    lines.append(
                        f"| {pid} | {pm.get('quality_score', 0):.3f} "
                        f"| {pm.get('lazy_score', 0):.2f} | {pm.get('prose_penalty', 0):.2f} "
                        f"| {pm.get('repetition_penalty', 0):.2f} | {pm.get('slop_guard_score', 50):.0f} |"
                    )
                lines.append("")
                sv = trial_obj.user_attrs.get('score_variance', 0)
                sm = trial_obj.user_attrs.get('score_min', 0)
                lines.append(f"**Cross-prompt variance:** {sv:.4f}  "
                             f"**Worst single:** {sm:.3f}")
                lines.append("")

            # Slop patterns if any
            lazy_count = int(row.get('lazy_pattern_count', 0))
            if lazy_count > 0:
                pats = trial_obj.user_attrs.get('lazy_patterns_found', [])
                if pats:
                    lines.append(f"**Slop patterns found ({lazy_count} total hits):** "
                                 + ", ".join(f"{p} ({c}x)" for p, c in pats[:10]))
                    lines.append("")

            # Generated text(s)
            response_texts = self._get_response_texts(trial_obj)

            if len(response_texts) > 1:
                lines.append("### Generated Texts")
                lines.append("")
                for pid, text in response_texts.items():
                    score = per_prompt.get(pid, {}).get('quality_score', 0)
                    lines.append(f"**{pid}** (score: {score:.3f}):")
                    lines.append("")
                    lines.append("```")
                    lines.append(text.strip()[:2000])
                    lines.append("```")
                    lines.append("")
            elif response_texts:
                lines.append("### Generated Text")
                lines.append("")
                text = next(iter(response_texts.values()))
                lines.append("```")
                lines.append(text.strip())
                lines.append("```")
            else:
                lines.append("### Generated Text")
                lines.append("")
                lines.append("*(text not available)*")

            lines.append("")
            lines.append("---")
            lines.append("")

        report = "\n".join(lines)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"Top {len(top)} results report saved to: {filename}")
        return filename


def main():
    """Main function to run the parameter sweep."""
    parser = argparse.ArgumentParser(description="Optuna LLM parameter sweep")
    parser.add_argument("--config", "-c", type=str, default=None,
                        help="Path to YAML config file")
    args = parser.parse_args()

    # --- Defaults (overridden by env vars, then by config file) ----------
    base_url = os.environ.get("LLM_BASE_URL",
                              "https://poly-wife-replacement-removal.trycloudflare.com/v1")
    model_name = os.environ.get("LLM_MODEL", "")
    api_key = os.environ.get("LLM_API_KEY", "not-needed")
    max_tokens = int(os.environ.get("LLM_MAX_TOKENS", "512"))
    max_trials = int(os.environ.get("SWEEP_MAX_TRIALS", "30"))
    n_jobs = 1
    study_name = "llm_parameter_optimization"
    top_n = 10
    label = ""
    checkpoint_interval = 0
    convergence_patience = 25
    convergence_min_delta = 0.005
    parallel_prompts = False
    n_startup_trials = 12
    api_extra_body = {}
    assistant_prefill = ""
    params_dict = dict(DEFAULT_PARAMETERS)

    cfg = {}

    # --- Load config file if provided ------------------------------------
    if args.config:
        cfg = load_config(args.config)
        api_cfg = cfg.get('api', {})
        base_url = api_cfg.get('base_url', base_url)
        model_name = api_cfg.get('model', model_name)
        api_key = api_cfg.get('api_key', api_key)
        max_tokens = api_cfg.get('max_tokens', max_tokens)
        api_extra_body = api_cfg.get('extra_body', {})
        assistant_prefill = api_cfg.get('assistant_prefill', assistant_prefill)

        sweep_cfg = cfg.get('sweep', {})
        max_trials = sweep_cfg.get('max_trials', max_trials)
        n_jobs = sweep_cfg.get('n_jobs', n_jobs)
        study_name = sweep_cfg.get('study_name', study_name)
        top_n = sweep_cfg.get('top_n', top_n)
        label = sweep_cfg.get('label', label)
        checkpoint_interval = sweep_cfg.get('checkpoint_interval', checkpoint_interval)
        convergence_patience = sweep_cfg.get('convergence_patience', convergence_patience)
        convergence_min_delta = sweep_cfg.get('convergence_min_delta', convergence_min_delta)
        parallel_prompts = sweep_cfg.get('parallel_prompts', parallel_prompts)
        n_startup_trials = sweep_cfg.get('n_startup_trials', n_startup_trials)

        if 'parameters' in cfg:
            params_dict = cfg['parameters']

        print(f"Loaded config from: {args.config}")

    parameter_space = parse_parameter_space(params_dict)

    # --- Build prompt pool -----------------------------------------------
    prompt_pool = build_prompt_pool(cfg)

    print(f"Prompt pool: {len(prompt_pool.prompts)} prompts, mode={prompt_pool.mode}"
          + (f", K={prompt_pool.samples_per_trial}" if prompt_pool.mode == "sample" else ""))
    for pid, entry in prompt_pool.prompts.items():
        fmt = entry['format']
        cat = entry.get('category', '?')
        print(f"   [{fmt:10s}] {pid} ({cat})")
    print()

    # --- Auto-detect model name if empty ---------------------------------
    if not model_name:
        try:
            client = OpenAI(base_url=base_url, api_key=api_key)
            models = client.models.list()
            model_name = models.data[0].id
            print(f"Auto-detected model: {model_name}")
        except Exception as e:
            print(f"Could not auto-detect model: {e}")
            model_name = "default"

    print("OPTUNA PARAMETER SWEEP FOR LLM OPTIMIZATION")
    print("=" * 60)
    print(f"Parameters: {', '.join(p['name'] for p in parameter_space)}")
    print()

    sweep = OptunaParameterSweep(
        base_url=base_url,
        model_name=model_name,
        prompt_pool=prompt_pool,
        parameter_space=parameter_space,
        max_tokens=max_tokens,
        max_trials=max_trials,
        n_jobs=n_jobs,
        study_name=study_name,
        api_key=api_key,
        label=label,
        checkpoint_interval=checkpoint_interval,
        top_n=top_n,
        convergence_patience=convergence_patience,
        convergence_min_delta=convergence_min_delta,
        parallel_prompts=parallel_prompts,
        n_startup_trials=n_startup_trials,
        api_extra_body=api_extra_body,
        assistant_prefill=assistant_prefill,
    )

    study = sweep.run_study()

    results_filename = sweep.save_results(study)

    df = sweep.analyze_results(study)
    if len(df) > 0:
        summary = sweep.generate_summary(df)
        sweep.print_summary(summary)
        sweep.print_pattern_summary(study)

        analyzed_filename = f"optuna_parameter_sweep_analyzed_{sweep.run_suffix}.csv"
        df.to_csv(analyzed_filename, index=False)
        print(f"Analyzed results saved to: {analyzed_filename}")

        # Generate human-readable top results report
        sweep.generate_top_results_report(study, df, top_n=top_n)

        print()
        print("RECOMMENDED PARAMETERS:")
        for param, value in study.best_params.items():
            if isinstance(value, float):
                print(f"   {param}: {value:.4f}")
            else:
                print(f"   {param}: {value}")
    else:
        print("No completed trials to analyze.")

    print()
    print(f"Optimization complete!  Results: {results_filename}")


if __name__ == "__main__":
    main()
