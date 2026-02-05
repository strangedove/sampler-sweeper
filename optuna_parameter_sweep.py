#!/usr/bin/env python3
"""
Optuna-based parameter sweep for LLM sampling parameters

This script uses Optuna to optimize LLM sampling parameters by maximizing
the text quality score calculated from analyze_results.py.

Usage:
    uv run optuna_parameter_sweep.py --config sweep_config.yaml
    uv run optuna_parameter_sweep.py   # uses defaults / env vars

Dependencies:
    optuna, openai, pandas, numpy, pyyaml
    (same as analyze_results.py)
"""

import argparse
from collections import Counter
import optuna
import json
import os
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


class OptunaParameterSweep:
    """Optuna-based parameter sweep for LLM sampling optimization"""

    def __init__(self,
                 base_url: str,
                 model_name: str,
                 prompt: str,
                 parameter_space: Optional[List[Dict[str, Any]]] = None,
                 max_tokens: int = 1024,
                 max_trials: int = 100,
                 timeout: Optional[int] = None,
                 n_jobs: int = 1,
                 study_name: str = "llm_parameter_optimization",
                 storage: Optional[str] = None,
                 api_key: str = "not-needed",
                 label: str = ""):
        self.base_url = base_url
        self.model_name = model_name
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.max_trials = max_trials
        self.timeout = timeout
        self.n_jobs = n_jobs
        self.study_name = study_name
        self.storage = storage
        self.label = label
        self.parameter_space = parameter_space or parse_parameter_space(DEFAULT_PARAMETERS)

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
        """Objective function for Optuna optimization."""
        params = {}
        for p in self.parameter_space:
            if p['type'] == 'int':
                params[p['name']] = trial.suggest_int(p['name'], int(p['low']), int(p['high']))
            else:
                params[p['name']] = trial.suggest_float(p['name'], float(p['low']), float(p['high']))

        try:
            response_text = self._generate_llm_response(params)
        except Exception as e:
            print(f"  [trial {trial.number}] API error: {e}")
            raise optuna.exceptions.TrialPruned(f"API error: {e}")

        if not response_text or len(response_text.strip()) < 20:
            raise optuna.exceptions.TrialPruned("Empty or very short response")

        quality_metrics = analyze_text_quality(response_text)
        score = quality_metrics['quality_score']

        # Store metrics for analysis
        trial.set_user_attr('response_text', response_text)
        trial.set_user_attr('response_length', len(response_text))
        trial.set_user_attr('repetition_penalty', quality_metrics['repetition_penalty'])
        trial.set_user_attr('coherence_score', quality_metrics['coherence_score'])
        trial.set_user_attr('readability_score', quality_metrics['readability_score'])
        trial.set_user_attr('lazy_score', quality_metrics.get('lazy_score', 0.0))
        trial.set_user_attr('lazy_pattern_count', quality_metrics.get('lazy_pattern_count', 0))
        trial.set_user_attr('prose_penalty', quality_metrics.get('prose_penalty', 0.0))
        trial.set_user_attr('sentence_start_monotony', quality_metrics.get('sentence_start_monotony', 0.0))
        trial.set_user_attr('word_frequency_spike', quality_metrics.get('word_frequency_spike', 0.0))
        trial.set_user_attr('telling_verb_density', quality_metrics.get('telling_verb_density', 0.0))
        # Store the specific patterns that matched (list of [pattern, count])
        patterns_found = quality_metrics.get('lazy_patterns_found', [])
        trial.set_user_attr('lazy_patterns_found',
                            [[p, c] for p, c in patterns_found])

        if score < 0.1:
            raise optuna.exceptions.TrialPruned(f"Low quality score: {score}")

        return score

    # ------------------------------------------------------------------
    # LLM generation
    # ------------------------------------------------------------------
    def _generate_llm_response(self, params: Dict[str, Any]) -> str:
        """Generate a completion from the LLM using the given sampling params."""
        native = {k: v for k, v in params.items() if k in NATIVE_OPENAI_PARAMS}
        extra = {k: v for k, v in params.items() if k not in NATIVE_OPENAI_PARAMS}

        kwargs: Dict[str, Any] = dict(
            model=self.model_name,
            prompt=self.prompt,
            max_tokens=self.max_tokens,
            **native,
        )
        if extra:
            kwargs['extra_body'] = extra

        response = self.client.completions.create(**kwargs)
        return response.choices[0].text

    # ------------------------------------------------------------------
    # Study runner
    # ------------------------------------------------------------------
    def run_study(self) -> optuna.Study:
        """Run the Optuna optimization study."""
        print(f"Starting Optuna parameter sweep")
        print(f"   API: {self.base_url}")
        print(f"   Model: {self.model_name}")
        print(f"   Max trials: {self.max_trials}")
        print(f"   Parallel jobs: {self.n_jobs}")
        print(f"   Study name: {self.study_name}")
        print()

        sampler = optuna.samplers.TPESampler(n_startup_trials=20)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)

        study = optuna.create_study(
            study_name=self.study_name,
            sampler=sampler,
            pruner=pruner,
            direction='maximize',
            storage=self.storage,
            load_if_exists=True,
        )

        study.optimize(
            self.objective,
            n_trials=self.max_trials,
            timeout=self.timeout,
            n_jobs=self.n_jobs,
            callbacks=[self._progress_callback],
        )

        return study

    def _progress_callback(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """Print progress every 5 trials."""
        if trial.number % 5 == 0 or trial.number == 0:
            pruned = len(study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED]))
            print(f"[{trial.number + 1}/{self.max_trials}]  "
                  f"best={study.best_value:.4f}  pruned={pruned}  "
                  f"params={study.best_params}")

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    def save_results(self, study: optuna.Study, filename: Optional[str] = None) -> str:
        """Save optimization results to JSON file."""
        if filename is None:
            filename = f"optuna_parameter_sweep_results_{self.run_suffix}.json"

        results = []
        for trial in study.trials:
            result = {
                'trial_number': trial.number,
                'state': trial.state.name,
                'value': trial.value,
                'parameters': trial.params,
                'user_attrs': {k: v for k, v in trial.user_attrs.items()
                               if k != 'response_text'},
                'response_text': trial.user_attrs.get('response_text', ''),
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
                    'response_text': trial.user_attrs.get('response_text', ''),
                }
                data.append(row)

        df = pd.DataFrame(data)

        if len(df) > 0:
            # Mirror the formula from analyze_results.analyze_text_quality:
            # base = coherence*0.3 + readability*0.3 + (1-lazy)*0.4
            # overall = base * (1 - repetition_penalty) * (1 - prose_penalty)
            base = (
                df['coherence_score'] * 0.30 +
                df['readability_score'] * 0.30 +
                (1 - df['lazy_score']) * 0.40
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
        """Generate a human-readable markdown report of the top N trials.

        The report includes a sweep summary header, then each trial's sampler
        parameters, quality metrics, and the full generated text -- designed to
        be skimmed quickly by a human reviewer or fed to a judge model.
        """
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
        lines.append(header)
        lines.append(sep)

        for rank, (_, row) in enumerate(top.iterrows(), 1):
            line = f"| {rank} | #{int(row['trial'])} | {row['overall_score']:.3f} |"
            for col in param_cols:
                val = row.get(col, 0)
                line += f" {val:.2f} |" if isinstance(val, float) else f" {val} |"
            line += f" {row['lazy_score']:.2f} | {row['prose_penalty']:.2f} | {row['repetition_penalty']:.2f} |"
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

            # Slop patterns if any
            lazy_count = int(row.get('lazy_pattern_count', 0))
            if lazy_count > 0:
                # Look up the patterns from the study's trial attrs
                for trial in study.trials:
                    if trial.number == trial_num:
                        pats = trial.user_attrs.get('lazy_patterns_found', [])
                        if pats:
                            lines.append(f"**Slop patterns found ({lazy_count} total hits):** "
                                         + ", ".join(f"{p} ({c}x)" for p, c in pats[:10]))
                            lines.append("")
                        break

            # Generated text
            lines.append("### Generated Text")
            lines.append("")
            text = row.get('response_text', '')
            if text:
                lines.append("```")
                lines.append(text.strip())
                lines.append("```")
            else:
                lines.append("*(text not available)*")
            lines.append("")
            lines.append("---")
            lines.append("")

        report = "\n".join(lines)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"Top {len(top)} results report saved to: {filename}")
        return filename


DEFAULT_PROMPT = """Continue the following story in vivid, creative prose:

The morning Dagna found the dead radio tower, she was already three days late getting back to town.
She sat on a lichen-covered boulder and ate the last of her dried fish while studying the structure.
The cables had been cut -- not frayed, not corroded, but sliced clean through with something sharp.
"""


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
    prompt = DEFAULT_PROMPT
    params_dict = dict(DEFAULT_PARAMETERS)

    # --- Load config file if provided ------------------------------------
    if args.config:
        cfg = load_config(args.config)
        api_cfg = cfg.get('api', {})
        base_url = api_cfg.get('base_url', base_url)
        model_name = api_cfg.get('model', model_name)
        api_key = api_cfg.get('api_key', api_key)
        max_tokens = api_cfg.get('max_tokens', max_tokens)

        sweep_cfg = cfg.get('sweep', {})
        max_trials = sweep_cfg.get('max_trials', max_trials)
        n_jobs = sweep_cfg.get('n_jobs', n_jobs)
        study_name = sweep_cfg.get('study_name', study_name)
        top_n = sweep_cfg.get('top_n', top_n)
        label = sweep_cfg.get('label', label)

        if 'prompt' in cfg:
            prompt = cfg['prompt']

        if 'parameters' in cfg:
            params_dict = cfg['parameters']

        print(f"Loaded config from: {args.config}")

    parameter_space = parse_parameter_space(params_dict)

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
        prompt=prompt,
        parameter_space=parameter_space,
        max_tokens=max_tokens,
        max_trials=max_trials,
        n_jobs=n_jobs,
        study_name=study_name,
        api_key=api_key,
        label=label,
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
