#!/usr/bin/env python3
"""
Optuna-based parameter sweep for LLM sampling parameters

This script uses Optuna to optimize LLM sampling parameters by maximizing
the text quality score calculated from analyze_results.py.

Usage:
    uv run optuna_parameter_sweep.py

Dependencies:
    optuna
    openai
    pandas
    numpy
    (same as analyze_results.py)
"""

import optuna
import json
import os
import time
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from openai import OpenAI
from analyze_results import analyze_text_quality


class OptunaParameterSweep:
    """Optuna-based parameter sweep for LLM sampling optimization"""

    def __init__(self,
                 base_url: str,
                 model_name: str,
                 prompt: str,
                 max_tokens: int = 1024,
                 max_trials: int = 100,
                 timeout: Optional[int] = None,
                 n_jobs: int = 1,
                 study_name: str = "llm_parameter_optimization",
                 storage: Optional[str] = None,
                 api_key: str = "not-needed"):
        """
        Initialize the parameter sweep.

        Args:
            base_url: Base URL for the OpenAI-compatible API
            model_name: Name/ID of the model to use
            prompt: Input prompt for the LLM (completion-style)
            max_tokens: Maximum tokens to generate per trial
            max_trials: Maximum number of optimization trials
            timeout: Timeout in seconds for the study
            n_jobs: Number of parallel jobs (1 for sequential)
            study_name: Name for the Optuna study
            storage: Optional storage URL for distributed optimization
            api_key: API key (use "not-needed" for local servers)
        """
        self.base_url = base_url
        self.model_name = model_name
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.max_trials = max_trials
        self.timeout = timeout
        self.n_jobs = n_jobs
        self.study_name = study_name
        self.storage = storage

        self.client = OpenAI(base_url=base_url, api_key=api_key)

    # ------------------------------------------------------------------
    # Optuna objective
    # ------------------------------------------------------------------
    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for Optuna optimization."""
        params = {
            'temperature': trial.suggest_float('temperature', 0.1, 2.0),
            'min_p': trial.suggest_float('min_p', 0.0, 0.3),
            'top_p': trial.suggest_float('top_p', 0.5, 1.0),
            'top_k': trial.suggest_int('top_k', 0, 100),
            'rep_pen': trial.suggest_float('rep_pen', 1.0, 1.3),
            'typical': trial.suggest_float('typical', 0.5, 1.0),
        }

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

        if score < 0.1:
            raise optuna.exceptions.TrialPruned(f"Low quality score: {score}")

        return score

    # ------------------------------------------------------------------
    # LLM generation
    # ------------------------------------------------------------------
    def _generate_llm_response(self, params: Dict[str, Any]) -> str:
        """Generate a completion from the LLM using the given sampling params."""
        # KoboldCpp accepts extra sampler params via extra_body
        response = self.client.completions.create(
            model=self.model_name,
            prompt=self.prompt,
            max_tokens=self.max_tokens,
            temperature=params['temperature'],
            top_p=params['top_p'],
            extra_body={
                'min_p': params['min_p'],
                'top_k': params['top_k'],
                'rep_pen': params['rep_pen'],
                'typical': params['typical'],
            },
        )
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
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"optuna_parameter_sweep_results_{timestamp}.json"

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
                    'response_text': trial.user_attrs.get('response_text', ''),
                }
                data.append(row)

        df = pd.DataFrame(data)

        if len(df) > 0:
            df['overall_score'] = (
                df['value'] * 0.4 +
                (1 - df['repetition_penalty']) * 0.3 +
                df['coherence_score'] * 0.2 +
                df['readability_score'] * 0.1
            )

        return df

    def generate_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics from results."""
        if len(df) == 0:
            return {'total_trials': 0, 'completed_trials': 0}

        best_idx = df['overall_score'].idxmax()
        param_cols = ['temperature', 'min_p', 'top_p', 'top_k', 'rep_pen', 'typical']
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


def main():
    """Main function to run the parameter sweep"""

    # Configuration
    BASE_URL = os.environ.get(
        "LLM_BASE_URL",
        "https://poly-wife-replacement-removal.trycloudflare.com/v1",
    )
    MODEL_NAME = os.environ.get("LLM_MODEL", "")  # empty = use whatever the server has
    MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "512"))
    MAX_TRIALS = int(os.environ.get("SWEEP_MAX_TRIALS", "30"))

    PROMPT = """Continue the following story in vivid, creative prose:

Once upon a time in a small village nestled between rolling hills, there lived a young girl named Elara.
She was known throughout the village for her curiosity and her unusual ability to understand the language of animals.
One day, while exploring the forest edge, she heard a faint whimpering sound coming from a dense thicket.
"""

    # Auto-detect model name if not set
    if not MODEL_NAME:
        try:
            client = OpenAI(base_url=BASE_URL, api_key="not-needed")
            models = client.models.list()
            MODEL_NAME = models.data[0].id
            print(f"Auto-detected model: {MODEL_NAME}")
        except Exception as e:
            print(f"Could not auto-detect model: {e}")
            MODEL_NAME = "default"

    print("OPTUNA PARAMETER SWEEP FOR LLM OPTIMIZATION")
    print("=" * 60)
    print()

    sweep = OptunaParameterSweep(
        base_url=BASE_URL,
        model_name=MODEL_NAME,
        prompt=PROMPT,
        max_tokens=MAX_TOKENS,
        max_trials=MAX_TRIALS,
        n_jobs=1,
    )

    study = sweep.run_study()

    results_filename = sweep.save_results(study)

    df = sweep.analyze_results(study)
    if len(df) > 0:
        summary = sweep.generate_summary(df)
        sweep.print_summary(summary)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        analyzed_filename = f"optuna_parameter_sweep_analyzed_{timestamp}.csv"
        df.to_csv(analyzed_filename, index=False)
        print(f"Analyzed results saved to: {analyzed_filename}")

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
