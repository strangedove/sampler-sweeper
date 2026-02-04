#!/usr/bin/env python3
"""
Optuna-based parameter sweep for LLM sampling parameters

This script uses Optuna to optimize LLM sampling parameters by maximizing
the text quality score calculated from analyze_results.py.

Usage:
    uv run optuna_parameter_sweep.py

Dependencies:
    optuna
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
from analyze_results import analyze_text_quality
import re


class OptunaParameterSweep:
    """Optuna-based parameter sweep for LLM sampling optimization"""
    
    def __init__(self, 
                 model_name: str,
                 prompt: str,
                 max_trials: int = 100,
                 timeout: Optional[int] = None,
                 n_jobs: int = 1,
                 study_name: str = "llm_parameter_optimization",
                 storage: Optional[str] = None):
        """
        Initialize the parameter sweep.
        
        Args:
            model_name: Name of the LLM model to use
            prompt: Input prompt for the LLM
            max_trials: Maximum number of optimization trials
            timeout: Timeout in seconds for the study
            n_jobs: Number of parallel jobs (1 for sequential)
            study_name: Name for the Optuna study
            storage: Optional storage URL for distributed optimization
        """
        self.model_name = model_name
        self.prompt = prompt
        self.max_trials = max_trials
        self.timeout = timeout
        self.n_jobs = n_jobs
        self.study_name = study_name
        self.storage = storage
        
        # Initialize LLM client (placeholder - replace with actual client)
        self.client = self._initialize_llm_client()
        
    def _initialize_llm_client(self):
        """
        Initialize the LLM client.
        
        Returns:
            LLM client object (placeholder - replace with actual implementation)
        """
        # This is a placeholder - replace with actual LLM client initialization
        # For example:
        # from some_llm_library import Client
        # return Client(api_key="your-api-key")
        
        class MockLLMClient:
            """Mock LLM client for demonstration"""
            def completions_create(self, **kwargs):
                """Mock completion method"""
                # In a real implementation, this would call the actual LLM API
                # For now, we'll return a mock response
                return MockCompletionResponse()
        
        return MockLLMClient()
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna optimization.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Quality score (higher is better)
        """
        # Define hyper-parameters to optimize
        params = {
            'temperature': trial.suggest_float('temperature', 0.1, 1.5),
            'min_p': trial.suggest_float('min_p', 0.0, 0.9),
            'adaptive_target': trial.suggest_float('adaptive_target', 0.5, 1.5),
            'adaptive_decay': trial.suggest_float('adaptive_decay', 0.8, 0.99),
            'top_nsigma': trial.suggest_float('top_nsigma', 1.0, 3.0),
        }
        
        # Generate response from LLM
        response_text = self._generate_llm_response(params)
        
        # Calculate quality score
        quality_metrics = analyze_text_quality(response_text)
        score = quality_metrics['quality_score']
        
        # Store additional metrics for analysis
        trial.set_user_attr('response_text', response_text)
        trial.set_user_attr('response_length', len(response_text))
        trial.set_user_attr('repetition_penalty', quality_metrics['repetition_penalty'])
        trial.set_user_attr('coherence_score', quality_metrics['coherence_score'])
        trial.set_user_attr('readability_score', quality_metrics['readability_score'])
        
        # Prune trials with very low quality
        if score < 0.1:
            raise optuna.exceptions.TrialPruned(f"Low quality score: {score}")
        
        return score
    
    def _generate_llm_response(self, params: Dict[str, float]) -> str:
        """
        Generate a response from the LLM using given parameters.
        
        Args:
            params: Dictionary of sampling parameters
            
        Returns:
            Generated text response
        """
        # This is a placeholder - replace with actual LLM API call
        # Example for a real implementation:
        # stream = self.client.completions.create(
        #     model=self.model_name,
        #     prompt=self.prompt,
        #     temperature=params['temperature'],
        #     min_p=params['min_p'],
        #     max_tokens=3000,
        #     stop="</response>",
        #     stream=True,
        #     extra_query={
        #         "adaptive_target": params['adaptive_target'],
        #         "adaptive_decay": params['adaptive_decay'],
        #         "top_nsigma": params['top_nsigma'],
        #     },
        # )
        # 
        # completion = ""
        # for chunk in stream:
        #     completion += chunk.choices[0].text
        # 
        # return completion
        
        # Mock response for demonstration
        return self._generate_mock_response(params)
    
    def _generate_mock_response(self, params: Dict[str, float]) -> str:
        """
        Generate a mock response for demonstration purposes.
        
        Args:
            params: Sampling parameters
            
        Returns:
            Mock response text
        """
        # Generate mock text based on parameters
        temperature = params['temperature']
        min_p = params['min_p']
        
        # Simulate different response styles based on parameters
        if temperature < 0.5:
            # Low temperature - deterministic, focused
            response = f"""The story continues with a clear, focused narrative. "
            f"The characters make logical decisions based on their motivations. "
            f"The plot progresses in a straightforward manner with minimal digressions. "
            f"Word choice is precise and vocabulary is varied. "
            f"Sentence structure is consistent and coherent."""
        elif temperature < 1.0:
            # Medium temperature - balanced
            response = f"""The narrative unfolds with a good balance of creativity and structure. "
            f"Characters exhibit both logical and somewhat unpredictable behavior. "
            f"The plot develops with interesting twists while maintaining coherence. "
            f"Language use shows moderate diversity with occasional repetition. "
            f"Sentences vary in length, creating a natural rhythm."""
        else:
            # High temperature - creative, diverse
            response = f"""The story takes unexpected turns with creative flair. "
            f"Characters behave in surprising ways, driven by emotional impulses. "
            f"The plot explores multiple themes and subplots simultaneously. "
            f"Vocabulary is diverse but may include some repetitive phrases. "
            f"Sentence structure varies widely, from short fragments to complex constructions."""
        
        # Add some variation based on min_p
        if min_p < 0.3:
            response += f" "
            f"The text demonstrates excellent control over repetition, with minimal "
            f"repetitive patterns or overused phrases."
        elif min_p < 0.6:
            response += f" "
            f"There is some repetition in the text, but it doesn't significantly "
            f"detract from the overall quality."
        else:
            response += f" "
            f"The text contains noticeable repetition, which may affect readability."
        
        return response
    
    def run_study(self) -> optuna.Study:
        """
        Run the Optuna optimization study.
        
        Returns:
            Completed Optuna study object
        """
        print(f"🚀 Starting Optuna parameter sweep")
        print(f"   Model: {self.model_name}")
        print(f"   Max trials: {self.max_trials}")
        print(f"   Parallel jobs: {self.n_jobs}")
        print(f"   Study name: {self.study_name}")
        print()
        
        # Create sampler with pruning
        sampler = optuna.samplers.TPESampler(n_startup_trials=20)
        
        # Create pruner
        pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
        
        # Create study
        study = optuna.create_study(
            study_name=self.study_name,
            sampler=sampler,
            pruner=pruner,
            direction='maximize',
            storage=self.storage,
            load_if_exists=True
        )
        
        # Run optimization
        study.optimize(
            self.objective,
            n_trials=self.max_trials,
            timeout=self.timeout,
            n_jobs=self.n_jobs,
            callbacks=[self._progress_callback]
        )
        
        return study
    
    def _progress_callback(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """
        Callback function to display progress during optimization.
        
        Args:
            study: Optuna study object
            trial: Current trial object
        """
        if trial.number % 10 == 0 or trial.number == 0:
            print(f"📊 Trial {trial.number + 1}/{self.max_trials}")
            print(f"   Best value: {study.best_value:.4f}")
            print(f"   Best params: {study.best_params}")
            print(f"   Pruned trials: {len(study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED]))}")
            print()
    
    def save_results(self, study: optuna.Study, filename: Optional[str] = None) -> str:
        """
        Save optimization results to JSON file.
        
        Args:
            study: Completed Optuna study
            filename: Optional filename (auto-generated if None)
            
        Returns:
            Path to saved results file
        """
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"optuna_parameter_sweep_results_{timestamp}.json"
        
        # Convert study results to list of dictionaries
        results = []
        for trial in study.trials:
            result = {
                'trial_number': trial.number,
                'state': trial.state.name,
                'value': trial.value,
                'parameters': trial.params,
                'user_attrs': trial.user_attrs,
                'datetime_start': trial.datetime_start.isoformat() if trial.datetime_start else None,
                'datetime_complete': trial.datetime_complete.isoformat() if trial.datetime_complete else None,
            }
            results.append(result)
        
        # Save to JSON
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Results saved to: {filename}")
        return filename
    
    def analyze_results(self, study: optuna.Study) -> pd.DataFrame:
        """
        Analyze optimization results and generate summary statistics.
        
        Args:
            study: Completed Optuna study
            
        Returns:
            DataFrame with analyzed results
        """
        # Convert to DataFrame
        data = []
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                row = {
                    'trial': trial.number,
                    'value': trial.value,
                    'temperature': trial.params.get('temperature'),
                    'min_p': trial.params.get('min_p'),
                    'adaptive_target': trial.params.get('adaptive_target'),
                    'adaptive_decay': trial.params.get('adaptive_decay'),
                    'top_nsigma': trial.params.get('top_nsigma'),
                    'response_length': trial.user_attrs.get('response_length', 0),
                    'repetition_penalty': trial.user_attrs.get('repetition_penalty', 0.0),
                    'coherence_score': trial.user_attrs.get('coherence_score', 0.0),
                    'readability_score': trial.user_attrs.get('readability_score', 0.0),
                    'response_text': trial.user_attrs.get('response_text', ''),
                }
                data.append(row)
        
        df = pd.DataFrame(data)
        
        # Calculate additional metrics
        df['overall_score'] = (
            df['value'] * 0.4 +
            (1 - df['repetition_penalty']) * 0.3 +
            df['coherence_score'] * 0.2 +
            df['readability_score'] * 0.1
        )
        
        return df
    
    def generate_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate summary statistics from results.
        
        Args:
            df: DataFrame with analyzed results
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
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
            'best_parameters': df.loc[df['overall_score'].idxmax(), 'parameters'].to_dict() if 'parameters' in df.columns else {},
        }
        
        return summary
    
    def print_summary(self, summary: Dict[str, Any]) -> None:
        """
        Print summary statistics in a readable format.
        
        Args:
            summary: Summary statistics dictionary
        """
        print("=" * 60)
        print("OPTUNA PARAMETER SWEEP SUMMARY")
        print("=" * 60)
        print()
        
        print("📊 GENERAL STATISTICS:")
        print(f"   Total trials: {summary['total_trials']}")
        print(f"   Completed trials: {summary['completed_trials']}")
        print()
        
        print("🎯 QUALITY METRICS:")
        print(f"   Best quality score: {summary['best_quality_score']:.4f}")
        print(f"   Average quality score: {summary['avg_quality_score']:.4f}")
        print(f"   Best overall score: {summary['best_overall_score']:.4f}")
        print(f"   Average overall score: {summary['avg_overall_score']:.4f}")
        print()
        
        print("📝 TEXT METRICS:")
        print(f"   Average response length: {summary['avg_response_length']:.1f} characters")
        print(f"   Average repetition penalty: {summary['avg_repetition_penalty']:.4f}")
        print(f"   Average coherence score: {summary['avg_coherence_score']:.4f}")
        print(f"   Average readability score: {summary['avg_readability_score']:.4f}")
        print()
        
        print("✅ BEST PARAMETERS:")
        best_params = summary['best_parameters']
        for param, value in best_params.items():
            print(f"   {param}: {value:.4f}")
        print()
        
        print("=" * 60)


class MockCompletionResponse:
    """Mock completion response for demonstration"""
    def __init__(self):
        self.choices = [MockChoice()]


class MockChoice:
    """Mock choice object for demonstration"""
    def __init__(self):
        self.text = "Mock response text"


def main():
    """Main function to run the parameter sweep"""
    
    # Configuration
    MODEL_NAME = "gpt-4"  # Replace with your actual model name
    PROMPT = """
    Continue the following story:
    
    Once upon a time in a small village nestled between rolling hills, there lived a young girl named Elara.
    She was known throughout the village for her curiosity and her unusual ability to understand the language of animals.
    One day, while exploring the forest edge, she heard a faint whimpering sound coming from a dense thicket.
    
    <response>
    """
    
    MAX_TRIALS = 50  # Set to a higher number for actual optimization
    
    print("🔧 OPTUNA PARAMETER SWEEP FOR LLM OPTIMIZATION")
    print("=" * 60)
    print()
    
    # Initialize and run the sweep
    sweep = OptunaParameterSweep(
        model_name=MODEL_NAME,
        prompt=PROMPT,
        max_trials=MAX_TRIALS,
        n_jobs=1  # Set to higher value for parallel execution
    )
    
    # Run the optimization study
    study = sweep.run_study()
    
    # Save results
    results_filename = sweep.save_results(study)
    
    # Analyze results
    df = sweep.analyze_results(study)
    summary = sweep.generate_summary(df)
    
    # Print summary
    sweep.print_summary(summary)
    
    # Save analyzed results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    analyzed_filename = f"optuna_parameter_sweep_analyzed_{timestamp}.csv"
    df.to_csv(analyzed_filename, index=False)
    print(f"📊 Analyzed results saved to: {analyzed_filename}")
    
    # Print best parameters
    print()
    print("🎯 RECOMMENDED PARAMETERS:")
    best_params = study.best_params
    for param, value in best_params.items():
        print(f"   {param}: {value:.4f}")
    
    print()
    print("✅ Optimization complete!")
    print(f"   Results saved to: {results_filename}")
    print(f"   Analyzed data saved to: {analyzed_filename}")


if __name__ == "__main__":
    main()
