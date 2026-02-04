#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis script for parameter sweep results

Usage with UV:
    uv run analyze_results.py
    
Dependencies (automatically handled by UV):
    pandas
    matplotlib
    seaborn
    numpy
"""

import json
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import numpy as np
import re
from typing import List, Dict, Tuple
import string

# For advanced NLP metrics (if available)
try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk import ngrams
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False
    print("NLTK not available - some advanced metrics will be disabled")


def load_results(filename: str) -> dict:
    """Load results from JSON file"""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)


def preprocess_text(text: str) -> str:
    """Basic text preprocessing for analysis"""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def calculate_repetition_metrics(text: str) -> Dict[str, float]:
    """Calculate various repetition metrics for text quality analysis"""
    if not text or not isinstance(text, str):
        return {
            'token_repetition_rate': 0.0,
            'bigram_repetition_rate': 0.0,
            'trigram_repetition_rate': 0.0,
            'consecutive_repetition_score': 0.0,
            'repetition_penalty': 0.0
        }
    
    # Tokenize text
    if HAS_NLTK:
        tokens = word_tokenize(text)
    else:
        tokens = text.split()
    
    if len(tokens) <= 1:
        return {
            'token_repetition_rate': 0.0,
            'bigram_repetition_rate': 0.0,
            'trigram_repetition_rate': 0.0,
            'consecutive_repetition_score': 0.0,
            'repetition_penalty': 0.0
        }
    
    # Token repetition rate
    token_counts = Counter(tokens)
    total_tokens = len(tokens)
    repeated_tokens = sum(1 for count in token_counts.values() if count > 1)
    token_repetition_rate = repeated_tokens / total_tokens
    
    # N-gram repetition
    def ngram_repetition(n: int) -> float:
        if len(tokens) < n:
            return 0.0
        
        n_grams = list(ngrams(tokens, n))
        ngram_counts = Counter(n_grams)
        total_ngrams = len(n_grams)
        
        if total_ngrams == 0:
            return 0.0
            
        repeated_ngrams = sum(1 for count in ngram_counts.values() if count > 1)
        return repeated_ngrams / total_ngrams
    
    bigram_repetition_rate = ngram_repetition(2)
    trigram_repetition_rate = ngram_repetition(3)
    
    # Consecutive repetition score
    consecutive_repeats = 0
    for i in range(len(tokens) - 1):
        if tokens[i] == tokens[i + 1]:
            consecutive_repeats += 1
    
    consecutive_repetition_score = consecutive_repeats / max(1, len(tokens) - 1)
    
    # Overall repetition penalty (composite score)
    repetition_penalty = (
        token_repetition_rate * 0.4 +
        bigram_repetition_rate * 0.3 +
        trigram_repetition_rate * 0.2 +
        consecutive_repetition_score * 0.1
    )
    
    return {
        'token_repetition_rate': token_repetition_rate,
        'bigram_repetition_rate': bigram_repetition_rate,
        'trigram_repetition_rate': trigram_repetition_rate,
        'consecutive_repetition_score': consecutive_repetition_score,
        'repetition_penalty': repetition_penalty
    }


def calculate_coherence_metrics(text: str) -> Dict[str, float]:
    """Calculate text coherence metrics"""
    if not text or not isinstance(text, str):
        return {
            'sentence_count': 0,
            'avg_sentence_length': 0.0,
            'sentence_length_variation': 0.0,
            'coherence_score': 0.0
        }
    
    # Sentence analysis
    if HAS_NLTK:
        sentences = sent_tokenize(text)
    else:
        # Simple sentence splitting if NLTK not available
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) == 0:
        return {
            'sentence_count': 0,
            'avg_sentence_length': 0.0,
            'sentence_length_variation': 0.0,
            'coherence_score': 0.0
        }
    
    # Sentence length metrics
    sentence_lengths = [len(sentence.split()) for sentence in sentences]
    avg_sentence_length = np.mean(sentence_lengths)
    sentence_length_variation = np.std(sentence_lengths) / avg_sentence_length if avg_sentence_length > 0 else 0.0
    
    # Simple coherence score (lower variation = more coherent)
    coherence_score = 1.0 - min(sentence_length_variation, 1.0)
    
    return {
        'sentence_count': len(sentences),
        'avg_sentence_length': avg_sentence_length,
        'sentence_length_variation': sentence_length_variation,
        'coherence_score': coherence_score
    }


def calculate_readability_metrics(text: str) -> Dict[str, float]:
    """Calculate basic readability metrics"""
    if not text or not isinstance(text, str):
        return {
            'word_count': 0,
            'avg_word_length': 0.0,
            'lexical_diversity': 0.0,
            'readability_score': 0.0
        }
    
    # Tokenize
    if HAS_NLTK:
        tokens = word_tokenize(text)
        words = [word.lower() for word in tokens if word.isalpha()]
    else:
        words = re.findall(r'\b\w+\b', text.lower())
    
    if len(words) == 0:
        return {
            'word_count': 0,
            'avg_word_length': 0.0,
            'lexical_diversity': 0.0,
            'readability_score': 0.0
        }
    
    # Basic metrics
    word_count = len(words)
    char_count = sum(len(word) for word in words)
    avg_word_length = char_count / word_count
    
    # Lexical diversity (type-token ratio)
    unique_words = len(set(words))
    lexical_diversity = unique_words / word_count
    
    # Simple readability score (higher = more readable)
    readability_score = lexical_diversity * (1.0 - min(avg_word_length / 10.0, 1.0))
    
    return {
        'word_count': word_count,
        'avg_word_length': avg_word_length,
        'lexical_diversity': lexical_diversity,
        'readability_score': readability_score
    }


def detect_lazy_uncreative_text(text: str, patterns: List[str] = None) -> Dict[str, Any]:
    """
    Detect lazy/uncreative text patterns in responses.
    
    Args:
        text: The text to analyze
        patterns: List of literal strings or regex patterns to match
                 If None, uses a default set of common lazy/uncreative patterns
    
    Returns:
        Dictionary with detection results including:
        - lazy_pattern_count: Number of matches found
        - lazy_patterns_found: List of matched patterns
        - lazy_score: Score from 0-1 (higher = more lazy/uncreative)
        - has_lazy_text: Boolean indicating if any patterns were found
    """
    if not text or not isinstance(text, str):
        return {
            'lazy_pattern_count': 0,
            'lazy_patterns_found': [],
            'lazy_score': 0.0,
            'has_lazy_text': False
        }
    
    # Default patterns for lazy/uncreative text
    if patterns is None:
        patterns = [
            # Common uncreative responses
            r'i don\'t know',
            r'i cannot',
            r'i can\'t',
            r'i am unable',
            r'i\'m sorry',
            r'sorry, i',
            r'not sure',
            r'unknown',
            r'no information',
            r'cannot provide',
            r'not available',
            r'out of scope',
            r'beyond my knowledge',
            r'beyond my capabilities',
            
            # Filler phrases
            r'the quick brown fox',
            r'lorem ipsum',
            r'to be or not to be',
            r'once upon a time',
            r'however, it is important to note',
            r'in conclusion',
            
            # Repetitive structures (3+ repetitions of same phrase)
            r'(\b\w+\s+){3,}(\1\s+){3,}',
            
            # Generic filler
            r'\bthe fact that\b',
            r'\bthe reality is\b',
            r'\bit is clear that\b',
            r'\bit should be noted\b',
            r'\bwithout a doubt\b',
            
            # Overused metaphors/clichés
            r'\bthink outside the box\b',
            r'\bat the end of the day\b',
            r'\bparadigm shift\b',
            r'\blow-hanging fruit\b',
            r'\braise the bar\b',
            
            # Placeholder text
            r'\bplaceholder\b',
            r'\bTODO\b',
            r'\bFIXME\b',
            r'\bXXX\b',
        ]
    
    text_lower = text.lower()
    matches = []
    
    for pattern in patterns:
        try:
            # Compile regex pattern
            if not pattern.startswith('^') and not pattern.endswith('$'):
                # Make it case-insensitive and allow partial matches
                regex = re.compile(pattern, re.IGNORECASE)
            else:
                regex = re.compile(pattern)
            
            if regex.search(text_lower):
                matches.append(pattern)
        except re.error:
            # If pattern is not valid regex, treat as literal string
            if pattern.lower() in text_lower:
                matches.append(pattern)
    
    # Calculate lazy score (0-1, higher = more lazy/uncreative)
    # Normalize by text length to account for longer responses
    text_length = len(text)
    if text_length == 0:
        lazy_score = 0.0
    else:
        # Score is proportional to number of matches, inversely proportional to text length
        lazy_score = min(1.0, len(matches) / max(1, text_length / 100))
    
    return {
        'lazy_pattern_count': len(matches),
        'lazy_patterns_found': matches,
        'lazy_score': lazy_score,
        'has_lazy_text': len(matches) > 0
    }


def analyze_text_quality(text: str, lazy_patterns: List[str] = None) -> Dict[str, float]:
    """Comprehensive text quality analysis with lazy text detection"""
    if not text or not isinstance(text, str):
        return {
            'quality_score': 0.0,
            'repetition_penalty': 0.0,
            'coherence_score': 0.0,
            'readability_score': 0.0,
            'lazy_score': 0.0,
            'overall_quality': 0.0
        }
    
    # Get all metrics
    repetition_metrics = calculate_repetition_metrics(text)
    coherence_metrics = calculate_coherence_metrics(text)
    readability_metrics = calculate_readability_metrics(text)
    lazy_metrics = detect_lazy_uncreative_text(text, lazy_patterns)
    
    # Calculate overall quality score (weighted average)
    # Penalize lazy text heavily since it indicates poor creativity
    lazy_penalty = lazy_metrics['lazy_score']
    overall_quality = (
        (1.0 - repetition_metrics['repetition_penalty']) * 0.3 +
        coherence_metrics['coherence_score'] * 0.2 +
        readability_metrics['readability_score'] * 0.2 +
        (1.0 - lazy_penalty) * 0.3  # Heavy weighting against lazy text
    )
    
    # Combine all metrics
    all_metrics = {
        'quality_score': overall_quality,
        'repetition_penalty': repetition_metrics['repetition_penalty'],
        'coherence_score': coherence_metrics['coherence_score'],
        'readability_score': readability_metrics['readability_score'],
        'lazy_score': lazy_metrics['lazy_score'],
        'overall_quality': overall_quality
    }
    
    # Add all individual metrics
    all_metrics.update(repetition_metrics)
    all_metrics.update(coherence_metrics)
    all_metrics.update(readability_metrics)
    all_metrics.update(lazy_metrics)
    
    return all_metrics


def analyze_results(results: list, custom_lazy_patterns: List[str] = None) -> pd.DataFrame:
    """Convert results to pandas DataFrame for analysis with text quality metrics"""
    
    data = []
    
    for result in results:
        params = result['parameters']
        response_text = result['response']
        
        # Calculate text quality metrics with error handling
        try:
            quality_metrics = analyze_text_quality(response_text, custom_lazy_patterns)
            
            # Ensure all expected keys are present with default values
            default_metrics = {
                'quality_score': 0.0,
                'repetition_penalty': 0.0,
                'coherence_score': 0.0,
                'readability_score': 0.0,
                'lazy_score': 0.0,
                'lazy_pattern_count': 0,
                'has_lazy_text': False,
                'token_repetition_rate': 0.0,
                'bigram_repetition_rate': 0.0,
                'trigram_repetition_rate': 0.0,
                'consecutive_repetition_score': 0.0,
                'sentence_count': 0,
                'avg_sentence_length': 0.0,
                'sentence_length_variation': 0.0,
                'word_count': 0,
                'avg_word_length': 0.0,
                'lexical_diversity': 0.0,
                'overall_quality': 0.0
            }
            
            # Merge with actual metrics, using defaults for any missing keys
            quality_metrics = {**default_metrics, **quality_metrics}
            
        except Exception as e:
            print(f"Error analyzing text quality for response: {e}")
            # Use all default values if analysis fails
            quality_metrics = {
                'quality_score': 0.0,
                'repetition_penalty': 0.0,
                'coherence_score': 0.0,
                'readability_score': 0.0,
                'lazy_score': 0.0,
                'lazy_pattern_count': 0,
                'has_lazy_text': False,
                'token_repetition_rate': 0.0,
                'bigram_repetition_rate': 0.0,
                'trigram_repetition_rate': 0.0,
                'consecutive_repetition_score': 0.0,
                'sentence_count': 0,
                'avg_sentence_length': 0.0,
                'sentence_length_variation': 0.0,
                'word_count': 0,
                'avg_word_length': 0.0,
                'lexical_diversity': 0.0,
                'overall_quality': 0.0
            }
        
        data.append({
            'prompt': result['prompt'],
            'temperature': params['temperature'],
            'min_p': params['min_p'],
            'adaptive_target': params['adaptive_target'],
            'adaptive_decay': params['adaptive_decay'],
            'top_nsigma': params.get('top_nsigma', 1.0),  # Default to 1.0 if not present
            'response_length': len(response_text) if isinstance(response_text, str) else 0,
            'success': result['success'],
            'response': response_text,
            # Text quality metrics (with guaranteed defaults)
            'quality_score': quality_metrics['quality_score'],
            'repetition_penalty': quality_metrics['repetition_penalty'],
            'coherence_score': quality_metrics['coherence_score'],
            'readability_score': quality_metrics['readability_score'],
            'lazy_score': quality_metrics['lazy_score'],
            'lazy_pattern_count': quality_metrics['lazy_pattern_count'],
            'has_lazy_text': quality_metrics['has_lazy_text'],
            'token_repetition_rate': quality_metrics['token_repetition_rate'],
            'bigram_repetition_rate': quality_metrics['bigram_repetition_rate'],
            'trigram_repetition_rate': quality_metrics['trigram_repetition_rate'],
            'consecutive_repetition_score': quality_metrics['consecutive_repetition_score'],
            'sentence_count': quality_metrics['sentence_count'],
            'avg_sentence_length': quality_metrics['avg_sentence_length'],
            'sentence_length_variation': quality_metrics['sentence_length_variation'],
            'word_count': quality_metrics['word_count'],
            'avg_word_length': quality_metrics['avg_word_length'],
            'lexical_diversity': quality_metrics['lexical_diversity'],
            'overall_quality': quality_metrics['overall_quality']
        })
    
    return pd.DataFrame(data)


def plot_parameter_distributions(df: pd.DataFrame):
    """Plot distributions of parameter values"""
    
    # Only include top_nsigma in plots if it was varied
    params_to_plot = ['temperature', 'min_p', 'adaptive_target']
    if 'top_nsigma' in df.columns and len(df['top_nsigma'].unique()) > 1:
        params_to_plot.append('top_nsigma')
    
    plt.figure(figsize=(15, 10))
    
    for i, param in enumerate(params_to_plot, 1):
        plt.subplot(2, 2, i)
        sns.histplot(df[param], bins=10, kde=True)
        plt.title(f'Distribution of {param}')
        plt.xlabel(param)
        plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('parameter_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Saved parameter distributions plot to parameter_distributions.png")


def plot_quality_metrics(df: pd.DataFrame):
    """Plot text quality metrics including lazy text analysis"""
    
    quality_metrics = ['quality_score', 'repetition_penalty', 'coherence_score', 'readability_score']
    
    plt.figure(figsize=(16, 12))
    
    # Overall quality score distribution
    plt.subplot(2, 2, 1)
    sns.histplot(df['quality_score'], bins=20, kde=True)
    plt.title('Overall Quality Score Distribution')
    plt.xlabel('Quality Score')
    plt.ylabel('Count')
    
    # Repetition metrics
    plt.subplot(2, 2, 2)
    sns.histplot(df['repetition_penalty'], bins=20, kde=True, color='orange')
    plt.title('Repetition Penalty Distribution')
    plt.xlabel('Repetition Penalty (lower = better)')
    plt.ylabel('Count')
    
    # Coherence vs Repetition scatter
    plt.subplot(2, 2, 3)
    sns.scatterplot(x='repetition_penalty', y='coherence_score', 
                    hue='quality_score', palette='viridis', data=df)
    plt.title('Coherence vs Repetition (colored by Quality)')
    plt.xlabel('Repetition Penalty')
    plt.ylabel('Coherence Score')
    
    # Readability vs Quality
    plt.subplot(2, 2, 4)
    sns.scatterplot(x='readability_score', y='quality_score', data=df)
    plt.title('Readability vs Quality')
    plt.xlabel('Readability Score')
    plt.ylabel('Quality Score')
    
    plt.tight_layout()
    plt.savefig('quality_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Saved quality metrics plot to quality_metrics.png")


def plot_lazy_text_analysis(df: pd.DataFrame):
    """Analyze lazy/uncreative text patterns"""
    
    plt.figure(figsize=(16, 12))
    
    # Lazy score distribution
    plt.subplot(2, 2, 1)
    sns.histplot(df['lazy_score'], bins=20, kde=True, color='red')
    plt.title('Lazy/Uncreative Text Score Distribution')
    plt.xlabel('Lazy Score (higher = more lazy/uncreative)')
    plt.ylabel('Count')
    
    # Lazy pattern count distribution
    plt.subplot(2, 2, 2)
    sns.histplot(df['lazy_pattern_count'], bins=10, kde=True, color='purple')
    plt.title('Lazy Pattern Count Distribution')
    plt.xlabel('Number of Lazy Patterns Found')
    plt.ylabel('Count')
    
    # Lazy score vs Quality score
    plt.subplot(2, 2, 3)
    sns.scatterplot(x='lazy_score', y='quality_score', 
                    hue='repetition_penalty', palette='RdYlGn_r', data=df)
    plt.title('Lazy Score vs Quality Score')
    plt.xlabel('Lazy Score')
    plt.ylabel('Quality Score')
    
    # Lazy text by parameter (temperature)
    plt.subplot(2, 2, 4)
    sns.boxplot(x='temperature', y='lazy_score', data=df)
    plt.title('Lazy Score by Temperature')
    plt.xlabel('Temperature')
    plt.ylabel('Lazy Score')
    
    plt.tight_layout()
    plt.savefig('lazy_text_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Saved lazy text analysis plot to lazy_text_analysis.png")


def plot_repetition_analysis(df: pd.DataFrame):
    """Detailed analysis of repetition metrics"""
    
    plt.figure(figsize=(16, 12))
    
    # Token repetition by parameter
    plt.subplot(2, 2, 1)
    sns.boxplot(x='temperature', y='token_repetition_rate', data=df)
    plt.title('Token Repetition by Temperature')
    plt.xlabel('Temperature')
    plt.ylabel('Token Repetition Rate')
    
    # Bigram repetition by parameter
    plt.subplot(2, 2, 2)
    sns.boxplot(x='min_p', y='bigram_repetition_rate', data=df)
    plt.title('Bigram Repetition by min_p')
    plt.xlabel('min_p')
    plt.ylabel('Bigram Repetition Rate')
    
    # Consecutive repetition by parameter
    plt.subplot(2, 2, 3)
    sns.boxplot(x='adaptive_target', y='consecutive_repetition_score', data=df)
    plt.title('Consecutive Repetition by Adaptive Target')
    plt.xlabel('Adaptive Target')
    plt.ylabel('Consecutive Repetition Score')
    
    # Overall repetition penalty by parameter
    plt.subplot(2, 2, 4)
    sns.boxplot(x='temperature', y='repetition_penalty', data=df)
    plt.title('Overall Repetition Penalty by Temperature')
    plt.xlabel('Temperature')
    plt.ylabel('Repetition Penalty')
    
    plt.tight_layout()
    plt.savefig('repetition_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Saved repetition analysis plot to repetition_analysis.png")


def plot_coherence_analysis(df: pd.DataFrame):
    """Analysis of coherence metrics"""
    
    plt.figure(figsize=(16, 8))
    
    # Coherence score by parameter
    plt.subplot(1, 2, 1)
    sns.boxplot(x='temperature', y='coherence_score', data=df)
    plt.title('Coherence Score by Temperature')
    plt.xlabel('Temperature')
    plt.ylabel('Coherence Score')
    
    # Sentence length variation by parameter
    plt.subplot(1, 2, 2)
    sns.boxplot(x='min_p', y='sentence_length_variation', data=df)
    plt.title('Sentence Length Variation by min_p')
    plt.xlabel('min_p')
    plt.ylabel('Sentence Length Variation')
    
    plt.tight_layout()
    plt.savefig('coherence_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Saved coherence analysis plot to coherence_analysis.png")


def plot_response_length_analysis(df: pd.DataFrame):
    """Analyze how parameters affect response length"""
    
    plt.figure(figsize=(18, 12))
    
    # Temperature vs Response Length
    plt.subplot(2, 2, 1)
    sns.boxplot(x='temperature', y='response_length', data=df)
    plt.title('Response Length by Temperature')
    plt.xlabel('Temperature')
    plt.ylabel('Response Length (characters)')
    
    # min_p vs Response Length
    plt.subplot(2, 2, 2)
    sns.boxplot(x='min_p', y='response_length', data=df)
    plt.title('Response Length by min_p')
    plt.xlabel('min_p')
    plt.ylabel('Response Length (characters)')
    
    # adaptive_target vs Response Length
    plt.subplot(2, 2, 3)
    sns.boxplot(x='adaptive_target', y='response_length', data=df)
    plt.title('Response Length by Adaptive Target')
    plt.xlabel('Adaptive Target')
    plt.ylabel('Response Length (characters)')
    
    # Only plot top_nsigma if it was varied in the sweep
    if 'top_nsigma' in df.columns and len(df['top_nsigma'].unique()) > 1:
        plt.subplot(2, 2, 4)
        sns.boxplot(x='top_nsigma', y='response_length', data=df)
        plt.title('Response Length by top_nsigma')
        plt.xlabel('top_nsigma')
        plt.ylabel('Response Length (characters)')
    
    plt.tight_layout()
    plt.savefig('response_length_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Saved response length analysis plot to response_length_analysis.png")


def plot_parameter_correlations(df: pd.DataFrame):
    """Plot correlations between parameters"""
    
    # Select only numeric parameters
    # Include top_nsigma in correlation matrix if it was varied
    corr_columns = ['temperature', 'min_p', 'adaptive_target', 'response_length']
    if 'top_nsigma' in df.columns and len(df['top_nsigma'].unique()) > 1:
        corr_columns.append('top_nsigma')
    numeric_df = df[corr_columns]
    
    plt.figure(figsize=(12, 8))
    corr_matrix = numeric_df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Parameter Correlation Matrix')
    plt.savefig('parameter_correlations.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Saved parameter correlation plot to parameter_correlations.png")


def generate_summary_statistics(df: pd.DataFrame):
    """Generate summary statistics"""
    
    stats = {
        'total_requests': len(df),
        'success_rate': df['success'].mean(),
        'avg_response_length': df['response_length'].mean(),
        'median_response_length': df['response_length'].median(),
        'min_response_length': df['response_length'].min(),
        'max_response_length': df['response_length'].max(),
        'temperature_range': f"{df['temperature'].min()} - {df['temperature'].max()}",
        'min_p_range': f"{df['min_p'].min()} - {df['min_p'].max()}",
        'adaptive_target_range': f"{df['adaptive_target'].min()} - {df['adaptive_target'].max()}",
        'top_nsigma_range': f"{df['top_nsigma'].min()} - {df['top_nsigma'].max()}" if 'top_nsigma' in df.columns else '1.0 (fixed)',
        # Text quality metrics
        'avg_quality_score': df['quality_score'].mean(),
        'avg_repetition_penalty': df['repetition_penalty'].mean(),
        'avg_coherence_score': df['coherence_score'].mean(),
        'avg_readability_score': df['readability_score'].mean(),
        'avg_overall_quality': df['overall_quality'].mean(),
        'best_quality_score': df['quality_score'].max(),
        'worst_quality_score': df['quality_score'].min()
    }
    
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    # Basic statistics
    print("BASIC STATISTICS:")
    basic_stats = {
        'total_requests': stats['total_requests'],
        'success_rate': stats['success_rate'],
        'avg_response_length': stats['avg_response_length'],
        'temperature_range': stats['temperature_range'],
        'min_p_range': stats['min_p_range'],
        'adaptive_target_range': stats['adaptive_target_range'],
        'top_nsigma_range': stats['top_nsigma_range']
    }
    
    for key, value in basic_stats.items():
        print(f"  {key:>20}: {value}")
    
    print("\nTEXT QUALITY METRICS:")
    quality_stats = {
        'avg_overall_quality': stats['avg_overall_quality'],
        'avg_quality_score': stats['avg_quality_score'],
        'best_quality_score': stats['best_quality_score'],
        'worst_quality_score': stats['worst_quality_score'],
        'avg_repetition_penalty': stats['avg_repetition_penalty'],
        'avg_coherence_score': stats['avg_coherence_score'],
        'avg_readability_score': stats['avg_readability_score']
    }
    
    for key, value in quality_stats.items():
        print(f"  {key:>20}: {value:.3f}")
    
    print("="*60)
    
    return stats


def find_optimal_parameters(df: pd.DataFrame, metric: str = 'quality_score'):
    """Find parameter combinations that optimize a given metric"""
    
    # Group by all parameters that were varied
    group_columns = ['temperature', 'min_p', 'adaptive_target']
    if 'top_nsigma' in df.columns and len(df['top_nsigma'].unique()) > 1:
        group_columns.append('top_nsigma')
    
    # Calculate average metrics for each parameter combination
    grouped = df.groupby(group_columns).agg({
        'quality_score': 'mean',
        'repetition_penalty': 'mean',
        'coherence_score': 'mean',
        'readability_score': 'mean',
        'response_length': 'mean',
        'response': lambda x: list(x)  # Keep sample responses
    }).reset_index()
    
    # Calculate overall score with inverse weighting for repetition
    grouped['overall_score'] = (
        grouped['quality_score'] * 0.4 +
        (1 - grouped['repetition_penalty']) * 0.3 +  # Inverse weighted
        grouped['coherence_score'] * 0.2 +
        grouped['readability_score'] * 0.1
    )
    
    # Find top 10 combinations
    top_combinations = grouped.nlargest(10, 'overall_score')
    
    print(f"\nTop 10 parameter combinations by overall score:")
    print("="*80)
    
    for i, row in top_combinations.iterrows():
        params_str = f"Temperature: {row['temperature']:.2f}, min_p: {row['min_p']:.2f}, adaptive_target: {row['adaptive_target']:.2f}"
        if 'top_nsigma' in row:
            params_str += f", top_nsigma: {row['top_nsigma']:.1f}"
        print(f"{i+1}. {params_str}")
        print(f"   Overall Score: {row['overall_score']:.3f}")
        print(f"   Quality: {row['quality_score']:.3f}, Repetition: {row['repetition_penalty']:.3f}, Coherence: {row['coherence_score']:.3f}")
        print("-" * 60)
    
    # Generate human-readable markdown report
    generate_top_parameters_report(top_combinations, df)
    
    return top_combinations


def generate_top_parameters_report(top_combinations: pd.DataFrame, full_df: pd.DataFrame):
    """Generate a human-readable markdown report of top parameter combinations"""
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_filename = f"top_parameters_report_{timestamp}.md"
    
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write("# Top Parameter Combinations Report\n\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Summary\n\n")
        f.write("This report presents the top 10 parameter combinations based on overall text quality score.\n")
        f.write("The overall score is calculated with inverse weighting for repetition (lower repetition = better).\n\n")
        
        f.write("## Scoring Methodology\n\n")
        f.write("- **Overall Score** = (Quality × 0.4) + ((1 - Repetition) × 0.3) + (Coherence × 0.2) + (Readability × 0.1)\n")
        f.write("- **Quality Score**: Composite measure of text quality (0-1, higher = better)\n")
        f.write("- **Repetition Penalty**: Measures repetitiveness (0-1, lower = better)\n")
        f.write("- **Coherence Score**: Sentence structure consistency (0-1, higher = better)\n")
        f.write("- **Readability Score**: Vocabulary diversity and complexity (0-1, higher = better)\n\n")
        
        f.write("## Top 10 Parameter Combinations\n\n")
        
        for i, row in top_combinations.iterrows():
            f.write(f"### Rank {i+1}\n\n")
            
            # Parameter details
            f.write("**Parameters:**\n")
            f.write(f"- Temperature: `{row['temperature']:.2f}`\n")
            f.write(f"- min_p: `{row['min_p']:.3f}`\n")
            f.write(f"- adaptive_target: `{row['adaptive_target']:.2f}`\n")
            if 'top_nsigma' in row:
                f.write(f"- top_nsigma: `{row['top_nsigma']:.1f}`\n")
            f.write(f"- adaptive_decay: `0.9` (fixed)\n\n")
            
            # Quality metrics
            f.write("**Quality Metrics:**\n")
            f.write(f"- Overall Score: **{row['overall_score']:.3f}**\n")
            f.write(f"- Quality Score: {row['quality_score']:.3f}\n")
            f.write(f"- Repetition Penalty: {row['repetition_penalty']:.3f}\n")
            f.write(f"- Coherence Score: {row['coherence_score']:.3f}\n")
            f.write(f"- Readability Score: {row['readability_score']:.3f}\n")
            f.write(f"- Average Response Length: {row['response_length']:.1f} characters\n\n")
            
            # Sample response
            if len(row['response']) > 0:
                sample_response = row['response'][0][:200] + "..." if len(row['response'][0]) > 200 else row['response'][0]
                f.write("**Sample Response:**\n")
                f.write(f"```\n{sample_response}\n```\n\n")
            
            # Interpretation
            f.write("**Interpretation:**\n")
            
            # Temperature interpretation
            if row['temperature'] < 1.05:
                f.write("- Lower temperature suggests more deterministic, focused output\n")
            elif row['temperature'] >= 1.05:
                f.write("- Higher temperature suggests more creative, diverse output\n")
            
            # Repetition interpretation
            if row['repetition_penalty'] < 0.3:
                f.write("- Excellent repetition control - text is varied and non-repetitive\n")
            elif row['repetition_penalty'] < 0.5:
                f.write("- Good repetition control - minimal repetitive patterns\n")
            else:
                f.write("- Some repetition detected - may benefit from parameter adjustment\n")
            
            # Coherence interpretation
            if row['coherence_score'] > 0.7:
                f.write("- High coherence - sentences flow well together\n")
            elif row['coherence_score'] > 0.5:
                f.write("- Moderate coherence - generally logical progression\n")
            else:
                f.write("- Lower coherence - may have some disjointed sentences\n")
            
            f.write("\n---\n\n")
        
        # Recommendations section
        f.write("## Recommendations\n\n")
        
        # Find the best parameter for each metric
        best_temp = top_combinations.loc[top_combinations['overall_score'].idxmax(), 'temperature']
        best_min_p = top_combinations.loc[top_combinations['overall_score'].idxmax(), 'min_p']
        best_adaptive = top_combinations.loc[top_combinations['overall_score'].idxmax(), 'adaptive_target']
        
        f.write(f"**Optimal Parameter Ranges:**\n")
        f.write(f"- Temperature: Around `{best_temp:.2f}`\n")
        f.write(f"- min_p: Around `{best_min_p:.3f}`\n")
        f.write(f"- adaptive_target: Around `{best_adaptive:.2f}`\n\n")
        
        f.write("**Suggested Starting Configuration:**\n")
        f.write(f"```json\n")
        f.write('{\n')
        f.write(f'  "temperature": {best_temp:.2f},\n')
        f.write(f'  "min_p": {best_min_p:.3f},\n')
        f.write(f'  "adaptive_target": {best_adaptive:.2f},\n')
        f.write('  "adaptive_decay": 0.9\n')
        f.write('}\n')
        f.write(f"```\n\n")
        
        f.write("**Next Steps:**\n")
        f.write("- Test these top parameter combinations with your specific prompts\n")
        f.write("- Fine-tune around these values for optimal results\n")
        f.write("- Consider running additional sweeps with narrower ranges around these optima\n")
        
        f.write("\n---\n\n")
        f.write(f"*Report generated by analyze_results.py*\n")
    
    print(f"\nGenerated human-readable report: {report_filename}")
    print("This report contains the top 10 parameter combinations with detailed analysis.")


def main():
    """Main analysis function"""
    
    # Find the most recent results file
    results_files = [f for f in os.listdir('.') if f.startswith('parameter_sweep_results_') and f.endswith('.json')]
    
    if not results_files:
        print("No results files found. Please run parameter_sweep.py first.")
        return
    
    # Use the most recent file
    results_files.sort(reverse=True)
    latest_file = results_files[0]
    
    print(f"Loading results from: {latest_file}")
    
    # Load and analyze results
    results = load_results(latest_file)
    df = analyze_results(results)
    
    # Generate analysis
    generate_summary_statistics(df)
    plot_parameter_distributions(df)
    plot_response_length_analysis(df)
    plot_parameter_correlations(df)
    
    # New text quality analysis
    plot_quality_metrics(df)
    plot_repetition_analysis(df)
    plot_coherence_analysis(df)
    
    find_optimal_parameters(df)
    
    # Save processed data
    processed_filename = latest_file.replace('.json', '_processed.csv')
    df.to_csv(processed_filename, index=False)
    print(f"\nProcessed data saved to: {processed_filename}")
    
    print("\nAnalysis complete! Check the generated plots and CSV file.")


if __name__ == "__main__":
    main()