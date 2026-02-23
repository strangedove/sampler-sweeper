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
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import numpy as np
import re
from typing import Any, List, Dict, Tuple
import string

# Slop-guard: structural/rhetorical AI-tell detection
# (from https://github.com/eric-tramel/slop-guard, MCP wrapper stripped)
try:
    from slop_guard import analyze as slop_guard_analyze
    HAS_SLOP_GUARD = True
except ImportError:
    HAS_SLOP_GUARD = False
    print("slop_guard module not found - structural AI-tell detection disabled")

# For advanced NLP metrics (if available)
try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk import ngrams

    # Download NLTK data with a timeout so slow connections don't block
    import threading

    def _nltk_download():
        for pkg in ('punkt', 'punkt_tab', 'stopwords', 'averaged_perceptron_tagger'):
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass

    _dl = threading.Thread(target=_nltk_download, daemon=True)
    _dl.start()
    _dl.join(timeout=15)  # wait at most 15 seconds

    # Verify the tokenizer actually works; if data is missing fall back
    try:
        word_tokenize("test")
        HAS_NLTK = True
    except LookupError:
        HAS_NLTK = False
        print("NLTK data not yet downloaded - falling back to simple tokenization")
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

    # ------------------------------------------------------------------
    # Sentence-level duplicate detection
    # ------------------------------------------------------------------
    # Split into sentences and normalise whitespace for comparison
    if HAS_NLTK:
        sentences = sent_tokenize(text)
    else:
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

    num_sentences = len(sentences)
    if num_sentences >= 2:
        normalised = [re.sub(r'\s+', ' ', s.strip().lower()) for s in sentences]
        unique_sentences = len(set(normalised))
        # Ratio of duplicated sentences (0 = all unique, 1 = all copies)
        sentence_duplicate_ratio = 1.0 - unique_sentences / num_sentences
    else:
        sentence_duplicate_ratio = 0.0

    # ------------------------------------------------------------------
    # Long-span (sequence) repetition detection
    # ------------------------------------------------------------------
    # Slide a window of W words across the text and check how many windows
    # are near-duplicates of an earlier window. This catches the "stuck in
    # a loop" failure mode where multi-sentence blocks repeat verbatim.
    W = 30  # window size in tokens (roughly one sentence)
    if len(tokens) >= W * 2:
        seen: set = set()
        dupes = 0
        total_windows = len(tokens) - W + 1
        for i in range(total_windows):
            window = ' '.join(tokens[i:i + W])
            if window in seen:
                dupes += 1
            else:
                seen.add(window)
        sequence_repetition_score = dupes / total_windows
    else:
        sequence_repetition_score = 0.0

    # ------------------------------------------------------------------
    # Overall repetition penalty (composite score)
    # ------------------------------------------------------------------
    # The n-gram metrics catch small-scale word reuse.
    # sentence_duplicate_ratio catches whole-sentence copy-paste.
    # sequence_repetition_score catches multi-sentence looping.
    # We take the MAX of the sentence/sequence scores with the n-gram
    # composite so that any single severe failure mode dominates.
    ngram_penalty = (
        token_repetition_rate * 0.4 +
        bigram_repetition_rate * 0.3 +
        trigram_repetition_rate * 0.2 +
        consecutive_repetition_score * 0.1
    )
    block_penalty = max(sentence_duplicate_ratio, sequence_repetition_score)
    repetition_penalty = max(ngram_penalty, block_penalty)

    return {
        'token_repetition_rate': token_repetition_rate,
        'bigram_repetition_rate': bigram_repetition_rate,
        'trigram_repetition_rate': trigram_repetition_rate,
        'consecutive_repetition_score': consecutive_repetition_score,
        'sentence_duplicate_ratio': sentence_duplicate_ratio,
        'sequence_repetition_score': sequence_repetition_score,
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


# ---------------------------------------------------------------------------
# "Telling" verbs — weak constructions like "she felt sad" instead of showing
# ---------------------------------------------------------------------------
TELLING_VERBS = {
    'seemed', 'felt', 'realized', 'knew', 'noticed', 'sensed',
    'understood', 'recognized', 'appeared', 'looked',  # "looked" as "looked happy"
    'sounded', 'wondered', 'thought', 'decided', 'believed',
}


def calculate_prose_quality_metrics(text: str) -> Dict[str, float]:
    """Measure prose-craft signals that aren't covered by repetition/slop checks.

    Returns a dict with individual metrics AND a composite ``prose_penalty``
    (0 = no problems, approaches 1.0 for severely monotonous / staccato /
    tell-heavy prose).

    Metrics computed
    ----------------
    sentence_start_monotony : float  0-1
        How repetitive the first word(s) of each sentence are.  Measured via
        normalised entropy of the (first_word,) distribution. 0 = every
        sentence starts differently, 1 = every sentence starts the same way.

    word_frequency_spike : float  0-1
        Worst-case overuse of any single non-stopword, expressed as a 0-1
        penalty.  "seemed" appearing 8× in 400 words would score high.

    sentence_length_uniformity : float  0-1
        Inverse of sentence-length coefficient of variation.  Prose where
        every sentence is the same length scores near 1.  Good prose with
        varied rhythm scores near 0.

    telling_verb_density : float  0-1
        Proportion of sentences that contain a "telling" verb (seemed, felt,
        realized …).  High density → tell-don't-show writing.

    paragraph_uniformity : float  0-1
        How uniform paragraph lengths are.  Pure staccato (every paragraph is
        one sentence) or pure wall-of-text (single giant block) both score
        high.  Mixed paragraph lengths score low.

    dialogue_narration_balance : float  0-1
        Penalty for having ALL dialogue or ALL narration.  A healthy mix
        scores 0; 100 % one mode scores ~0.5 (mild penalty — some stories
        legitimately have no dialogue).

    prose_penalty : float  0-1
        Composite of the above, designed so that moderate issues across
        several dimensions don't stack too harshly, but a single severe
        problem still registers.
    """
    defaults = {
        'sentence_start_monotony': 0.0,
        'word_frequency_spike': 0.0,
        'sentence_length_uniformity': 0.0,
        'telling_verb_density': 0.0,
        'paragraph_uniformity': 0.0,
        'dialogue_narration_balance': 0.0,
        'prose_penalty': 0.0,
    }

    if not text or not isinstance(text, str):
        return defaults

    # --- Sentence splitting ------------------------------------------------
    if HAS_NLTK:
        sentences = sent_tokenize(text)
    else:
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) < 3:
        return defaults  # too little text to judge

    # --- Tokenise for word-level stats ------------------------------------
    if HAS_NLTK:
        all_tokens = word_tokenize(text)
        words = [w.lower() for w in all_tokens if w.isalpha()]
        try:
            stop_words = set(stopwords.words('english'))
        except LookupError:
            stop_words = set()
    else:
        words = [w.lower() for w in re.findall(r'[a-zA-Z]+', text)]
        # Minimal stopword set if NLTK unavailable
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'from', 'is', 'was', 'are', 'were',
            'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'shall',
            'not', 'no', 'nor', 'so', 'yet', 'as', 'if', 'it', 'its',
            'that', 'this', 'these', 'those', 'i', 'me', 'my', 'we', 'our',
            'you', 'your', 'he', 'him', 'his', 'she', 'her', 'they', 'them',
            'their', 'what', 'which', 'who', 'whom', 'how', 'when', 'where',
            'there', 'here', 'all', 'each', 'every', 'both', 'few', 'more',
            'most', 'other', 'some', 'such', 'than', 'too', 'very', 'just',
            'about', 'up', 'out', 'then', 'into',
        }

    word_count = len(words)
    if word_count < 20:
        return defaults

    # ======================================================================
    # 1. Sentence-start monotony
    # ======================================================================
    # Take the first word of each sentence (lowered, alpha only)
    first_words = []
    for s in sentences:
        m = re.match(r'["\']?\s*([a-zA-Z]+)', s)
        if m:
            first_words.append(m.group(1).lower())

    if first_words:
        fw_counts = Counter(first_words)
        n = len(first_words)
        # Shannon entropy of the first-word distribution
        entropy = -sum((c / n) * np.log2(c / n) for c in fw_counts.values())
        # Maximum entropy for n items with k unique values
        max_entropy = np.log2(n) if n > 1 else 1.0
        # Normalise: 1 = all same, 0 = all different
        normalised_entropy = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.0
        # Also check the *dominant* first word — if one pronoun starts >50%
        # of sentences, that's monotonous regardless of entropy
        most_common_frac = fw_counts.most_common(1)[0][1] / n
        # Blend: entropy catches overall flatness, dominant-fraction catches
        # "she she she" specifically
        sentence_start_monotony = max(normalised_entropy, most_common_frac - 0.3)
        sentence_start_monotony = np.clip(sentence_start_monotony, 0.0, 1.0)
    else:
        sentence_start_monotony = 0.0

    # ======================================================================
    # 2. Non-stopword frequency spike
    # ======================================================================
    content_words = [w for w in words if w not in stop_words and len(w) > 2]
    if content_words:
        cw_counts = Counter(content_words)
        # Density = occurrences per 100 words for the most-repeated content word
        worst_density = cw_counts.most_common(1)[0][1] / (word_count / 100.0)
        # A content word appearing 1× per 100 words is normal.
        # 3× per 100 words is noticeable.  5+ is bad.
        # Sigmoid curve: 0 at density≤1, ~0.5 at density=3, ~0.8 at density=5
        word_frequency_spike = 1.0 - 1.0 / (1.0 + max(0, worst_density - 1.0) / 2.0)
    else:
        word_frequency_spike = 0.0

    # ======================================================================
    # 3. Sentence-length uniformity
    # ======================================================================
    sent_lengths = [len(s.split()) for s in sentences]
    mean_len = np.mean(sent_lengths)
    std_len = np.std(sent_lengths)
    # Coefficient of variation — good prose has CV ≈ 0.4-0.8
    cv = std_len / mean_len if mean_len > 0 else 0.0
    # Low CV = monotonous rhythm.  We penalise CV < 0.3 heavily, 0.3-0.5
    # mildly, >0.5 not at all.
    if cv >= 0.5:
        sentence_length_uniformity = 0.0
    elif cv >= 0.2:
        # Linear ramp from 0 at cv=0.5 to ~0.7 at cv=0.2
        sentence_length_uniformity = (0.5 - cv) / 0.3 * 0.7
    else:
        sentence_length_uniformity = 0.7 + (0.2 - cv) / 0.2 * 0.3

    # ======================================================================
    # 4. Telling-verb density
    # ======================================================================
    telling_count = 0
    for s in sentences:
        s_words = set(re.findall(r'[a-zA-Z]+', s.lower()))
        if s_words & TELLING_VERBS:
            telling_count += 1
    telling_ratio = telling_count / len(sentences)
    # Up to 30% of sentences having a telling verb is fine.
    # 50%+ is problematic.  80%+ is terrible.
    if telling_ratio <= 0.3:
        telling_verb_density = 0.0
    else:
        telling_verb_density = min(1.0, (telling_ratio - 0.3) / 0.5)

    # ======================================================================
    # 5. Paragraph structure
    # ======================================================================
    # Split on double-newlines (or treat the whole text as one paragraph)
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    n_para = len(paragraphs)
    if n_para == 1:
        # Single wall of text — mild penalty (some completions are short
        # enough that one paragraph is fine)
        para_sent_counts = [len(sentences)]
        paragraph_uniformity = 0.3 if len(sentences) > 8 else 0.0
    else:
        # Count sentences per paragraph
        para_sent_counts = []
        for p in paragraphs:
            if HAS_NLTK:
                p_sents = sent_tokenize(p)
            else:
                p_sents = [s.strip() for s in re.split(r'[.!?]+', p) if s.strip()]
            para_sent_counts.append(max(1, len(p_sents)))

        mean_ps = np.mean(para_sent_counts)
        std_ps = np.std(para_sent_counts)
        cv_para = std_ps / mean_ps if mean_ps > 0 else 0.0

        # Staccato detection: if mean paragraph is ~1 sentence and there
        # are many paragraphs, that's the annoying one-line-paragraph style
        if mean_ps <= 1.5 and n_para >= 6:
            staccato = 0.6
        elif mean_ps <= 2.0 and n_para >= 8:
            staccato = 0.4
        else:
            staccato = 0.0

        # Uniformity: same logic as sentence length — low CV = every
        # paragraph is the same size = boring
        if cv_para >= 0.4:
            uniformity = 0.0
        else:
            uniformity = (0.4 - cv_para) / 0.4 * 0.4

        paragraph_uniformity = max(staccato, uniformity)

    # ======================================================================
    # 6. Dialogue / narration balance
    # ======================================================================
    # Count lines that look like dialogue (contain quoted speech)
    dialogue_sents = 0
    for s in sentences:
        if re.search(r'["\u201c\u201d]', s) or re.search(r"['\u2018\u2019].*said", s, re.I):
            dialogue_sents += 1

    dialogue_frac = dialogue_sents / len(sentences) if sentences else 0.0
    # Ideal range is roughly 0.15 - 0.60.
    # All-narration (0%) gets a small penalty.  All-dialogue (100%) gets a
    # bigger one.  We use a U-shaped curve centred around 0.35.
    if 0.10 <= dialogue_frac <= 0.65:
        dialogue_narration_balance = 0.0
    elif dialogue_frac < 0.10:
        # No dialogue — mild penalty, some stories legitimately have none
        dialogue_narration_balance = (0.10 - dialogue_frac) / 0.10 * 0.25
    else:
        # Mostly dialogue — bigger penalty
        dialogue_narration_balance = min(0.6, (dialogue_frac - 0.65) / 0.35 * 0.6)

    # ======================================================================
    # Composite prose penalty
    # ======================================================================
    # Weighted average — sentence start monotony and telling density are the
    # most noticeable flaws, followed by rhythm issues.
    prose_penalty = (
        sentence_start_monotony * 0.30 +
        word_frequency_spike * 0.20 +
        sentence_length_uniformity * 0.15 +
        telling_verb_density * 0.15 +
        paragraph_uniformity * 0.10 +
        dialogue_narration_balance * 0.10
    )
    # Clamp
    prose_penalty = np.clip(prose_penalty, 0.0, 1.0)

    return {
        'sentence_start_monotony': float(sentence_start_monotony),
        'word_frequency_spike': float(word_frequency_spike),
        'sentence_length_uniformity': float(sentence_length_uniformity),
        'telling_verb_density': float(telling_verb_density),
        'paragraph_uniformity': float(paragraph_uniformity),
        'dialogue_narration_balance': float(dialogue_narration_balance),
        'prose_penalty': float(prose_penalty),
    }


# ---------------------------------------------------------------------------
# Default pattern lists for lazy / uncreative text detection
# ---------------------------------------------------------------------------
# Patterns are loaded from slop_patterns.yaml if present, otherwise use
# minimal fallback patterns. The YAML file has two sections:
#   - literal_strings: case-insensitive substring matches
#   - regex_patterns: regular expressions (compiled with re.IGNORECASE)

# Minimal fallback patterns (used only if YAML file is missing)
_FALLBACK_LITERAL_STRINGS: List[str] = [
    "couldn't help but", "a wave of", "steeled himself", "steeled herself",
    "eyes widened", "let out a breath", "palpable tension", "mysterious stranger",
]
_FALLBACK_REGEX_PATTERNS: List[str] = [
    r'\b(\w+)(?:\s+\1){2,}\b',  # repeated words
]


def load_slop_patterns(yaml_path: str = None) -> Tuple[List[str], List[str]]:
    """
    Load slop detection patterns from a YAML file.

    Args:
        yaml_path: Path to YAML file. If None, looks for 'slop_patterns.yaml'
            in the same directory as this script.

    Returns:
        Tuple of (literal_strings, regex_patterns) lists.
        Falls back to minimal built-in patterns if file is missing or invalid.
    """
    if yaml_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        yaml_path = os.path.join(script_dir, 'slop_patterns.yaml')

    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        literal_strings = data.get('literal_strings', [])
        regex_patterns = data.get('regex_patterns', [])
        if not literal_strings and not regex_patterns:
            raise ValueError("Empty patterns file")
        return literal_strings, regex_patterns
    except Exception:
        # Fall back to minimal built-in patterns
        return _FALLBACK_LITERAL_STRINGS, _FALLBACK_REGEX_PATTERNS


# Load default patterns at import time
DEFAULT_LITERAL_STRINGS, DEFAULT_REGEX_PATTERNS = load_slop_patterns()


def detect_lazy_uncreative_text(
    text: str,
    literal_strings: List[str] = None,
    regex_patterns: List[str] = None,
    patterns: List[str] = None,  # legacy compat: treated as literal_strings
) -> Dict[str, Any]:
    """
    Detect lazy/uncreative text patterns in responses.

    Matching is split into two categories:
      1. **Literal strings** — case-insensitive substring search. Every
         occurrence in the text is counted (so "a wave of ... a wave of"
         counts as 2 hits).
      2. **Regex patterns** — compiled with ``re.IGNORECASE`` and searched
         against the original text. ``findall`` is used so multiple matches
         are counted.

    Args:
        text: The text to analyze.
        literal_strings: Case-insensitive literal substrings to look for.
            If *None*, ``DEFAULT_LITERAL_STRINGS`` is used.
        regex_patterns: Regex pattern strings (compiled internally).
            If *None*, ``DEFAULT_REGEX_PATTERNS`` is used.
        patterns: Legacy parameter — if provided and *literal_strings* is
            None, these are used as literal strings for backward compat.

    Returns:
        Dictionary with:
        - lazy_pattern_count: Total number of match *instances* found
        - lazy_patterns_found: List of (pattern, count) tuples
        - lazy_score: 0–1 (higher = more lazy/uncreative)
        - has_lazy_text: Boolean
    """
    if not text or not isinstance(text, str):
        return {
            'lazy_pattern_count': 0,
            'lazy_patterns_found': [],
            'lazy_score': 0.0,
            'has_lazy_text': False,
        }

    # Resolve which lists to use
    if literal_strings is None:
        literal_strings = list(patterns) if patterns is not None else DEFAULT_LITERAL_STRINGS
    if regex_patterns is None and patterns is None:
        regex_patterns = DEFAULT_REGEX_PATTERNS
    elif regex_patterns is None:
        regex_patterns = []

    text_lower = text.lower()
    found: List[Tuple[str, int]] = []  # (pattern_or_string, hit_count)
    total_hits = 0

    # --- Literal string matching (case-insensitive, count every occurrence) ---
    for s in literal_strings:
        s_lower = s.lower()
        count = text_lower.count(s_lower)
        if count > 0:
            found.append((s, count))
            total_hits += count

    # --- Regex pattern matching ---
    for pat in regex_patterns:
        try:
            compiled = re.compile(pat, re.IGNORECASE)
            hits = compiled.findall(text)
            if hits:
                found.append((pat, len(hits)))
                total_hits += len(hits)
        except re.error:
            # Malformed regex — skip silently
            pass

    # --- Scoring ---
    # We want a score in [0, 1] that:
    #   - accounts for how many hits there are relative to text length
    #   - saturates smoothly (diminishing returns past a certain density)
    #
    # density = total_hits per 500 chars of text (roughly per paragraph)
    # score   = 1 - 1/(1 + density)   (shifted hyperbola, 0→0, inf→1)
    word_count = max(1, len(text.split()))
    density = total_hits / (word_count / 100.0)  # hits per 100 words
    lazy_score = 1.0 - 1.0 / (1.0 + density)

    return {
        'lazy_pattern_count': total_hits,
        'lazy_patterns_found': found,
        'lazy_score': lazy_score,
        'has_lazy_text': total_hits > 0,
    }


def analyze_text_quality(
    text: str,
    lazy_patterns: List[str] = None,
    lazy_literal_strings: List[str] = None,
    lazy_regex_patterns: List[str] = None,
) -> Dict[str, float]:
    """Comprehensive text quality analysis with lazy text detection.

    Args:
        text: The text to analyze.
        lazy_patterns: Legacy shorthand — passed as literal strings if
            *lazy_literal_strings* is not provided.
        lazy_literal_strings: Explicit literal-string list for lazy
            text detection (overrides *lazy_patterns*).
        lazy_regex_patterns: Explicit regex-pattern list for lazy text
            detection.
    """
    if not text or not isinstance(text, str):
        return {
            'quality_score': 0.0,
            'repetition_penalty': 0.0,
            'coherence_score': 0.0,
            'readability_score': 0.0,
            'lazy_score': 0.0,
            'slop_guard_score': 50.0,
            'slop_guard_band': 'unknown',
            'slop_guard_violations': 0,
            'overall_quality': 0.0
        }

    # Get all metrics
    repetition_metrics = calculate_repetition_metrics(text)
    coherence_metrics = calculate_coherence_metrics(text)
    readability_metrics = calculate_readability_metrics(text)
    prose_metrics = calculate_prose_quality_metrics(text)
    lazy_metrics = detect_lazy_uncreative_text(
        text,
        literal_strings=lazy_literal_strings,
        regex_patterns=lazy_regex_patterns,
        patterns=lazy_patterns,
    )

    # Slop-guard: structural/rhetorical AI-tell detection
    if HAS_SLOP_GUARD:
        sg_result = slop_guard_analyze(text)
        slop_guard_score = sg_result.get('score', 50.0)       # 0-100, higher = cleaner
        slop_guard_band = sg_result.get('band', 'unknown')
        slop_guard_violations = len(sg_result.get('violations', []))
    else:
        slop_guard_score = 50.0   # neutral default when module unavailable
        slop_guard_band = 'unavailable'
        slop_guard_violations = 0

    # Normalise slop-guard score to 0-1 (higher = better, matching our convention)
    slop_guard_normalized = slop_guard_score / 100.0

    # Calculate overall quality score
    # Base score: weighted blend of positive signals
    lazy_penalty = lazy_metrics['lazy_score']
    rep_penalty = repetition_metrics['repetition_penalty']
    prose_pen = prose_metrics['prose_penalty']

    base_quality = (
        coherence_metrics['coherence_score'] * 0.20 +
        readability_metrics['readability_score'] * 0.20 +
        (1.0 - lazy_penalty) * 0.30 +
        slop_guard_normalized * 0.30
    )
    # Multiplicative penalties: each one independently crushes the score
    # when severe, but mild values across multiple dimensions don't
    # stack too harshly (e.g. 0.9 * 0.9 * 0.9 = 0.73, not 0.3).
    overall_quality = base_quality * (1.0 - rep_penalty) * (1.0 - prose_pen)

    # Combine all metrics
    all_metrics = {
        'quality_score': overall_quality,
        'repetition_penalty': repetition_metrics['repetition_penalty'],
        'coherence_score': coherence_metrics['coherence_score'],
        'readability_score': readability_metrics['readability_score'],
        'lazy_score': lazy_metrics['lazy_score'],
        'slop_guard_score': slop_guard_score,
        'slop_guard_band': slop_guard_band,
        'slop_guard_violations': slop_guard_violations,
        'overall_quality': overall_quality
    }

    # Add all individual metrics
    all_metrics.update(repetition_metrics)
    all_metrics.update(coherence_metrics)
    all_metrics.update(readability_metrics)
    all_metrics.update(prose_metrics)
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
                'slop_guard_score': 50.0,
                'slop_guard_band': 'unknown',
                'slop_guard_violations': 0,
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
                'slop_guard_score': 50.0,
                'slop_guard_band': 'unknown',
                'slop_guard_violations': 0,
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
            'slop_guard_score': quality_metrics['slop_guard_score'],
            'slop_guard_band': quality_metrics['slop_guard_band'],
            'slop_guard_violations': quality_metrics['slop_guard_violations'],
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