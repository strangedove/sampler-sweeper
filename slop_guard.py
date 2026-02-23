"""Prose linting — detects AI slop patterns in text.

Extracted from https://github.com/eric-tramel/slop-guard (MCP wrapper removed).
Single entry point: analyze(text) returns a dict with score (0-100, higher = cleaner),
band label, violations, and advice.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from functools import partial, reduce
from typing import Callable

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Hyperparameters:
    """Tunable thresholds, caps, and penalties used by the analyzer."""

    # Scoring curve and concentration
    concentration_alpha: float = 2.5
    decay_lambda: float = 0.04
    claude_categories: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {"contrast_pairs", "pithy_fragment", "setup_resolution"}
        )
    )

    # Short-text behavior
    context_window_chars: int = 60
    short_text_word_count: int = 10

    # Repeated n-gram detection
    repeated_ngram_min_n: int = 4
    repeated_ngram_max_n: int = 8
    repeated_ngram_min_count: int = 3

    # Rule penalties and thresholds
    slop_word_penalty: int = -2
    slop_phrase_penalty: int = -3
    structural_bold_header_min: int = 3
    structural_bold_header_penalty: int = -5
    structural_bullet_run_min: int = 6
    structural_bullet_run_penalty: int = -3
    triadic_record_cap: int = 5
    triadic_penalty: int = -1
    triadic_advice_min: int = 3
    tone_penalty: int = -3
    sentence_opener_penalty: int = -2
    weasel_penalty: int = -2
    ai_disclosure_penalty: int = -10
    placeholder_penalty: int = -5
    rhythm_min_sentences: int = 5
    rhythm_cv_threshold: float = 0.3
    rhythm_penalty: int = -5
    em_dash_words_basis: float = 150.0
    em_dash_density_threshold: float = 1.0
    em_dash_penalty: int = -3
    contrast_record_cap: int = 5
    contrast_penalty: int = -1
    contrast_advice_min: int = 2
    setup_resolution_record_cap: int = 5
    setup_resolution_penalty: int = -3
    colon_words_basis: float = 150.0
    colon_density_threshold: float = 1.5
    colon_density_penalty: int = -3
    pithy_max_sentence_words: int = 6
    pithy_record_cap: int = 3
    pithy_penalty: int = -2
    bullet_density_threshold: float = 0.40
    bullet_density_penalty: int = -8
    blockquote_min_lines: int = 3
    blockquote_free_lines: int = 2
    blockquote_cap: int = 4
    blockquote_penalty_step: int = -3
    bold_bullet_run_min: int = 3
    bold_bullet_run_penalty: int = -5
    horizontal_rule_min: int = 4
    horizontal_rule_penalty: int = -3
    phrase_reuse_record_cap: int = 5
    phrase_reuse_penalty: int = -1

    # Score normalization and banding
    density_words_basis: float = 1000.0
    score_min: int = 0
    score_max: int = 100
    band_clean_min: int = 80
    band_light_min: int = 60
    band_moderate_min: int = 40
    band_heavy_min: int = 20


HYPERPARAMETERS = Hyperparameters()


@dataclass(frozen=True)
class Violation:
    rule: str
    match: str
    context: str
    penalty: int

    def to_payload(self) -> dict[str, object]:
        return {
            "type": "Violation",
            "rule": self.rule,
            "match": self.match,
            "context": self.context,
            "penalty": self.penalty,
        }


@dataclass
class RuleContext:
    text: str
    word_count: int
    sentences: list[str]
    advice: list[str]
    counts: dict[str, int]
    hyperparameters: Hyperparameters


@dataclass(frozen=True)
class AnalysisContext:
    text: str
    word_count: int
    sentences: list[str]
    hyperparameters: Hyperparameters


@dataclass
class RuleResult:
    violations: list[Violation]
    advice: list[str]
    count_deltas: dict[str, int]


@dataclass(frozen=True)
class AnalysisState:
    violations: tuple[Violation, ...]
    advice: tuple[str, ...]
    counts: dict[str, int]

    @classmethod
    def initial(cls, counts: dict[str, int]) -> "AnalysisState":
        return cls(violations=(), advice=(), counts=dict(counts))

    def merge(
        self,
        violations: list[Violation],
        advice: list[str],
        count_deltas: dict[str, int],
    ) -> "AnalysisState":
        merged_counts = dict(self.counts)
        for key, delta in count_deltas.items():
            if delta:
                merged_counts[key] = merged_counts.get(key, 0) + delta
        return AnalysisState(
            violations=self.violations + tuple(violations),
            advice=self.advice + tuple(advice),
            counts=merged_counts,
        )


LegacyRulePrototype = Callable[[list[str], list[Violation], RuleContext], None]
RulePrototype = Callable[[list[str], AnalysisContext], RuleResult]

# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

_SLOP_ADJECTIVES = [
    "crucial", "groundbreaking", "pivotal", "paramount", "seamless", "holistic",
    "multifaceted", "meticulous", "profound", "comprehensive", "invaluable",
    "notable", "noteworthy", "game-changing", "revolutionary", "pioneering",
    "visionary", "formidable", "quintessential", "unparalleled",
    "stunning", "breathtaking", "captivating", "nestled", "robust",
    "innovative", "cutting-edge", "impactful",
]

_SLOP_VERBS = [
    "delve", "delves", "delved", "delving", "embark", "embrace", "elevate",
    "foster", "harness", "unleash", "unlock", "orchestrate", "streamline",
    "transcend", "navigate", "underscore", "showcase", "leverage",
    "ensuring", "highlighting", "emphasizing", "reflecting",
]

_SLOP_NOUNS = [
    "landscape", "tapestry", "journey", "paradigm", "testament", "trajectory",
    "nexus", "symphony", "spectrum", "odyssey", "pinnacle", "realm", "intricacies",
]

_SLOP_HEDGE = [
    "notably", "importantly", "furthermore", "additionally", "particularly",
    "significantly", "interestingly", "remarkably", "surprisingly", "fascinatingly",
    "moreover", "however", "overall",
]

_ALL_SLOP_WORDS = _SLOP_ADJECTIVES + _SLOP_VERBS + _SLOP_NOUNS + _SLOP_HEDGE

_SLOP_WORD_RE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in _ALL_SLOP_WORDS) + r")\b",
    re.IGNORECASE,
)

_SLOP_PHRASES_LITERAL = [
    "it's worth noting", "it's important to note",
    "this is where things get interesting", "here's the thing",
    "at the end of the day", "in today's fast-paced",
    "as technology continues to", "something shifted", "everything changed",
    "the answer? it's simpler than you think", "what makes this work is",
    "this is exactly", "let's break this down", "let's dive in",
    "in this post, we'll explore", "in this article, we'll",
    "let me know if", "would you like me to", "i hope this helps",
    "as mentioned earlier", "as i mentioned", "without further ado",
    "on the other hand", "in addition", "in summary", "in conclusion",
    "you might be wondering", "the obvious question is",
    "no discussion would be complete", "great question", "that's a great",
    "if you want, i can", "i can adapt this", "i can make this",
    "here are some options", "here are a few options", "would you prefer",
    "shall i", "if you'd like, i can", "i can also",
    "in other words", "put differently", "that is to say",
    "to put it simply", "to put it another way", "what this means is",
    "the takeaway is", "the bottom line is", "the key takeaway",
    "the key insight",
]

_SLOP_PHRASES_RE_LIST: list[re.Pattern[str]] = [
    re.compile(re.escape(p), re.IGNORECASE) for p in _SLOP_PHRASES_LITERAL
]

_NOT_JUST_BUT_RE = re.compile(
    r"not (just|only) .{1,40}, but (also )?", re.IGNORECASE
)

_BOLD_HEADER_RE = re.compile(r"\*\*[^*]+[.:]\*\*\s+\S")
_BULLET_LINE_RE = re.compile(r"^(\s*[-*]\s|\s*\d+\.\s)")
_TRIADIC_RE = re.compile(r"\w+, \w+, and \w+", re.IGNORECASE)

_META_COMM_PATTERNS = [
    re.compile(r"would you like", re.IGNORECASE),
    re.compile(r"let me know if", re.IGNORECASE),
    re.compile(r"as mentioned", re.IGNORECASE),
    re.compile(r"i hope this", re.IGNORECASE),
    re.compile(r"feel free to", re.IGNORECASE),
    re.compile(r"don't hesitate to", re.IGNORECASE),
]

_FALSE_NARRATIVITY_PATTERNS = [
    re.compile(r"then something interesting happened", re.IGNORECASE),
    re.compile(r"this is where things get interesting", re.IGNORECASE),
    re.compile(r"that's when everything changed", re.IGNORECASE),
]

_SENTENCE_OPENER_PATTERNS = [
    re.compile(r"(?:^|[.!?]\s+)(certainly[,! ])", re.IGNORECASE | re.MULTILINE),
    re.compile(r"(?:^|[.!?]\s+)(absolutely[,! ])", re.IGNORECASE | re.MULTILINE),
]

_WEASEL_PATTERNS = [
    re.compile(r"\bsome critics argue\b", re.IGNORECASE),
    re.compile(r"\bmany believe\b", re.IGNORECASE),
    re.compile(r"\bexperts suggest\b", re.IGNORECASE),
    re.compile(r"\bstudies show\b", re.IGNORECASE),
    re.compile(r"\bsome argue\b", re.IGNORECASE),
    re.compile(r"\bit is widely believed\b", re.IGNORECASE),
    re.compile(r"\bresearch suggests\b", re.IGNORECASE),
]

_AI_DISCLOSURE_PATTERNS = [
    re.compile(r"\bas an ai\b", re.IGNORECASE),
    re.compile(r"\bas a language model\b", re.IGNORECASE),
    re.compile(r"\bi don't have personal\b", re.IGNORECASE),
    re.compile(r"\bi cannot browse\b", re.IGNORECASE),
    re.compile(r"\bup to my last training\b", re.IGNORECASE),
    re.compile(r"\bas of my (last |knowledge )?cutoff\b", re.IGNORECASE),
    re.compile(r"\bi'm just an? ai\b", re.IGNORECASE),
]

_PLACEHOLDER_RE = re.compile(
    r"\[insert [^\]]*\]|\[describe [^\]]*\]|\[url [^\]]*\]|\[your [^\]]*\]|\[todo[^\]]*\]",
    re.IGNORECASE,
)

_SENTENCE_SPLIT_RE = re.compile(r"[.!?][\"'\u201D\u2019)\]]*(?:\s|$)")
_EM_DASH_RE = re.compile(r"\u2014| -- ")
_CONTRAST_PAIR_RE = re.compile(r"\b(\w+), not (\w+)\b")

_SETUP_RESOLUTION_A_RE = re.compile(
    r"\b(this|that|these|those|it|they|we)\s+"
    r"(isn't|aren't|wasn't|weren't|doesn't|don't|didn't|hasn't|haven't|won't|can't|couldn't|shouldn't"
    r"|is\s+not|are\s+not|was\s+not|were\s+not|does\s+not|do\s+not|did\s+not"
    r"|has\s+not|have\s+not|will\s+not|cannot|could\s+not|should\s+not)\b"
    r".{0,80}[.;:,]\s*"
    r"(it's|they're|that's|he's|she's|we're|it\s+is|they\s+are|that\s+is|this\s+is"
    r"|these\s+are|those\s+are|he\s+is|she\s+is|we\s+are|what's|what\s+is"
    r"|the\s+real|the\s+actual|instead|rather)",
    re.IGNORECASE,
)

_SETUP_RESOLUTION_B_RE = re.compile(
    r"\b(it's|that's|this\s+is|they're|he's|she's|we're)\s+not\b"
    r".{0,80}[.;:,]\s*"
    r"(it's|they're|that's|he's|she's|we're|it\s+is|they\s+are|that\s+is|this\s+is"
    r"|these\s+are|those\s+are|what's|what\s+is|the\s+real|the\s+actual|instead|rather)",
    re.IGNORECASE,
)

_ELABORATION_COLON_RE = re.compile(r": [a-z]")
_FENCED_CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
_URL_COLON_RE = re.compile(r"https?:")
_MD_HEADER_LINE_RE = re.compile(r"^\s*#", re.MULTILINE)
_JSON_COLON_RE = re.compile(r': ["{\[\d]|: true|: false|: null')

_PITHY_PIVOT_RE = re.compile(r",\s+(?:but|yet|and|not|or)\b", re.IGNORECASE)
_BULLET_DENSITY_RE = re.compile(r"^\s*[-*]\s|^\s*\d+[.)]\s", re.MULTILINE)
_BLOCKQUOTE_LINE_RE = re.compile(r"^>", re.MULTILINE)
_BOLD_TERM_BULLET_RE = re.compile(r"^\s*[-*]\s+\*\*|^\s*\d+[.)]\s+\*\*")
_HORIZONTAL_RULE_RE = re.compile(r"^\s*(?:---+|\*\*\*+|___+)\s*$", re.MULTILINE)

_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "is", "it", "that", "this", "with", "as", "by", "from", "was", "were", "are",
    "be", "been", "has", "have", "had", "not", "no", "do", "does", "did", "will",
    "would", "could", "should", "can", "may", "might", "if", "then", "than", "so",
    "up", "out", "about", "into", "over", "after", "before", "between", "through",
    "just", "also", "very", "more", "most", "some", "any", "each", "every", "all",
    "both", "few", "other", "such", "only", "own", "same", "too", "how", "what",
    "which", "who", "when", "where", "why",
})

_PUNCT_STRIP_RE = re.compile(r"^[^\w]+|[^\w]+$")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _context_around(
    text: str, start: int, end: int, width: int | None = None,
    hyperparameters: Hyperparameters = HYPERPARAMETERS,
) -> str:
    if width is None:
        width = hyperparameters.context_window_chars
    mid = (start + end) // 2
    half = width // 2
    ctx_start = max(0, mid - half)
    ctx_end = min(len(text), mid + half)
    snippet = text[ctx_start:ctx_end].replace("\n", " ")
    prefix = "..." if ctx_start > 0 else ""
    suffix = "..." if ctx_end < len(text) else ""
    return f"{prefix}{snippet}{suffix}"


def _word_count(text: str) -> int:
    return len(text.split())


def _strip_code_blocks(text: str) -> str:
    return _FENCED_CODE_BLOCK_RE.sub("", text)


def _find_repeated_ngrams(text: str, hyperparameters: Hyperparameters) -> list[dict]:
    raw_tokens = text.split()
    tokens = [_PUNCT_STRIP_RE.sub("", t).lower() for t in raw_tokens]
    tokens = [t for t in tokens if t]

    min_n = hyperparameters.repeated_ngram_min_n
    max_n = hyperparameters.repeated_ngram_max_n

    if len(tokens) < min_n:
        return []

    ngram_counts: dict[tuple[str, ...], int] = {}
    for n in range(min_n, max_n + 1):
        for i in range(len(tokens) - n + 1):
            gram = tuple(tokens[i : i + n])
            ngram_counts[gram] = ngram_counts.get(gram, 0) + 1

    repeated = {
        gram: count
        for gram, count in ngram_counts.items()
        if count >= hyperparameters.repeated_ngram_min_count
        and not all(w in _STOPWORDS for w in gram)
    }

    if not repeated:
        return []

    to_remove: set[tuple[str, ...]] = set()
    sorted_grams = sorted(repeated.keys(), key=len, reverse=True)
    for i, longer in enumerate(sorted_grams):
        longer_str = " ".join(longer)
        for shorter in sorted_grams[i + 1 :]:
            if shorter in to_remove:
                continue
            shorter_str = " ".join(shorter)
            if shorter_str in longer_str and repeated[longer] >= repeated[shorter]:
                to_remove.add(shorter)

    results = []
    for gram in sorted(repeated.keys(), key=lambda g: (-len(g), -repeated[g])):
        if gram not in to_remove:
            results.append({
                "phrase": " ".join(gram),
                "count": repeated[gram],
                "n": len(gram),
            })

    return results


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def _initial_counts() -> dict[str, int]:
    return {
        "slop_words": 0, "slop_phrases": 0, "structural": 0, "tone": 0,
        "weasel": 0, "ai_disclosure": 0, "placeholder": 0, "rhythm": 0,
        "em_dash": 0, "contrast_pairs": 0, "colon_density": 0,
        "pithy_fragment": 0, "setup_resolution": 0, "bullet_density": 0,
        "blockquote_density": 0, "bold_bullet_list": 0, "horizontal_rules": 0,
        "phrase_reuse": 0,
    }


def _short_text_result(word_count: int, counts: dict[str, int], hp: Hyperparameters) -> dict:
    return {
        "score": hp.score_max, "band": "clean", "word_count": word_count,
        "violations": [], "counts": counts, "total_penalty": 0,
        "weighted_sum": 0.0, "density": 0.0, "advice": [],
    }


def _run_legacy_rule(legacy_rule, lines, context):
    scratch_violations: list[Violation] = []
    scratch_context = RuleContext(
        text=context.text, word_count=context.word_count,
        sentences=context.sentences, advice=[], counts=_initial_counts(),
        hyperparameters=context.hyperparameters,
    )
    legacy_rule(lines, scratch_violations, scratch_context)
    return RuleResult(
        violations=list(scratch_violations),
        advice=list(scratch_context.advice),
        count_deltas=scratch_context.counts,
    )


def _functionalize_rule(legacy_rule):
    def _rule(lines, context):
        return _run_legacy_rule(legacy_rule, lines, context)
    return _rule


def _run_analysis_pipeline(lines, context, pipeline):
    initial_state = AnalysisState.initial(_initial_counts())
    curried_rules = [partial(rule, lines, context) for rule in pipeline]

    def _merge_rule_result(state, curried_rule):
        result = curried_rule()
        return state.merge(
            violations=result.violations,
            advice=result.advice,
            count_deltas=result.count_deltas,
        )

    return reduce(_merge_rule_result, curried_rules, initial_state)


# ---------------------------------------------------------------------------
# Rule implementations
# ---------------------------------------------------------------------------

def _collect_slop_word_rule(lines, violations, context):
    text, advice, counts, hp = context.text, context.advice, context.counts, context.hyperparameters
    for m in _SLOP_WORD_RE.finditer(text):
        word = m.group(0)
        violations.append(Violation(rule="slop_word", match=word.lower(),
            context=_context_around(text, m.start(), m.end(), hyperparameters=hp),
            penalty=hp.slop_word_penalty))
        advice.append(f"Replace '{word.lower()}' \u2014 what specifically do you mean?")
        counts["slop_words"] += 1


def _collect_slop_phrase_rules(lines, violations, context):
    text, advice, counts, hp = context.text, context.advice, context.counts, context.hyperparameters
    for pat in _SLOP_PHRASES_RE_LIST:
        for m in pat.finditer(text):
            phrase = m.group(0)
            violations.append(Violation(rule="slop_phrase", match=phrase.lower(),
                context=_context_around(text, m.start(), m.end(), hyperparameters=hp),
                penalty=hp.slop_phrase_penalty))
            advice.append(f"Cut '{phrase.lower()}' \u2014 just state the point directly.")
            counts["slop_phrases"] += 1
    for m in _NOT_JUST_BUT_RE.finditer(text):
        phrase = m.group(0)
        violations.append(Violation(rule="slop_phrase", match=phrase.strip().lower(),
            context=_context_around(text, m.start(), m.end(), hyperparameters=hp),
            penalty=hp.slop_phrase_penalty))
        advice.append(f"Cut '{phrase.strip().lower()}' \u2014 just state the point directly.")
        counts["slop_phrases"] += 1


def _collect_structural_patterns(lines, violations, context):
    text, advice, counts, hp = context.text, context.advice, context.counts, context.hyperparameters
    bold_matches = list(_BOLD_HEADER_RE.finditer(text))
    if len(bold_matches) >= hp.structural_bold_header_min:
        violations.append(Violation(rule="structural", match="bold_header_explanation",
            context=f"Found {len(bold_matches)} instances of **Bold.** pattern",
            penalty=hp.structural_bold_header_penalty))
        advice.append(f"Vary paragraph structure \u2014 {len(bold_matches)} bold-header-explanation blocks.")
        counts["structural"] += 1
    run_length = 0
    for line in lines:
        if _BULLET_LINE_RE.match(line):
            run_length += 1
        else:
            if run_length >= hp.structural_bullet_run_min:
                violations.append(Violation(rule="structural", match="excessive_bullets",
                    context=f"Run of {run_length} consecutive bullet lines",
                    penalty=hp.structural_bullet_run_penalty))
                advice.append(f"Consider prose instead of this {run_length}-item bullet list.")
                counts["structural"] += 1
            run_length = 0
    if run_length >= hp.structural_bullet_run_min:
        violations.append(Violation(rule="structural", match="excessive_bullets",
            context=f"Run of {run_length} consecutive bullet lines",
            penalty=hp.structural_bullet_run_penalty))
        counts["structural"] += 1
    triadic_matches = list(_TRIADIC_RE.finditer(text))
    for m in triadic_matches[:hp.triadic_record_cap]:
        violations.append(Violation(rule="structural", match="triadic",
            context=_context_around(text, m.start(), m.end(), hyperparameters=hp),
            penalty=hp.triadic_penalty))
        counts["structural"] += 1
    if len(triadic_matches) >= hp.triadic_advice_min:
        advice.append(f"{len(triadic_matches)} triadic structures ('X, Y, and Z') \u2014 vary your list cadence.")


def _collect_tone_marker_rules(lines, violations, context):
    text, advice, counts, hp = context.text, context.advice, context.counts, context.hyperparameters
    for pat in _META_COMM_PATTERNS:
        for m in pat.finditer(text):
            phrase = m.group(0)
            violations.append(Violation(rule="tone", match=phrase.lower(),
                context=_context_around(text, m.start(), m.end(), hyperparameters=hp),
                penalty=hp.tone_penalty))
            advice.append(f"Remove '{phrase.lower()}' \u2014 this is a direct AI tell.")
            counts["tone"] += 1
    for pat in _FALSE_NARRATIVITY_PATTERNS:
        for m in pat.finditer(text):
            phrase = m.group(0)
            violations.append(Violation(rule="tone", match=phrase.lower(),
                context=_context_around(text, m.start(), m.end(), hyperparameters=hp),
                penalty=hp.tone_penalty))
            counts["tone"] += 1
    for pat in _SENTENCE_OPENER_PATTERNS:
        for m in pat.finditer(text):
            word = m.group(1).strip(" ,!")
            violations.append(Violation(rule="tone", match=word.lower(),
                context=_context_around(text, m.start(), m.end(), hyperparameters=hp),
                penalty=hp.sentence_opener_penalty))
            counts["tone"] += 1


def _collect_weasel_phrase_rules(lines, violations, context):
    text, advice, counts, hp = context.text, context.advice, context.counts, context.hyperparameters
    for pat in _WEASEL_PATTERNS:
        for m in pat.finditer(text):
            phrase = m.group(0)
            violations.append(Violation(rule="weasel", match=phrase.lower(),
                context=_context_around(text, m.start(), m.end(), hyperparameters=hp),
                penalty=hp.weasel_penalty))
            counts["weasel"] += 1


def _collect_ai_disclosure_rules(lines, violations, context):
    text, advice, counts, hp = context.text, context.advice, context.counts, context.hyperparameters
    for pat in _AI_DISCLOSURE_PATTERNS:
        for m in pat.finditer(text):
            phrase = m.group(0)
            violations.append(Violation(rule="ai_disclosure", match=phrase.lower(),
                context=_context_around(text, m.start(), m.end(), hyperparameters=hp),
                penalty=hp.ai_disclosure_penalty))
            counts["ai_disclosure"] += 1


def _collect_placeholder_rules(lines, violations, context):
    text, advice, counts, hp = context.text, context.advice, context.counts, context.hyperparameters
    for m in _PLACEHOLDER_RE.finditer(text):
        match_text = m.group(0)
        violations.append(Violation(rule="placeholder", match=match_text.lower(),
            context=_context_around(text, m.start(), m.end(), hyperparameters=hp),
            penalty=hp.placeholder_penalty))
        counts["placeholder"] += 1


def _collect_rhythm_rule(lines, violations, context):
    sentences, counts, hp = context.sentences, context.counts, context.hyperparameters
    if len(sentences) < hp.rhythm_min_sentences:
        return
    lengths = [len(s.split()) for s in sentences]
    mean = sum(lengths) / len(lengths)
    if mean <= 0:
        return
    variance = sum((x - mean) ** 2 for x in lengths) / len(lengths)
    std = math.sqrt(variance)
    cv = std / mean
    if cv < hp.rhythm_cv_threshold:
        violations.append(Violation(rule="rhythm", match="monotonous_rhythm",
            context=f"CV={cv:.2f} across {len(sentences)} sentences (mean {mean:.1f} words)",
            penalty=hp.rhythm_penalty))
        counts["rhythm"] += 1


def _collect_em_dash_density_rule(lines, violations, context):
    text, counts, hp = context.text, context.counts, context.hyperparameters
    wc = context.word_count
    if wc <= 0:
        return
    em_dash_count = len(list(_EM_DASH_RE.finditer(text)))
    ratio = (em_dash_count / wc) * hp.em_dash_words_basis
    if ratio > hp.em_dash_density_threshold:
        violations.append(Violation(rule="em_dash", match="em_dash_density",
            context=f"{em_dash_count} em dashes in {wc} words ({ratio:.1f} per 150 words)",
            penalty=hp.em_dash_penalty))
        counts["em_dash"] += 1


def _collect_contrast_pair_rule(lines, violations, context):
    text, advice, counts, hp = context.text, context.advice, context.counts, context.hyperparameters
    contrast_matches = list(_CONTRAST_PAIR_RE.finditer(text))
    for m in contrast_matches[:hp.contrast_record_cap]:
        matched = m.group(0)
        violations.append(Violation(rule="contrast_pair", match=matched,
            context=_context_around(text, m.start(), m.end(), hyperparameters=hp),
            penalty=hp.contrast_penalty))
        counts["contrast_pairs"] += 1
    if len(contrast_matches) >= hp.contrast_advice_min:
        advice.append(f"{len(contrast_matches)} 'X, not Y' contrasts \u2014 Claude rhetorical tic.")


def _collect_setup_resolution_rule(lines, violations, context):
    text, advice, counts, hp = context.text, context.advice, context.counts, context.hyperparameters
    recorded = 0
    for pat in (_SETUP_RESOLUTION_A_RE, _SETUP_RESOLUTION_B_RE):
        for m in pat.finditer(text):
            if recorded < hp.setup_resolution_record_cap:
                matched = m.group(0)
                violations.append(Violation(rule="setup_resolution", match=matched,
                    context=_context_around(text, m.start(), m.end(), hyperparameters=hp),
                    penalty=hp.setup_resolution_penalty))
                recorded += 1
            counts["setup_resolution"] += 1


def _collect_colon_density_rule(lines, violations, context):
    text, counts, hp = context.text, context.counts, context.hyperparameters
    stripped_text = _strip_code_blocks(text)
    colon_count = 0
    for line in stripped_text.split("\n"):
        if _MD_HEADER_LINE_RE.match(line):
            continue
        for cm in _ELABORATION_COLON_RE.finditer(line):
            col_pos = cm.start()
            before = line[:col_pos + 1]
            if before.endswith("http:") or before.endswith("https:"):
                continue
            snippet = line[col_pos:col_pos + 10]
            if _JSON_COLON_RE.match(snippet):
                continue
            colon_count += 1
    swc = _word_count(stripped_text)
    if swc <= 0:
        return
    ratio = (colon_count / swc) * hp.colon_words_basis
    if ratio > hp.colon_density_threshold:
        violations.append(Violation(rule="colon_density", match="colon_density",
            context=f"{colon_count} elaboration colons in {swc} words ({ratio:.1f} per 150 words)",
            penalty=hp.colon_density_penalty))
        counts["colon_density"] += 1


def _collect_pithy_fragment_rule(lines, violations, context):
    sentences, counts, hp = context.sentences, context.counts, context.hyperparameters
    pithy_count = 0
    for sent in sentences:
        sent_stripped = sent.strip()
        if not sent_stripped:
            continue
        if len(sent_stripped.split()) > hp.pithy_max_sentence_words:
            continue
        if _PITHY_PIVOT_RE.search(sent_stripped):
            if pithy_count < hp.pithy_record_cap:
                violations.append(Violation(rule="pithy_fragment", match=sent_stripped,
                    context=sent_stripped, penalty=hp.pithy_penalty))
            pithy_count += 1
            counts["pithy_fragment"] += 1


def _collect_bullet_density_rule(lines, violations, context):
    counts, hp = context.counts, context.hyperparameters
    non_empty = [line for line in lines if line.strip()]
    total = len(non_empty)
    if total <= 0:
        return
    bullet_count = sum(1 for line in non_empty if _BULLET_DENSITY_RE.match(line))
    ratio = bullet_count / total
    if ratio > hp.bullet_density_threshold:
        violations.append(Violation(rule="structural", match="bullet_density",
            context=f"{bullet_count} of {total} non-empty lines are bullets ({ratio:.0%})",
            penalty=hp.bullet_density_penalty))
        counts["bullet_density"] += 1


def _collect_blockquote_density_rule(lines, violations, context):
    counts, hp = context.counts, context.hyperparameters
    in_code = False
    bq_count = 0
    for line in lines:
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if not in_code and line.startswith(">"):
            bq_count += 1
    if bq_count >= hp.blockquote_min_lines:
        excess = bq_count - hp.blockquote_free_lines
        capped = min(excess, hp.blockquote_cap)
        bq_penalty = hp.blockquote_penalty_step * capped
        violations.append(Violation(rule="structural", match="blockquote_density",
            context=f"{bq_count} blockquote lines", penalty=bq_penalty))
        counts["blockquote_density"] += 1


def _collect_bold_term_bullet_run_rule(lines, violations, context):
    counts, hp = context.counts, context.hyperparameters
    run = 0
    for line in lines:
        if _BOLD_TERM_BULLET_RE.match(line):
            run += 1
            continue
        if run >= hp.bold_bullet_run_min:
            violations.append(Violation(rule="structural", match="bold_bullet_list",
                context=f"Run of {run} bold-term bullets", penalty=hp.bold_bullet_run_penalty))
            counts["bold_bullet_list"] += 1
        run = 0
    if run >= hp.bold_bullet_run_min:
        violations.append(Violation(rule="structural", match="bold_bullet_list",
            context=f"Run of {run} bold-term bullets", penalty=hp.bold_bullet_run_penalty))
        counts["bold_bullet_list"] += 1


def _collect_horizontal_rule_overuse_rule(lines, violations, context):
    text, counts, hp = context.text, context.counts, context.hyperparameters
    hr_count = len(_HORIZONTAL_RULE_RE.findall(text))
    if hr_count >= hp.horizontal_rule_min:
        violations.append(Violation(rule="structural", match="horizontal_rules",
            context=f"{hr_count} horizontal rules", penalty=hp.horizontal_rule_penalty))
        counts["horizontal_rules"] += 1


def _collect_phrase_reuse_rule(lines, violations, context):
    text, counts, hp = context.text, context.counts, context.hyperparameters
    repeated_ngrams = _find_repeated_ngrams(text, hp)
    recorded = 0
    for ng in repeated_ngrams:
        if recorded >= hp.phrase_reuse_record_cap:
            break
        violations.append(Violation(rule="phrase_reuse", match=ng["phrase"],
            context=f"'{ng['phrase']}' ({ng['n']}-word phrase) appears {ng['count']} times",
            penalty=hp.phrase_reuse_penalty))
        counts["phrase_reuse"] += 1
        recorded += 1


# Functionalize all rules
_apply_slop_word_rule = _functionalize_rule(_collect_slop_word_rule)
_apply_slop_phrase_rules = _functionalize_rule(_collect_slop_phrase_rules)
_apply_structural_patterns = _functionalize_rule(_collect_structural_patterns)
_apply_tone_marker_rules = _functionalize_rule(_collect_tone_marker_rules)
_apply_weasel_phrase_rules = _functionalize_rule(_collect_weasel_phrase_rules)
_apply_ai_disclosure_rules = _functionalize_rule(_collect_ai_disclosure_rules)
_apply_placeholder_rules = _functionalize_rule(_collect_placeholder_rules)
_apply_rhythm_rule = _functionalize_rule(_collect_rhythm_rule)
_apply_em_dash_density_rule = _functionalize_rule(_collect_em_dash_density_rule)
_apply_contrast_pair_rule = _functionalize_rule(_collect_contrast_pair_rule)
_apply_setup_resolution_rule = _functionalize_rule(_collect_setup_resolution_rule)
_apply_colon_density_rule = _functionalize_rule(_collect_colon_density_rule)
_apply_pithy_fragment_rule = _functionalize_rule(_collect_pithy_fragment_rule)
_apply_bullet_density_rule = _functionalize_rule(_collect_bullet_density_rule)
_apply_blockquote_density_rule = _functionalize_rule(_collect_blockquote_density_rule)
_apply_bold_term_bullet_run_rule = _functionalize_rule(_collect_bold_term_bullet_run_rule)
_apply_horizontal_rule_overuse_rule = _functionalize_rule(_collect_horizontal_rule_overuse_rule)
_apply_phrase_reuse_rule = _functionalize_rule(_collect_phrase_reuse_rule)


def _compute_weighted_sum(violations, counts, hp):
    weighted_sum = 0.0
    for violation in violations:
        rule = violation.rule
        penalty = abs(violation.penalty)
        cat_count = counts.get(rule, 0) or counts.get(rule + "s", 0)
        count_key = (
            rule if rule in hp.claude_categories
            else (rule + "s" if (rule + "s") in hp.claude_categories else None)
        )
        if count_key and count_key in hp.claude_categories and cat_count > 1:
            weight = penalty * (1 + hp.concentration_alpha * (cat_count - 1))
        else:
            weight = penalty
        weighted_sum += weight
    return weighted_sum


def _band_for_score(score, hp):
    if score >= hp.band_clean_min:
        return "clean"
    if score >= hp.band_light_min:
        return "light"
    if score >= hp.band_moderate_min:
        return "moderate"
    if score >= hp.band_heavy_min:
        return "heavy"
    return "saturated"


def _deduplicate_advice(advice):
    seen = set()
    unique = []
    for item in advice:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze(text: str, hyperparameters: Hyperparameters = HYPERPARAMETERS) -> dict:
    """Run all slop checks and return score, diagnostics, and advice.

    Returns a dict with:
        score: int 0-100 (higher = cleaner, less sloppy)
        band: str ("clean", "light", "moderate", "heavy", "saturated")
        word_count: int
        violations: list of violation dicts
        counts: dict of per-rule hit counts
        total_penalty: int (sum of raw penalties)
        weighted_sum: float (concentration-adjusted penalty total)
        density: float (weighted_sum normalized by text length)
        advice: list of actionable suggestions
    """
    word_count = _word_count(text)
    counts = _initial_counts()

    if word_count < hyperparameters.short_text_word_count:
        return _short_text_result(word_count, counts, hyperparameters)

    lines = text.split("\n")
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    context = AnalysisContext(
        text=text, word_count=word_count, sentences=sentences,
        hyperparameters=hyperparameters,
    )
    pipeline = [
        _apply_slop_word_rule, _apply_slop_phrase_rules,
        _apply_structural_patterns, _apply_tone_marker_rules,
        _apply_weasel_phrase_rules, _apply_ai_disclosure_rules,
        _apply_placeholder_rules, _apply_rhythm_rule,
        _apply_em_dash_density_rule, _apply_contrast_pair_rule,
        _apply_setup_resolution_rule, _apply_colon_density_rule,
        _apply_pithy_fragment_rule, _apply_bullet_density_rule,
        _apply_blockquote_density_rule, _apply_bold_term_bullet_run_rule,
        _apply_horizontal_rule_overuse_rule, _apply_phrase_reuse_rule,
    ]
    state = _run_analysis_pipeline(lines, context, pipeline)

    total_penalty = sum(v.penalty for v in state.violations)
    weighted_sum = _compute_weighted_sum(list(state.violations), state.counts, hyperparameters)
    density = (
        weighted_sum / (word_count / hyperparameters.density_words_basis)
        if word_count > 0 else 0.0
    )
    raw_score = hyperparameters.score_max * math.exp(
        -hyperparameters.decay_lambda * density
    )
    score = max(hyperparameters.score_min, min(hyperparameters.score_max, round(raw_score)))
    band = _band_for_score(score, hyperparameters)

    return {
        "score": score,
        "band": band,
        "word_count": word_count,
        "violations": [v.to_payload() for v in state.violations],
        "counts": state.counts,
        "total_penalty": total_penalty,
        "weighted_sum": round(weighted_sum, 2),
        "density": round(density, 2),
        "advice": _deduplicate_advice(list(state.advice)),
    }
