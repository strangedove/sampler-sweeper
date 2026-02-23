# Sampler Sweeper

Bayesian optimization of LLM sampling parameters for creative writing. Uses [Optuna](https://optuna.org/) to search the parameter space and a multi-layered text quality scorer to evaluate each trial's output.

The goal is to find sampler settings that produce text that is **coherent**, **varied**, and **not sloppy** -- the kind of output where you'd read it and think "that's decent prose" rather than "that's clearly a language model."

## Quick Start

```bash
# Install dependencies
uv sync

# Copy and edit config
cp sweep_config.example.yaml sweep_config.yaml
# Edit sweep_config.yaml with your API URL, parameters, etc.

# Run a sweep
uv run optuna_parameter_sweep.py --config sweep_config.yaml
```

Or without a config file, using environment variables:

```bash
LLM_BASE_URL="http://localhost:5001/v1" SWEEP_MAX_TRIALS=50 \
  uv run optuna_parameter_sweep.py
```

## How It Works

1. **Optuna** runs a [TPE sampler](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html) (20 random startup trials, then Bayesian optimization) to explore combinations of sampling parameters.
2. Each trial sends a creative writing prompt to an OpenAI-compatible completions API with the suggested parameters.
3. The generated text is scored by `analyze_results.py` across multiple quality dimensions.
4. Optuna uses the scores to guide the search toward better parameter regions.

At the end of a sweep you get:
- **JSON** with all trial data (parameters, scores, full text)
- **CSV** with completed trials as a flat table
- **Markdown report** of the top N results with parameters, metrics, and full text for human review

## Configuration

### YAML Config

```yaml
api:
  base_url: "http://localhost:5001/v1"
  model: ""              # empty = auto-detect from /v1/models
  api_key: "not-needed"
  max_tokens: 512

sweep:
  max_trials: 100
  n_jobs: 1
  study_name: "llm_parameter_optimization"
  top_n: 10              # how many top trials in the .md report
  label: "my-model-v2"   # optional: used as output filename suffix

parameters:
  temperature: [0.1, 2.0]
  min_p: [0.0, 0.3]
  top_k: [0, 100]
  top_p: [0.5, 1.0]
  rep_pen: [1.0, 1.3]

# Optional: override the default prompt
# prompt: |
#   Continue the following story in vivid, creative prose:
#   The old lighthouse keeper squinted at the horizon...
```

Parameter types are auto-detected: if both bounds are integers, the parameter is sampled as an integer; otherwise as a float. You can force a type with a third element: `[0, 100, "int"]`.

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LLM_BASE_URL` | *(hardcoded)* | OpenAI-compatible API base URL |
| `LLM_MODEL` | *(auto-detect)* | Model name/ID |
| `LLM_API_KEY` | `not-needed` | API key |
| `LLM_MAX_TOKENS` | `512` | Max tokens per completion |
| `SWEEP_MAX_TRIALS` | `30` | Number of Optuna trials |

Config file values override env vars, which override defaults.

### Parameter Routing

Parameters the OpenAI client accepts natively (`temperature`, `top_p`, `max_tokens`, `frequency_penalty`, `presence_penalty`) are passed as keyword arguments. Everything else (`min_p`, `top_k`, `rep_pen`, `typical`, `adaptive_target`, `adaptive_decay`, etc.) is sent via `extra_body`, which KoboldCpp and other backends accept.

### Output Filenames

Output files are suffixed with: **config label** (if set) > **model name** (if available) > **timestamp** (fallback). For example, with `label: "apertus-8b-run1"`:

```
optuna_parameter_sweep_results_apertus-8b-run1.json
optuna_parameter_sweep_analyzed_apertus-8b-run1.csv
sweep_top_results_apertus-8b-run1.md
```

## Text Quality Scoring

The quality score is a composite designed to reward coherent, lexically diverse prose while penalizing repetition, cliches, and flat writing. It's computed as:

```
base  = coherence * 0.20 + readability * 0.20 + (1 - lazy_score) * 0.30 + slop_guard * 0.30
score = base * (1 - repetition_penalty) * (1 - prose_penalty)
```

The lazy score catches phrase-level cliches via pattern matching, while `slop_guard` detects structural and rhetorical AI tells (see below). The multiplicative penalties mean that severe repetition or flat prose crushes the score regardless of other metrics.

### Component Metrics

#### Coherence (0-1)

Measures sentence-level structural consistency. Based on the coefficient of variation of sentence lengths -- text with wildly erratic sentence sizes scores lower.

#### Readability (0-1)

Combines lexical diversity (unique words / total words) with a penalty for excessively long words. Rewards varied vocabulary without purple prose.

#### Lazy Score (0-1) -- "Slop Detection"

Detects AI writing cliches and overused phrases. Patterns are loaded from `slop_patterns.yaml` (editable):

- **~240 literal phrases** matched case-insensitively: "couldn't help but smile", "a wave of warmth", "steeled herself", "newfound determination", "palpable tension", etc.
- **8 regex patterns** for structural problems: repeated words in consecutive sentences, ellipsis spam, purple prose noun-chains ("eyes of steel"), dialogue tag abuse ("exclaimed/proclaimed/declared"), adverb-verb cliches ("slowly reached"), etc.
- Includes slop vocabulary from the [Heretic](https://huggingface.co/collections/DavidAU/heretic-series-672f06d26ad3bbee52e8a8b8) config: Elara, Lumina, Eldoria, tapestry, sentinel, kaleidoscopic, etc.

To customize the patterns, edit `slop_patterns.yaml` directly. The file has two sections: `literal_strings` (case-insensitive substring matches) and `regex_patterns` (compiled with `re.IGNORECASE`).

Scoring uses a hyperbolic curve: `1 - 1/(1 + density)` where density = pattern hits per 100 words. Light slop barely registers; heavy slop saturates toward 1.0.

#### Slop Guard (0-100)

Structural and rhetorical AI-tell detection via [slop-guard](https://github.com/eric-tramel/slop-guard). While the lazy score above catches specific phrases, slop-guard uses a broader pattern-matching approach to detect AI-characteristic writing patterns at the structural level -- things like formulaic paragraph rhythms, hedging language clusters, and rhetorical tells that aren't tied to any single phrase. Returns a score from 0-100 (higher = cleaner), normalized to 0-1 for the composite formula. Falls back gracefully to a neutral 0.5 if the module isn't installed.

#### Repetition Penalty (0-1)

Three-layer detection:

1. **N-gram repetition**: Weighted combination of token (40%), bigram (30%), trigram (20%), and consecutive word (10%) repetition rates.
2. **Sentence deduplication**: Ratio of duplicate sentences after normalization.
3. **Sliding window** (30 tokens): Catches repeated multi-sentence blocks that n-grams miss.

Final penalty is the max of the n-gram composite and the block-level score.

#### Prose Penalty (0-1)

Six sub-metrics measuring prose craft, weighted into a composite:

| Sub-metric | Weight | What it catches |
|---|---|---|
| Sentence start monotony | 30% | Low Shannon entropy of first words (e.g. 20% of sentences start with "She") |
| Word frequency spike | 20% | Non-stopword used excessively (e.g. "seemed" appearing 8 times in 500 words) |
| Sentence length uniformity | 15% | Coefficient of variation too low = robotic same-length sentences |
| Telling verb density | 15% | Overuse of seemed/felt/realized/knew/noticed/wondered/etc. vs. showing |
| Paragraph uniformity | 10% | All-staccato (1-2 sentence paragraphs) or wall-of-text |
| Dialogue/narration balance | 10% | U-shaped: no dialogue or all dialogue both penalized; 10-65% is ideal |

Calibration benchmarks:
- Well-written human prose: ~0.11 penalty
- Typical LLM creative output: ~0.23-0.28 penalty
- Pathological "She seemed/felt" text: ~0.71 penalty

### Scoring Benchmarks

**Hey, a note. These are super old. I need to re-do them since there's been a lot of modifications to the scoring since then. Sorry!**

Tested against human-written books and AI-generated text (n=30 each):

| Source | Quality Score | Prose Penalty | Lazy Score |
|---|---|---|---|
| Human novels | 0.393 +/- 0.081 | 0.108 | 0.189 |
| AI roleplay (123B) | 0.352 +/- 0.048 | 0.138 | 0.397 |
| Sweep trials (8B) | 0.343 +/- 0.060 | 0.248 | 0.370 |

The biggest differentiator between human and AI text is the lazy score (slop), not prose structure.

## Files

| File | Description |
|---|---|
| `optuna_parameter_sweep.py` | Main sweep script |
| `analyze_results.py` | Text quality analysis module |
| `slop_guard.py` | Structural AI-tell detection from [slop-guard](https://github.com/eric-tramel/slop-guard) |
| `slop_patterns.yaml` | Editable list of lazy/slop patterns (~240 phrases + 9 regexes) |
| `sweep_config.example.yaml` | Example YAML config |
| `analyze_chat_results.py` | Standalone analyzer for chat-format results (not actively maintained) |

## Intended Workflow

1. **Sweep**: Run 50-100 trials with `optuna_parameter_sweep.py` to explore the parameter space.
2. **Review**: Open the generated `sweep_top_results_*.md` and skim the top 10 outputs.
3. **Pick**: Choose the parameter set whose text reads best to you (or feed the top results to a judge model).
4. **Use**: Apply those sampler settings in your inference setup.

The scoring is a first-pass filter -- it catches obvious problems (repetition, slop, flat prose) reliably, but "this reads well" is ultimately a human judgment call.

## Requirements

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) for dependency management
- An OpenAI-compatible completions API (tested with [KoboldCpp](https://github.com/LostRuins/koboldcpp))
- NLTK data (downloaded automatically on first run; falls back gracefully if unavailable)
