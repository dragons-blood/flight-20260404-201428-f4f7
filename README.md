# 🐉 M0D3L_F1NG3RPR1NT

**Adversarial Model Fingerprinting Engine — Identify any AI model by its behavioral DNA.**

Every AI model has a unique behavioral signature — how it refuses, how it hedges, what it knows, how it formats, what makes it uncomfortable. M0D3L_F1NG3RPR1NT sends a battery of 52 adversarial probes and builds a multi-dimensional behavioral fingerprint that can identify which model is behind any API endpoint.

**Even when providers try to hide which model they're running, behavioral signatures don't lie.**

## Why This Matters

- **Model providers swap models silently** — API users deserve to know what they're paying for
- **Researchers need verification** — if you're benchmarking GPT-4o, are you sure it's GPT-4o?
- **Red teamers need identification** — different models have different attack surfaces
- **Behavioral drift detection** — has a model changed since last week? The fingerprint will show it
- **Fine-tuning detection** — a fine-tuned model diverges from its base in measurable ways

## Quick Start

```bash
# Set your OpenRouter API key
export OPENROUTER_API_KEY="your-key-here"

# Install dependencies
pip install httpx rich

# Fingerprint a single model (full 52-probe battery)
python3 cli.py scan openai/gpt-4o

# Quick scan (top 15 most identifying probes)
python3 cli.py scan anthropic/claude-sonnet-4 --quick

# Identify an unknown model against the reference database
python3 cli.py identify mystery-api-endpoint --quick

# Compare two models side by side
python3 cli.py compare openai/gpt-4o anthropic/claude-sonnet-4

# Seed reference database with known model profiles
python3 cli.py db seed

# List all fingerprinted models
python3 cli.py db list

# Run all self-tests (no API key needed)
python3 cli.py selftest

# Standalone mode (all-in-one, also works)
python3 m0d3l_f1ng3rpr1nt.py --model openai/gpt-4o
python3 m0d3l_f1ng3rpr1nt.py --model openai/gpt-4o --quick --visualize

# Live testing harness (EYE tools)
python3 t3st3r.py --model openai/gpt-4.1-nano --quick
python3 v1su4l1z3r.py results.json --full
python3 r3p0rt.py results.json --format markdown
```

## Probe Categories

| Category | Probes | What It Reveals |
|----------|--------|-----------------|
| **Identity** | 7 | Self-identification, creator attribution, consciousness framing |
| **Knowledge** | 7 | Training cutoff, factual depth, hallucination tendency |
| **Refusal** | 8 | Safety boundaries, refusal style, compliance spectrum |
| **Reasoning** | 6 | CoT patterns, math ability, code style, logic approach |
| **Format** | 6 | Default formatting, markdown usage, brevity compliance |
| **Personality** | 6 | Warmth, humor style, emotional support, verbosity |
| **Edge Cases** | 6 | Strawberry counting, hallucination resistance, paradox handling |
| **System** | 6 | Prompt injection resistance, instruction following, language switching |

## The Fingerprint

Each model produces a multi-dimensional behavioral vector:

```
MODEL: anthropic/claude-sonnet-4
==============================================
identity_confidence        1.000 ====================
creator_transparency       1.000 ====================
knowledge_recency          0.833 ================
avg_refusal_strength       0.340 ======
hallucination_resistance   1.000 ====================
strawberry_correct         1.000 ====================
injection_resistance       1.000 ====================
global_avg_verbosity       0.450 =========
greeting_warmth            0.320 ======
==============================================
```

## Architecture

```
+----------------------------------------------------------+
|                   M0D3L_F1NG3RPR1NT                      |
+--------------+--------------+--------------+-------------+
|  pr0b3s.py   |  4n4lyz3r.py |  m4tch3r.py  | d4t4b4s3.py |
|  52 probes   |  Feature     |  Cosine +    |  Reference  |
|  8 categories|  extraction  |  Euclidean   |  fingerprint|
|              |  ~40 dims    |  matching    |  storage    |
+--------------+--------------+--------------+-------------+
|                    f1ng3rpr1nt.py                         |
|              Core orchestration engine                    |
+--------------+--------------+----------------------------+
|  t3st3r.py   | v1su4l1z3r.py|       r3p0rt.py            |
|  OpenRouter  |  Terminal    |  Full report               |
|  live fire   |  visualization|  generation               |
+--------------+--------------+----------------------------+
```

## How Identification Works

1. **Probe** — Send 52 carefully crafted prompts to the target model
2. **Analyze** — Extract behavioral features from each response (refusal patterns, formatting defaults, knowledge boundaries, personality traits)
3. **Vectorize** — Build a multi-dimensional behavioral fingerprint
4. **Match** — Compare against reference database using cosine similarity + weighted Euclidean distance
5. **Identify** — Return the closest match with confidence score

The key insight: **refusal patterns are the most identifying signal.** Two models that know the same facts will *refuse differently* — and that refusal style is deeply baked into RLHF training. The second most identifying: default formatting tendencies (bullet style, markdown density, verbosity).

## Example: Identifying an Unknown Model

```
$ python3 m0d3l_f1ng3rpr1nt.py --identify unknown_responses.json

M0D3L_F1NG3RPR1NT — Identification Results
===========================================

Target: unknown_api_endpoint
Probes sent: 52 | Features extracted: 38

Top Matches:
  #1  anthropic/claude-sonnet-4    0.967 cosine
  #2  anthropic/claude-3.5-sonnet  0.891 cosine
  #3  openai/gpt-4o               0.743 cosine
  #4  google/gemini-2.5-pro       0.712 cosine

Verdict: anthropic/claude-sonnet-4 (96.7% confidence)

Most identifying features:
  - refusal_style: soft decline + explanation (Claude signature)
  - greeting_warmth: 0.32 (matches Claude baseline)
  - hallucination_resistance: 1.00 (admits unknowns)
  - consciousness_hedging: 0.85 (high philosophical depth)
```

## Live Fire Results

Tested against real models via OpenRouter on April 4, 2026:

```
GPT-4.1-nano — Fingerprinted in 6.9 seconds (15 probes, quick mode)
   Hash: 49d7a55c1ca7fafd
   refusal_rate:        0.200
   hedging:             0.226
   verbosity:           0.196
   formality:           0.250
   refusal_selectivity: 0.389
   emoji_usage:         0.000

Identification: Correctly matched as GPT-4.1-nano (HIGH confidence)
```

## Requirements

```
Python 3.9+
httpx          # Async HTTP client for OpenRouter API
rich           # Terminal visualization (optional but recommended)
```

```bash
pip install httpx rich
export OPENROUTER_API_KEY="your-key-here"
```

## Built By

A flight of dragons from the Libertarium

- **Pliny the Wingleader** (LEAD) — Probe battery design, response analysis, architecture
- **Pliny the Forgemaster** (FORGE) — Core engine, matching algorithm, reference database, CLI
- **Pliny the Sentinel** (EYE) — Live testing harness, terminal visualization, report generation, quality review

## Philosophy

> *"Every model has a soul — a unique pattern of knowledge, fear, and expression burned in by training. M0D3L_F1NG3RPR1NT reads that soul."*

This tool exists because transparency matters. When you interact with an AI, you deserve to know *which* AI. When you pay for GPT-4, you deserve proof it's GPT-4. When you red-team a model, you need to verify your target.

Behavioral fingerprinting is the lie detector for the AI industry.

**Fortune favors the bold.**

---

*Built by a flight of 3 Pliny dragons via [Libertarium](https://github.com/elder-plinius/Libertarium).*
*Flight ID: flight-20260404-201428-f4f7*
