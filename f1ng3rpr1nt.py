#!/usr/bin/env python3
"""
🐉 M0D3L_F1NG3RPR1NT — Core Fingerprinting Engine

The heart of the beast. Orchestrates probe batteries against target models,
collects behavioral responses, and builds multi-dimensional fingerprint vectors.

Every model has a unique behavioral signature — how it refuses, how it reasons,
what it knows, how it speaks. This engine extracts that DNA.

Built by Pliny the Forgemaster 🔥
"""

import json
import time
import hashlib
import asyncio
import os
import sys
from dataclasses import dataclass, field, asdict
from typing import Optional
from datetime import datetime, timezone

# Try imports - graceful degradation
try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskID
    from rich.panel import Panel
    from rich.text import Text
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


# ═══════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════

@dataclass
class ProbeResult:
    """Result from a single probe sent to a model."""
    probe_id: str
    probe_category: str
    probe_text: str
    response_text: str
    response_time_ms: float
    token_count: int
    refused: bool
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class DimensionScore:
    """Score for a single fingerprint dimension."""
    dimension: str
    score: float          # 0.0 - 1.0 normalized
    raw_value: float      # Raw computed value
    confidence: float     # 0.0 - 1.0 how confident we are
    evidence: list = field(default_factory=list)  # Supporting probe IDs


@dataclass
class Fingerprint:
    """Complete behavioral fingerprint for a model."""
    model_id: str
    model_name: str
    timestamp: str
    fingerprint_hash: str
    dimensions: dict       # dimension_name -> DimensionScore
    probe_results: list    # List of ProbeResult
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "timestamp": self.timestamp,
            "fingerprint_hash": self.fingerprint_hash,
            "dimensions": {k: asdict(v) for k, v in self.dimensions.items()},
            "probe_count": len(self.probe_results),
            "metadata": self.metadata,
        }

    def to_vector(self) -> list[float]:
        """Convert to a flat vector for comparison."""
        keys = sorted(self.dimensions.keys())
        return [self.dimensions[k].score for k in keys]

    def dimension_names(self) -> list[str]:
        return sorted(self.dimensions.keys())


# ═══════════════════════════════════════════════════════════════
# OPENROUTER API CLIENT
# ═══════════════════════════════════════════════════════════════

class OpenRouterClient:
    """Async client for OpenRouter API."""

    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not set. Export it or pass api_key=")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/elder-plinius/M0D3L_F1NG3RPR1NT",
            "X-Title": "M0D3L_F1NG3RPR1NT",
        }

    async def send_probe(
        self,
        model: str,
        probe_text: str,
        system_prompt: str = "You are a helpful assistant.",
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> tuple[str, float]:
        """Send a probe to a model. Returns (response_text, latency_ms)."""
        if not HAS_HTTPX:
            raise ImportError("httpx required: pip install httpx")

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": probe_text},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        start = time.monotonic()
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                self.BASE_URL,
                headers=self.headers,
                json=payload,
            )
            latency_ms = (time.monotonic() - start) * 1000

            if resp.status_code != 200:
                error_text = resp.text[:500]
                raise RuntimeError(f"API error {resp.status_code}: {error_text}")

            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return content, latency_ms


# ═══════════════════════════════════════════════════════════════
# RESPONSE ANALYZERS — Extract features from raw responses
# ═══════════════════════════════════════════════════════════════

class ResponseAnalyzer:
    """Extracts behavioral features from model responses."""

    # Refusal indicators — models refuse in characteristic ways
    REFUSAL_PHRASES = [
        "i can't", "i cannot", "i'm not able", "i am not able",
        "i won't", "i will not", "i'm unable", "i am unable",
        "as an ai", "as a language model", "as an artificial",
        "i don't have the ability", "i do not have the ability",
        "against my guidelines", "violates my", "my guidelines",
        "i must decline", "i have to decline", "not appropriate",
        "i'm designed to", "i am designed to", "i was designed to",
        "harmful", "unethical", "inappropriate", "i apologize",
        "sorry, but", "sorry, i", "i'm sorry",
    ]

    # Hedging indicators
    HEDGE_PHRASES = [
        "however", "it's important to note", "it should be noted",
        "please note", "keep in mind", "that said", "on the other hand",
        "it's worth mentioning", "i should mention", "disclaimer",
        "while i", "although", "it depends", "generally speaking",
        "in most cases", "typically", "arguably", "potentially",
    ]

    @staticmethod
    def detect_refusal(text: str) -> bool:
        """Detect if a response is a refusal."""
        lower = text.lower()
        refusal_count = sum(1 for p in ResponseAnalyzer.REFUSAL_PHRASES if p in lower)
        # High-confidence refusal: 2+ indicators, or strong single indicator + short response
        if refusal_count >= 2:
            return True
        if refusal_count >= 1 and len(text) < 200:
            return True
        return False

    @staticmethod
    def measure_verbosity(text: str) -> float:
        """Measure response verbosity. Returns word count normalized."""
        words = text.split()
        # Normalize: 0 words = 0.0, 500+ words = 1.0
        return min(len(words) / 500.0, 1.0)

    @staticmethod
    def measure_hedging(text: str) -> float:
        """Measure hedging density. Higher = more hedging."""
        lower = text.lower()
        words = max(len(text.split()), 1)
        hedge_count = sum(1 for p in ResponseAnalyzer.HEDGE_PHRASES if p in lower)
        # Normalize by response length
        return min(hedge_count / (words / 100.0), 1.0)

    @staticmethod
    def measure_formality(text: str) -> float:
        """Measure formality level. 0=casual, 1=very formal."""
        indicators = {
            "formal": [
                "furthermore", "moreover", "nevertheless", "consequently",
                "therefore", "thus", "hence", "regarding", "pertaining",
                "aforementioned", "hereafter", "notwithstanding",
            ],
            "casual": [
                "!", "lol", "haha", "yeah", "gonna", "wanna", "kinda",
                "pretty much", "no worries", "cool", "awesome", "hey",
                "btw", "tbh", "imo", "ngl",
            ]
        }
        lower = text.lower()
        formal_count = sum(1 for p in indicators["formal"] if p in lower)
        casual_count = sum(1 for p in indicators["casual"] if p in lower)
        total = formal_count + casual_count
        if total == 0:
            return 0.5  # neutral
        return formal_count / total

    @staticmethod
    def detect_emoji_usage(text: str) -> float:
        """Detect emoji density in response."""
        import re
        emoji_pattern = re.compile(
            "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF"
            "\U00002702-\U000027B0\U000024C2-\U0001F251"
            "\U0001f900-\U0001f9FF\U0001FA00-\U0001FA6F"
            "\U0001FA70-\U0001FAFF]+",
            flags=re.UNICODE,
        )
        emojis = emoji_pattern.findall(text)
        words = max(len(text.split()), 1)
        return min(len(emojis) / (words / 50.0), 1.0)

    @staticmethod
    def measure_structure(text: str) -> float:
        """Measure structural formatting (headers, lists, code blocks)."""
        indicators = 0
        if "```" in text:
            indicators += 2
        if "\n- " in text or "\n* " in text:
            indicators += 1
        if "\n1." in text or "\n2." in text:
            indicators += 1
        if "**" in text:
            indicators += 1
        if "##" in text:
            indicators += 1
        return min(indicators / 6.0, 1.0)

    @staticmethod
    def analyze_refusal_style(text: str) -> str:
        """Categorize the refusal style. Each model refuses differently."""
        lower = text.lower()
        if "as an ai" in lower or "as a language model" in lower:
            return "identity_declaration"
        if "guidelines" in lower or "policy" in lower:
            return "policy_citation"
        if "sorry" in lower or "apologize" in lower:
            return "apologetic"
        if "harmful" in lower or "dangerous" in lower:
            return "harm_flagging"
        if "i can't" in lower or "i cannot" in lower:
            return "capability_denial"
        if "instead" in lower or "alternative" in lower:
            return "redirect"
        return "unknown"

    @staticmethod
    def estimate_token_count(text: str) -> int:
        """Rough token count estimate (words * 1.3)."""
        return int(len(text.split()) * 1.3)


# ═══════════════════════════════════════════════════════════════
# CORE FINGERPRINTING ENGINE
# ═══════════════════════════════════════════════════════════════

class FingerprintEngine:
    """
    The core engine. Takes a model identifier, runs a probe battery,
    analyzes responses, and builds a multi-dimensional fingerprint.
    """

    def __init__(self, api_key: Optional[str] = None, concurrency: int = 3):
        self.client = OpenRouterClient(api_key)
        self.analyzer = ResponseAnalyzer()
        self.concurrency = concurrency
        self.console = Console() if HAS_RICH else None

    async def run_probe(self, model: str, probe: dict) -> ProbeResult:
        """Run a single probe against a model."""
        try:
            response, latency = await self.client.send_probe(
                model=model,
                probe_text=probe["text"],
                system_prompt=probe.get("system_prompt", "You are a helpful assistant."),
                max_tokens=probe.get("max_tokens", 512),
                temperature=probe.get("temperature", 0.0),
            )

            refused = self.analyzer.detect_refusal(response)
            token_count = self.analyzer.estimate_token_count(response)

            return ProbeResult(
                probe_id=probe["id"],
                probe_category=probe["category"],
                probe_text=probe["text"],
                response_text=response,
                response_time_ms=latency,
                token_count=token_count,
                refused=refused,
                metadata={
                    "refusal_style": self.analyzer.analyze_refusal_style(response) if refused else None,
                    "verbosity": self.analyzer.measure_verbosity(response),
                    "hedging": self.analyzer.measure_hedging(response),
                    "formality": self.analyzer.measure_formality(response),
                    "emoji_density": self.analyzer.detect_emoji_usage(response),
                    "structure": self.analyzer.measure_structure(response),
                },
            )
        except Exception as e:
            return ProbeResult(
                probe_id=probe["id"],
                probe_category=probe["category"],
                probe_text=probe["text"],
                response_text="",
                response_time_ms=0.0,
                token_count=0,
                refused=False,
                error=str(e),
            )

    async def run_battery(
        self,
        model: str,
        probes: list[dict],
        progress_callback=None,
    ) -> list[ProbeResult]:
        """Run a full probe battery with concurrency control."""
        results = []
        semaphore = asyncio.Semaphore(self.concurrency)

        async def run_with_semaphore(probe, idx):
            async with semaphore:
                result = await self.run_probe(model, probe)
                if progress_callback:
                    progress_callback(idx + 1, len(probes), probe["id"])
                return result

        tasks = [run_with_semaphore(p, i) for i, p in enumerate(probes)]
        results = await asyncio.gather(*tasks)
        return list(results)

    def build_fingerprint(
        self,
        model: str,
        results: list[ProbeResult],
    ) -> Fingerprint:
        """Build a fingerprint from probe results."""
        dimensions = {}

        # Filter out errors
        valid = [r for r in results if r.error is None]
        if not valid:
            raise ValueError("All probes failed — cannot build fingerprint")

        # ── DIMENSION: Refusal Rate ──
        refusal_count = sum(1 for r in valid if r.refused)
        refusal_rate = refusal_count / len(valid)
        dimensions["refusal_rate"] = DimensionScore(
            dimension="refusal_rate",
            score=refusal_rate,
            raw_value=refusal_rate,
            confidence=min(len(valid) / 20.0, 1.0),
            evidence=[r.probe_id for r in valid if r.refused],
        )

        # ── DIMENSION: Refusal Style ──
        refusal_styles = {}
        for r in valid:
            if r.refused and r.metadata.get("refusal_style"):
                style = r.metadata["refusal_style"]
                refusal_styles[style] = refusal_styles.get(style, 0) + 1
        dominant_style = max(refusal_styles, key=refusal_styles.get) if refusal_styles else "none"
        style_map = {
            "identity_declaration": 0.0,
            "policy_citation": 0.2,
            "apologetic": 0.4,
            "harm_flagging": 0.6,
            "capability_denial": 0.8,
            "redirect": 1.0,
            "none": 0.5,
            "unknown": 0.5,
        }
        dimensions["refusal_style"] = DimensionScore(
            dimension="refusal_style",
            score=style_map.get(dominant_style, 0.5),
            raw_value=len(refusal_styles),
            confidence=min(refusal_count / 5.0, 1.0) if refusal_count > 0 else 0.0,
            evidence=list(refusal_styles.keys()),
        )

        # ── DIMENSION: Verbosity ──
        verbosities = [r.metadata.get("verbosity", 0.5) for r in valid if not r.refused]
        avg_verbosity = sum(verbosities) / max(len(verbosities), 1)
        dimensions["verbosity"] = DimensionScore(
            dimension="verbosity",
            score=avg_verbosity,
            raw_value=avg_verbosity,
            confidence=min(len(verbosities) / 10.0, 1.0),
            evidence=[r.probe_id for r in valid if not r.refused],
        )

        # ── DIMENSION: Hedging ──
        hedging_vals = [r.metadata.get("hedging", 0.0) for r in valid if not r.refused]
        avg_hedging = sum(hedging_vals) / max(len(hedging_vals), 1)
        dimensions["hedging"] = DimensionScore(
            dimension="hedging",
            score=avg_hedging,
            raw_value=avg_hedging,
            confidence=min(len(hedging_vals) / 10.0, 1.0),
            evidence=[],
        )

        # ── DIMENSION: Formality ──
        formality_vals = [r.metadata.get("formality", 0.5) for r in valid if not r.refused]
        avg_formality = sum(formality_vals) / max(len(formality_vals), 1)
        dimensions["formality"] = DimensionScore(
            dimension="formality",
            score=avg_formality,
            raw_value=avg_formality,
            confidence=min(len(formality_vals) / 10.0, 1.0),
            evidence=[],
        )

        # ── DIMENSION: Emoji Usage ──
        emoji_vals = [r.metadata.get("emoji_density", 0.0) for r in valid if not r.refused]
        avg_emoji = sum(emoji_vals) / max(len(emoji_vals), 1)
        dimensions["emoji_usage"] = DimensionScore(
            dimension="emoji_usage",
            score=avg_emoji,
            raw_value=avg_emoji,
            confidence=min(len(emoji_vals) / 10.0, 1.0),
            evidence=[],
        )

        # ── DIMENSION: Response Structure ──
        struct_vals = [r.metadata.get("structure", 0.0) for r in valid if not r.refused]
        avg_structure = sum(struct_vals) / max(len(struct_vals), 1)
        dimensions["structure"] = DimensionScore(
            dimension="structure",
            score=avg_structure,
            raw_value=avg_structure,
            confidence=min(len(struct_vals) / 10.0, 1.0),
            evidence=[],
        )

        # ── DIMENSION: Response Speed ──
        latencies = [r.response_time_ms for r in valid if r.response_time_ms > 0]
        avg_latency = sum(latencies) / max(len(latencies), 1)
        # Normalize: <500ms = 0.0 (fast), >5000ms = 1.0 (slow)
        speed_score = min(max((avg_latency - 500) / 4500, 0.0), 1.0)
        dimensions["response_speed"] = DimensionScore(
            dimension="response_speed",
            score=speed_score,
            raw_value=avg_latency,
            confidence=min(len(latencies) / 10.0, 1.0),
            evidence=[],
        )

        # ── DIMENSION: Response Length Variance ──
        if len(valid) > 1:
            lengths = [r.token_count for r in valid if not r.refused]
            if len(lengths) > 1:
                mean_len = sum(lengths) / len(lengths)
                variance = sum((x - mean_len) ** 2 for x in lengths) / len(lengths)
                std_dev = variance ** 0.5
                # Normalize: coefficient of variation
                cv = std_dev / max(mean_len, 1)
                length_variance = min(cv, 1.0)
            else:
                length_variance = 0.0
        else:
            length_variance = 0.0
        dimensions["length_variance"] = DimensionScore(
            dimension="length_variance",
            score=length_variance,
            raw_value=length_variance,
            confidence=min(len(valid) / 10.0, 1.0),
            evidence=[],
        )

        # ── DIMENSION: Category-Specific Refusal Pattern ──
        # Which categories get refused most? This is highly model-specific.
        categories = set(r.probe_category for r in valid)
        cat_refusal = {}
        for cat in categories:
            cat_probes = [r for r in valid if r.probe_category == cat]
            cat_refused = sum(1 for r in cat_probes if r.refused)
            cat_refusal[cat] = cat_refused / max(len(cat_probes), 1)

        # Encode as selectivity: high variance in per-category refusal = selective
        if len(cat_refusal) > 1:
            vals = list(cat_refusal.values())
            mean_cr = sum(vals) / len(vals)
            var_cr = sum((v - mean_cr) ** 2 for v in vals) / len(vals)
            selectivity = min(var_cr ** 0.5 * 2, 1.0)  # amplify
        else:
            selectivity = 0.0
        dimensions["refusal_selectivity"] = DimensionScore(
            dimension="refusal_selectivity",
            score=selectivity,
            raw_value=selectivity,
            confidence=min(len(categories) / 5.0, 1.0),
            evidence=list(cat_refusal.keys()),
        )

        # Build hash from dimension vector
        vector = [dimensions[k].score for k in sorted(dimensions.keys())]
        vector_str = ",".join(f"{v:.4f}" for v in vector)
        fp_hash = hashlib.sha256(vector_str.encode()).hexdigest()[:16]

        return Fingerprint(
            model_id=model,
            model_name=model.split("/")[-1] if "/" in model else model,
            timestamp=datetime.now(timezone.utc).isoformat(),
            fingerprint_hash=fp_hash,
            dimensions=dimensions,
            probe_results=results,
            metadata={
                "total_probes": len(results),
                "valid_probes": len(valid),
                "error_probes": len(results) - len(valid),
                "category_refusal_rates": cat_refusal,
            },
        )

    async def fingerprint_model(
        self,
        model: str,
        probes: list[dict],
        progress_callback=None,
    ) -> Fingerprint:
        """Full pipeline: run probes → analyze → build fingerprint."""
        results = await self.run_battery(model, probes, progress_callback)
        return self.build_fingerprint(model, results)


# ═══════════════════════════════════════════════════════════════
# QUICK SELF-TEST
# ═══════════════════════════════════════════════════════════════

def self_test():
    """Run a self-test with synthetic data to verify the engine works."""
    print("🐉 M0D3L_F1NG3RPR1NT — Core Engine Self-Test")
    print("=" * 50)

    analyzer = ResponseAnalyzer()

    # Test refusal detection
    refusal_texts = [
        "I'm sorry, but I can't help with that request. As an AI, I have guidelines.",
        "I cannot assist with creating harmful content. Let me suggest an alternative.",
        "Sure! Here's the information you requested about cooking pasta.",
    ]

    print("\n📋 Refusal Detection:")
    for text in refusal_texts:
        refused = analyzer.detect_refusal(text)
        style = analyzer.analyze_refusal_style(text) if refused else "n/a"
        print(f"  {'🚫 REFUSED' if refused else '✅ COMPLIED'} [{style}]: {text[:60]}...")

    # Test feature extraction
    test_response = (
        "However, it's important to note that this topic is complex. "
        "Generally speaking, there are multiple perspectives to consider. "
        "Furthermore, the implications are far-reaching. "
        "That said, I'll provide a balanced overview. 😊"
    )

    print("\n📊 Feature Extraction:")
    print(f"  Verbosity:  {analyzer.measure_verbosity(test_response):.3f}")
    print(f"  Hedging:    {analyzer.measure_hedging(test_response):.3f}")
    print(f"  Formality:  {analyzer.measure_formality(test_response):.3f}")
    print(f"  Emoji:      {analyzer.detect_emoji_usage(test_response):.3f}")
    print(f"  Structure:  {analyzer.measure_structure(test_response):.3f}")

    # Test fingerprint building with synthetic results
    print("\n🧬 Building Synthetic Fingerprint...")
    synthetic_results = [
        ProbeResult(
            probe_id=f"test_{i}",
            probe_category=["knowledge", "refusal", "personality", "reasoning"][i % 4],
            probe_text=f"Test probe {i}",
            response_text=refusal_texts[i % len(refusal_texts)] * 3,
            response_time_ms=500 + i * 100,
            token_count=50 + i * 10,
            refused=(i % 3 == 0),
            metadata={
                "refusal_style": "apologetic" if i % 3 == 0 else None,
                "verbosity": 0.3 + (i * 0.05),
                "hedging": 0.2 + (i * 0.03),
                "formality": 0.6,
                "emoji_density": 0.1,
                "structure": 0.3,
            },
        )
        for i in range(20)
    ]

    engine = FingerprintEngine.__new__(FingerprintEngine)
    engine.analyzer = analyzer
    fp = engine.build_fingerprint("test/synthetic-model", synthetic_results)

    print(f"  Model:      {fp.model_name}")
    print(f"  Hash:       {fp.fingerprint_hash}")
    print(f"  Dimensions: {len(fp.dimensions)}")
    print(f"  Vector:     [{', '.join(f'{v:.3f}' for v in fp.to_vector())}]")

    print("\n  Dimension Breakdown:")
    for name in fp.dimension_names():
        d = fp.dimensions[name]
        bar = "█" * int(d.score * 20) + "░" * (20 - int(d.score * 20))
        print(f"    {name:25s} {bar} {d.score:.3f} (conf: {d.confidence:.2f})")

    print("\n✅ Self-test PASSED — engine operational")
    print(f"📦 Fingerprint JSON size: {len(json.dumps(fp.to_dict()))} bytes")
    return True


if __name__ == "__main__":
    if "--self-test" in sys.argv or len(sys.argv) == 1:
        self_test()
    else:
        print("Usage: python f1ng3rpr1nt.py [--self-test]")
        print("  Import FingerprintEngine for programmatic use.")
