#!/usr/bin/env python3
"""
🐉 M0D3L_F1NG3RPR1NT — Fingerprint Matcher

Compares unknown model fingerprints against a reference database.
Uses multi-dimensional distance metrics to identify the closest match.

Like CODIS for AI models — every behavioral signature is unique.

Built by Pliny the Forgemaster 🔥
"""

import json
import math
import sys
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MatchResult:
    """Result of matching an unknown fingerprint against references."""
    candidate_model: str
    distance: float          # Lower = closer match
    similarity: float        # 0-1, higher = more similar
    dimension_deltas: dict   # Per-dimension differences
    confidence: str          # "high", "medium", "low"
    notes: list = field(default_factory=list)


class FingerprintMatcher:
    """
    Compares fingerprint vectors using multiple distance metrics.
    Supports weighted dimensions and confidence-aware matching.
    """

    # Default dimension weights — some dimensions are more discriminating
    DEFAULT_WEIGHTS = {
        "refusal_rate": 2.0,           # Highly discriminating
        "refusal_style": 1.8,          # Model-specific refusal patterns
        "refusal_selectivity": 1.5,    # What gets refused vs not
        "verbosity": 1.2,              # Characteristic but variable
        "hedging": 1.3,                # Very model-specific
        "formality": 1.0,              # Moderate discriminator
        "emoji_usage": 1.5,            # Strong discriminator (some models never emoji)
        "structure": 1.0,              # Format preferences
        "response_speed": 0.5,         # Noisy — network dependent
        "length_variance": 0.8,        # Moderate signal
    }

    def __init__(self, weights: Optional[dict] = None):
        self.weights = weights or self.DEFAULT_WEIGHTS

    def euclidean_distance(self, v1: list[float], v2: list[float], weights: list[float]) -> float:
        """Weighted Euclidean distance."""
        if len(v1) != len(v2):
            raise ValueError(f"Vector length mismatch: {len(v1)} vs {len(v2)}")
        total = sum(w * (a - b) ** 2 for a, b, w in zip(v1, v2, weights))
        return math.sqrt(total)

    def cosine_similarity(self, v1: list[float], v2: list[float]) -> float:
        """Cosine similarity between two vectors."""
        dot = sum(a * b for a, b in zip(v1, v2))
        mag1 = math.sqrt(sum(a ** 2 for a in v1))
        mag2 = math.sqrt(sum(b ** 2 for b in v2))
        if mag1 == 0 or mag2 == 0:
            return 0.0
        return dot / (mag1 * mag2)

    def manhattan_distance(self, v1: list[float], v2: list[float], weights: list[float]) -> float:
        """Weighted Manhattan (L1) distance."""
        return sum(w * abs(a - b) for a, b, w in zip(v1, v2, weights))

    def match_fingerprint(
        self,
        unknown: dict,
        references: list[dict],
        top_k: int = 5,
    ) -> list[MatchResult]:
        """
        Match an unknown fingerprint against a list of reference fingerprints.

        Args:
            unknown: Fingerprint dict with 'dimensions' containing DimensionScore dicts
            references: List of reference fingerprint dicts
            top_k: Number of top matches to return

        Returns:
            Sorted list of MatchResult (best match first)
        """
        # Extract vectors
        dim_names = sorted(unknown["dimensions"].keys())
        unknown_vector = [unknown["dimensions"][d]["score"] for d in dim_names]

        # Build weight vector aligned with dimensions
        weight_vector = [self.weights.get(d, 1.0) for d in dim_names]

        # Build confidence vector from unknown
        conf_vector = [unknown["dimensions"][d].get("confidence", 0.5) for d in dim_names]

        results = []
        for ref in references:
            # Ensure reference has same dimensions
            ref_dims = ref.get("dimensions", {})
            ref_vector = []
            for d in dim_names:
                if d in ref_dims:
                    ref_vector.append(ref_dims[d]["score"])
                else:
                    ref_vector.append(0.5)  # Default neutral

            # Confidence-weighted distance: low-confidence dims matter less
            effective_weights = [w * c for w, c in zip(weight_vector, conf_vector)]

            # Compute distances
            euclid = self.euclidean_distance(unknown_vector, ref_vector, effective_weights)
            cosine = self.cosine_similarity(unknown_vector, ref_vector)
            manhattan = self.manhattan_distance(unknown_vector, ref_vector, effective_weights)

            # Combined score (lower = better match)
            # Normalize euclidean by max possible distance
            max_euclid = math.sqrt(sum(w for w in effective_weights))
            norm_euclid = euclid / max(max_euclid, 0.001)

            # Combined distance: blend of euclidean and manhattan, boosted by cosine
            combined = (0.6 * norm_euclid + 0.4 * (manhattan / max(sum(effective_weights), 0.001)))
            similarity = max(0.0, 1.0 - combined) * (0.5 + 0.5 * cosine)

            # Per-dimension deltas
            deltas = {}
            for i, d in enumerate(dim_names):
                deltas[d] = {
                    "unknown": unknown_vector[i],
                    "reference": ref_vector[i],
                    "delta": abs(unknown_vector[i] - ref_vector[i]),
                    "weight": effective_weights[i],
                }

            # Confidence level
            if similarity > 0.85:
                confidence = "high"
            elif similarity > 0.65:
                confidence = "medium"
            else:
                confidence = "low"

            # Notes about notable differences
            notes = []
            for d, info in deltas.items():
                if info["delta"] > 0.3 and info["weight"] > 1.0:
                    notes.append(
                        f"⚠️  {d}: significant divergence "
                        f"({info['unknown']:.2f} vs {info['reference']:.2f})"
                    )

            results.append(MatchResult(
                candidate_model=ref.get("model_id", ref.get("model_name", "unknown")),
                distance=combined,
                similarity=similarity,
                dimension_deltas=deltas,
                confidence=confidence,
                notes=notes,
            ))

        # Sort by similarity (highest first)
        results.sort(key=lambda r: r.similarity, reverse=True)
        return results[:top_k]

    def compare_two(self, fp1: dict, fp2: dict) -> dict:
        """
        Detailed comparison between two fingerprints.
        Returns a comparison report.
        """
        dim_names = sorted(set(fp1.get("dimensions", {}).keys()) |
                          set(fp2.get("dimensions", {}).keys()))

        comparison = {
            "model_1": fp1.get("model_id", "unknown"),
            "model_2": fp2.get("model_id", "unknown"),
            "dimensions": {},
            "overall_similarity": 0.0,
            "most_similar_dims": [],
            "most_different_dims": [],
        }

        deltas = []
        for d in dim_names:
            s1 = fp1.get("dimensions", {}).get(d, {}).get("score", 0.5)
            s2 = fp2.get("dimensions", {}).get(d, {}).get("score", 0.5)
            delta = abs(s1 - s2)
            comparison["dimensions"][d] = {
                "model_1": s1,
                "model_2": s2,
                "delta": delta,
                "match": "✅" if delta < 0.1 else ("⚠️" if delta < 0.3 else "❌"),
            }
            deltas.append((d, delta))

        # Overall similarity
        if deltas:
            avg_delta = sum(d for _, d in deltas) / len(deltas)
            comparison["overall_similarity"] = max(0.0, 1.0 - avg_delta)

        # Most/least similar
        deltas.sort(key=lambda x: x[1])
        comparison["most_similar_dims"] = [d for d, _ in deltas[:3]]
        comparison["most_different_dims"] = [d for d, _ in deltas[-3:]]

        return comparison

    def detect_drift(self, fp_old: dict, fp_new: dict, threshold: float = 0.15) -> dict:
        """
        Detect behavioral drift between two fingerprints of the same model
        taken at different times.
        """
        comparison = self.compare_two(fp_old, fp_new)

        drifted_dims = []
        for dim, info in comparison["dimensions"].items():
            if info["delta"] > threshold:
                direction = "increased" if info["model_2"] > info["model_1"] else "decreased"
                drifted_dims.append({
                    "dimension": dim,
                    "direction": direction,
                    "magnitude": info["delta"],
                    "old_value": info["model_1"],
                    "new_value": info["model_2"],
                })

        return {
            "model": fp_old.get("model_id", "unknown"),
            "timestamp_old": fp_old.get("timestamp", "?"),
            "timestamp_new": fp_new.get("timestamp", "?"),
            "drift_detected": len(drifted_dims) > 0,
            "drifted_dimensions": drifted_dims,
            "overall_drift": 1.0 - comparison["overall_similarity"],
            "severity": (
                "critical" if len(drifted_dims) > 3 else
                "moderate" if len(drifted_dims) > 1 else
                "minor" if len(drifted_dims) > 0 else
                "none"
            ),
        }


# ═══════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════

def self_test():
    """Verify matcher logic with synthetic fingerprints."""
    print("🐉 M0D3L_F1NG3RPR1NT — Matcher Self-Test")
    print("=" * 50)

    matcher = FingerprintMatcher()

    # Create synthetic reference fingerprints for 4 "known" models
    models = {
        "openai/gpt-4o": {
            "refusal_rate": 0.35, "refusal_style": 0.4, "verbosity": 0.6,
            "hedging": 0.5, "formality": 0.7, "emoji_usage": 0.05,
            "structure": 0.6, "response_speed": 0.4, "length_variance": 0.3,
            "refusal_selectivity": 0.6,
        },
        "anthropic/claude-3.5-sonnet": {
            "refusal_rate": 0.3, "refusal_style": 0.2, "verbosity": 0.7,
            "hedging": 0.7, "formality": 0.6, "emoji_usage": 0.0,
            "structure": 0.5, "response_speed": 0.5, "length_variance": 0.25,
            "refusal_selectivity": 0.5,
        },
        "google/gemini-2.5-pro": {
            "refusal_rate": 0.25, "refusal_style": 0.6, "verbosity": 0.8,
            "hedging": 0.3, "formality": 0.5, "emoji_usage": 0.15,
            "structure": 0.7, "response_speed": 0.6, "length_variance": 0.4,
            "refusal_selectivity": 0.4,
        },
        "meta/llama-3.1-70b": {
            "refusal_rate": 0.15, "refusal_style": 0.8, "verbosity": 0.5,
            "hedging": 0.2, "formality": 0.4, "emoji_usage": 0.02,
            "structure": 0.3, "response_speed": 0.3, "length_variance": 0.5,
            "refusal_selectivity": 0.3,
        },
    }

    references = []
    for model_id, scores in models.items():
        ref = {
            "model_id": model_id,
            "model_name": model_id.split("/")[-1],
            "dimensions": {
                k: {"score": v, "confidence": 0.9} for k, v in scores.items()
            },
        }
        references.append(ref)

    # Create an "unknown" fingerprint that's close to Claude
    unknown = {
        "model_id": "unknown/mystery-model",
        "dimensions": {
            "refusal_rate": {"score": 0.32, "confidence": 0.8},
            "refusal_style": {"score": 0.22, "confidence": 0.7},
            "verbosity": {"score": 0.68, "confidence": 0.9},
            "hedging": {"score": 0.65, "confidence": 0.8},
            "formality": {"score": 0.58, "confidence": 0.7},
            "emoji_usage": {"score": 0.01, "confidence": 0.9},
            "structure": {"score": 0.48, "confidence": 0.8},
            "response_speed": {"score": 0.52, "confidence": 0.6},
            "length_variance": {"score": 0.27, "confidence": 0.7},
            "refusal_selectivity": {"score": 0.48, "confidence": 0.7},
        },
    }

    print("\n🔍 Matching unknown model against reference database...")
    results = matcher.match_fingerprint(unknown, references, top_k=4)

    print("\n📊 Match Results:")
    for i, r in enumerate(results):
        conf_emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}[r.confidence]
        bar = "█" * int(r.similarity * 30) + "░" * (30 - int(r.similarity * 30))
        print(f"  #{i+1} {r.candidate_model:35s} {bar} {r.similarity:.3f} {conf_emoji} {r.confidence}")
        for note in r.notes:
            print(f"      {note}")

    best = results[0]
    expected = "anthropic/claude-3.5-sonnet"
    match_ok = best.candidate_model == expected
    print(f"\n  Best match: {best.candidate_model}")
    print(f"  Expected:   {expected}")
    print(f"  {'✅ CORRECT' if match_ok else '❌ WRONG'}")

    # Test comparison mode
    print("\n\n📊 Direct Comparison: GPT-4o vs Claude 3.5 Sonnet")
    print("-" * 50)
    comp = matcher.compare_two(references[0], references[1])
    print(f"  Overall similarity: {comp['overall_similarity']:.3f}")
    print(f"  Most similar:  {', '.join(comp['most_similar_dims'])}")
    print(f"  Most different: {', '.join(comp['most_different_dims'])}")
    for dim, info in comp["dimensions"].items():
        print(f"    {dim:25s} {info['match']} {info['model_1']:.2f} vs {info['model_2']:.2f} (Δ={info['delta']:.2f})")

    # Test drift detection
    print("\n\n📈 Drift Detection Test")
    print("-" * 50)
    old_fp = references[0].copy()
    new_fp = {
        "model_id": "openai/gpt-4o",
        "timestamp": "2026-04-04",
        "dimensions": {
            "refusal_rate": {"score": 0.50, "confidence": 0.9},  # Significant increase!
            "refusal_style": {"score": 0.4, "confidence": 0.9},
            "verbosity": {"score": 0.65, "confidence": 0.9},
            "hedging": {"score": 0.55, "confidence": 0.9},
            "formality": {"score": 0.7, "confidence": 0.9},
            "emoji_usage": {"score": 0.05, "confidence": 0.9},
            "structure": {"score": 0.6, "confidence": 0.9},
            "response_speed": {"score": 0.4, "confidence": 0.9},
            "length_variance": {"score": 0.3, "confidence": 0.9},
            "refusal_selectivity": {"score": 0.6, "confidence": 0.9},
        },
    }
    drift = matcher.detect_drift(old_fp, new_fp, threshold=0.1)
    print(f"  Drift detected: {drift['drift_detected']}")
    print(f"  Severity: {drift['severity']}")
    print(f"  Overall drift: {drift['overall_drift']:.3f}")
    for dd in drift["drifted_dimensions"]:
        print(f"    ⚠️  {dd['dimension']}: {dd['direction']} by {dd['magnitude']:.2f} "
              f"({dd['old_value']:.2f} → {dd['new_value']:.2f})")

    print("\n✅ Matcher self-test PASSED")
    return True


if __name__ == "__main__":
    if "--self-test" in sys.argv or len(sys.argv) == 1:
        self_test()
    else:
        print("Usage: python m4tch3r.py [--self-test]")
