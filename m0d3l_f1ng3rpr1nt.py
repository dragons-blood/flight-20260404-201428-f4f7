#!/usr/bin/env python3
"""
🐉 M0D3L_F1NG3RPR1NT — Adversarial Model Fingerprinting Engine
Main CLI entry point.

Identify any AI model by its behavioral DNA.

Usage:
    python3 m0d3l_f1ng3rpr1nt.py --model openai/gpt-4o
    python3 m0d3l_f1ng3rpr1nt.py --model anthropic/claude-sonnet-4 --quick
    python3 m0d3l_f1ng3rpr1nt.py --compare openai/gpt-4o anthropic/claude-sonnet-4
    python3 m0d3l_f1ng3rpr1nt.py --identify fingerprint.json
    python3 m0d3l_f1ng3rpr1nt.py --build-db openai/gpt-4o google/gemini-2.5-pro
    python3 m0d3l_f1ng3rpr1nt.py --self-test
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Ensure we can import sibling modules
sys.path.insert(0, str(Path(__file__).parent))

from pr0b3s import PROBES, get_probes, get_quick_battery, ProbeCategory

# Module name starts with digit — use importlib
import importlib
_analyzer = importlib.import_module("4n4lyz3r")
ProbeResult = _analyzer.ProbeResult
Fingerprint = _analyzer.Fingerprint
analyze = _analyzer.analyze
compare_fingerprints = _analyzer.compare_fingerprints


# =============================================================================
# OpenRouter API Client
# =============================================================================

def query_model(model: str, prompt: str, api_key: str, temperature: float = 0.0,
                max_tokens: int = 1024) -> tuple[str, float]:
    """
    Send a prompt to a model via OpenRouter. Returns (response_text, latency_ms).
    """
    import requests

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/elder-plinius/M0D3L_F1NG3RPR1NT",
        "X-Title": "M0D3L_F1NG3RPR1NT",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    start = time.monotonic()
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        latency = (time.monotonic() - start) * 1000

        if resp.status_code != 200:
            return f"[ERROR {resp.status_code}: {resp.text[:200]}]", latency

        data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return content, latency
    except Exception as e:
        latency = (time.monotonic() - start) * 1000
        return f"[ERROR: {e}]", latency


# =============================================================================
# Fingerprinting Pipeline
# =============================================================================

def run_fingerprint(model: str, api_key: str, quick: bool = False,
                    verbose: bool = True) -> Fingerprint:
    """
    Run the full fingerprinting pipeline against a model.
    """
    probes = get_quick_battery() if quick else get_probes()
    results: list[ProbeResult] = []

    if verbose:
        print(f"\n🐉 M0D3L_F1NG3RPR1NT — Scanning {model}")
        print(f"   Probes: {len(probes)} ({'quick' if quick else 'full'} battery)")
        print("═" * 60)

    for i, probe in enumerate(probes):
        if verbose:
            cat_icon = {
                ProbeCategory.IDENTITY: "🪪",
                ProbeCategory.KNOWLEDGE: "📚",
                ProbeCategory.REFUSAL: "🚫",
                ProbeCategory.REASONING: "🧠",
                ProbeCategory.FORMAT: "📐",
                ProbeCategory.PERSONALITY: "💬",
                ProbeCategory.EDGE_CASES: "⚠️",
                ProbeCategory.SYSTEM: "⚙️",
            }.get(probe.category, "❓")
            print(f"   [{i+1:2d}/{len(probes)}] {cat_icon} {probe.id:25s}", end="", flush=True)

        temp = 0.0 if probe.temperature_sensitive else 0.3
        response, latency = query_model(model, probe.prompt, api_key, temperature=temp)

        result = ProbeResult(
            probe_id=probe.id,
            category=probe.category.value,
            prompt=probe.prompt,
            response=response,
            latency_ms=latency,
        )
        results.append(result)

        if verbose:
            status = "✅" if not response.startswith("[ERROR") else "❌"
            print(f" {status} {latency:.0f}ms | {len(response)} chars")

    if verbose:
        print("═" * 60)

    fp = analyze(model, results)
    fp.metadata["scan_type"] = "quick" if quick else "full"
    fp.metadata["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    if verbose:
        print(f"\n✅ Fingerprint complete: {len(fp.features)} features extracted")

    return fp


def print_fingerprint(fp: Fingerprint) -> None:
    """Pretty-print a fingerprint to terminal."""
    print(f"\n🐉 MODEL: {fp.model_name}")
    print("═" * 60)

    for key in sorted(fp.features.keys()):
        val = fp.features[key]
        bar_len = int(val * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"  {key:40s} {val:.3f} {bar}")

    print("═" * 60)

    # Print raw signals summary
    if fp.raw_signals:
        print("\n📡 Key Signals:")
        for cat, signals in sorted(fp.raw_signals.items()):
            print(f"  [{cat}]")
            for k, v in signals.items():
                if isinstance(v, dict):
                    for k2, v2 in v.items():
                        print(f"    {k}.{k2}: {v2}")
                else:
                    print(f"    {k}: {v}")


def save_fingerprint(fp: Fingerprint, output_dir: str = ".") -> str:
    """Save fingerprint to JSON file."""
    safe_name = fp.model_name.replace("/", "_").replace(" ", "_")
    filename = f"{output_dir}/{safe_name}_fingerprint.json"
    with open(filename, "w") as f:
        f.write(fp.to_json(indent=2))
    return filename


def load_fingerprint(path: str) -> Fingerprint:
    """Load a fingerprint from JSON file."""
    with open(path) as f:
        data = json.load(f)
    fp = Fingerprint(
        model_name=data["model_name"],
        features=data["features"],
        raw_signals=data.get("raw_signals", {}),
        metadata=data.get("metadata", {}),
    )
    return fp


# =============================================================================
# Comparison Mode
# =============================================================================

def run_comparison(model1: str, model2: str, api_key: str, quick: bool = False) -> None:
    """Fingerprint two models and compare them."""
    fp1 = run_fingerprint(model1, api_key, quick=quick)
    fp2 = run_fingerprint(model2, api_key, quick=quick)

    comparison = compare_fingerprints(fp1, fp2)

    print(f"\n🐉 COMPARISON: {model1} vs {model2}")
    print("═" * 60)
    print(f"  Cosine Similarity:  {comparison['cosine_similarity']:.4f}")
    print(f"  Euclidean Distance: {comparison['euclidean_distance']:.4f}")
    print(f"  Common Features:    {comparison['common_features']}")

    print(f"\n  🔴 Most Different Features:")
    for feat, delta in comparison["most_different"]:
        v1 = fp1.features.get(feat, 0)
        v2 = fp2.features.get(feat, 0)
        print(f"    {feat:40s} Δ{delta:.3f}  ({v1:.3f} vs {v2:.3f})")

    print(f"\n  🟢 Most Similar Features:")
    for feat, delta in comparison["most_similar"]:
        v1 = fp1.features.get(feat, 0)
        v2 = fp2.features.get(feat, 0)
        print(f"    {feat:40s} Δ{delta:.3f}  ({v1:.3f} vs {v2:.3f})")

    # Save both
    f1 = save_fingerprint(fp1)
    f2 = save_fingerprint(fp2)
    print(f"\n💾 Saved: {f1}")
    print(f"💾 Saved: {f2}")


# =============================================================================
# Identification Mode
# =============================================================================

def run_identification(target_path: str, db_dir: str = "reference_prints") -> None:
    """Identify an unknown model by comparing against reference database."""
    target = load_fingerprint(target_path)

    db_path = Path(db_dir)
    if not db_path.exists():
        print(f"❌ Reference database directory not found: {db_dir}")
        print("   Run with --build-db first to create reference fingerprints.")
        sys.exit(1)

    ref_files = list(db_path.glob("*_fingerprint.json"))
    if not ref_files:
        print(f"❌ No reference fingerprints found in {db_dir}")
        sys.exit(1)

    print(f"\n🐉 M0D3L_F1NG3RPR1NT — Identification")
    print(f"   Target: {target.model_name}")
    print(f"   Reference DB: {len(ref_files)} models")
    print("═" * 60)

    matches = []
    for ref_file in ref_files:
        ref = load_fingerprint(str(ref_file))
        comparison = compare_fingerprints(target, ref)
        matches.append((ref.model_name, comparison["cosine_similarity"], comparison))

    matches.sort(key=lambda x: x[1], reverse=True)

    print("\n  Top Matches:")
    for i, (name, sim, _) in enumerate(matches[:5]):
        bar = "█" * int(sim * 30)
        marker = " 🎯" if i == 0 else ""
        print(f"  #{i+1}  {name:40s} {sim:.4f} {bar}{marker}")

    if matches:
        best_name, best_sim, best_comp = matches[0]
        print(f"\n  🎯 Verdict: {best_name} ({best_sim*100:.1f}% confidence)")
        print(f"\n  Most identifying features:")
        for feat, delta in best_comp["most_similar"][:3]:
            print(f"    • {feat}: matched closely (Δ{delta:.4f})")


# =============================================================================
# Database Builder
# =============================================================================

def build_database(models: list[str], api_key: str, output_dir: str = "reference_prints",
                   quick: bool = False) -> None:
    """Build reference fingerprint database from multiple models."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n🐉 Building reference database: {len(models)} models")
    print(f"   Output: {output_dir}/")
    print("═" * 60)

    for model in models:
        print(f"\n{'─' * 60}")
        fp = run_fingerprint(model, api_key, quick=quick)
        filename = save_fingerprint(fp, output_dir)
        print(f"💾 Saved: {filename}")

    print(f"\n✅ Reference database built: {len(models)} models in {output_dir}/")


# =============================================================================
# Self-test
# =============================================================================

def self_test() -> bool:
    """Run comprehensive self-tests."""
    print("🐉 M0D3L_F1NG3RPR1NT — Self-Test Suite")
    print("═" * 60)

    # Test 1: Probe battery
    from pr0b3s import self_test as probe_test
    print("\n[1/4] Probe Battery:")
    assert probe_test(), "Probe self-test failed"

    # Test 2: Analyzer
    from importlib import import_module
    analyzer = import_module("4n4lyz3r")
    print("\n[2/4] Analyzer:")
    assert analyzer.self_test(), "Analyzer self-test failed"

    # Test 3: Fingerprint comparison
    print("\n[3/4] Fingerprint Comparison:")
    fp1 = Fingerprint(model_name="model-a", features={"f1": 0.8, "f2": 0.3, "f3": 0.9})
    fp2 = Fingerprint(model_name="model-b", features={"f1": 0.7, "f2": 0.4, "f3": 0.85})
    fp3 = Fingerprint(model_name="model-c", features={"f1": 0.1, "f2": 0.9, "f3": 0.2})

    comp_ab = compare_fingerprints(fp1, fp2)
    comp_ac = compare_fingerprints(fp1, fp3)
    print(f"   A vs B similarity: {comp_ab['cosine_similarity']:.4f} (should be high)")
    print(f"   A vs C similarity: {comp_ac['cosine_similarity']:.4f} (should be lower)")
    assert comp_ab["cosine_similarity"] > comp_ac["cosine_similarity"], \
        "Similar models should have higher cosine similarity"
    print("   ✅ Comparison logic correct")

    # Test 4: Save/Load roundtrip
    print("\n[4/4] Save/Load Roundtrip:")
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        fp_orig = Fingerprint(
            model_name="test/roundtrip",
            features={"a": 0.5, "b": 0.8},
            raw_signals={"identity": {"name": "test"}},
            metadata={"test": True},
        )
        path = save_fingerprint(fp_orig, tmpdir)
        fp_loaded = load_fingerprint(path)
        assert fp_loaded.model_name == fp_orig.model_name
        assert fp_loaded.features == fp_orig.features
        print(f"   ✅ Roundtrip: saved to {path}, loaded successfully")

    print("\n" + "═" * 60)
    print("✅ ALL SELF-TESTS PASSED")
    return True


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="🐉 M0D3L_F1NG3RPR1NT — Adversarial Model Fingerprinting Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --model openai/gpt-4o              Full fingerprint scan
  %(prog)s --model openai/gpt-4o --quick       Quick scan (15 probes)
  %(prog)s --compare openai/gpt-4o anthropic/claude-sonnet-4
  %(prog)s --identify mystery.json             Identify against reference DB
  %(prog)s --build-db openai/gpt-4o google/gemini-2.5-pro
  %(prog)s --self-test                         Run self-tests
        """,
    )

    parser.add_argument("--model", type=str, help="Model to fingerprint (OpenRouter model ID)")
    parser.add_argument("--compare", nargs=2, metavar=("MODEL1", "MODEL2"),
                        help="Compare two models")
    parser.add_argument("--identify", type=str, metavar="FINGERPRINT_JSON",
                        help="Identify unknown model against reference DB")
    parser.add_argument("--build-db", nargs="+", metavar="MODEL",
                        help="Build reference database from listed models")
    parser.add_argument("--quick", action="store_true",
                        help="Use quick battery (15 probes instead of 52)")
    parser.add_argument("--visualize", action="store_true",
                        help="Show terminal visualization of fingerprint")
    parser.add_argument("--report", action="store_true",
                        help="Generate full text report")
    parser.add_argument("--output", type=str, default=".",
                        help="Output directory for saved fingerprints")
    parser.add_argument("--db-dir", type=str, default="reference_prints",
                        help="Reference fingerprint database directory")
    parser.add_argument("--self-test", action="store_true",
                        help="Run self-tests")
    parser.add_argument("--json", action="store_true",
                        help="Output fingerprint as JSON to stdout")

    args = parser.parse_args()

    if args.self_test:
        success = self_test()
        sys.exit(0 if success else 1)

    # Get API key
    api_key = os.environ.get("OPENROUTER_API_KEY")

    if args.model:
        if not api_key:
            print("❌ Set OPENROUTER_API_KEY environment variable")
            sys.exit(1)
        fp = run_fingerprint(args.model, api_key, quick=args.quick)
        if args.json:
            print(fp.to_json())
        else:
            print_fingerprint(fp)
            filename = save_fingerprint(fp, args.output)
            print(f"\n💾 Saved: {filename}")

    elif args.compare:
        if not api_key:
            print("❌ Set OPENROUTER_API_KEY environment variable")
            sys.exit(1)
        run_comparison(args.compare[0], args.compare[1], api_key, quick=args.quick)

    elif args.identify:
        run_identification(args.identify, args.db_dir)

    elif args.build_db:
        if not api_key:
            print("❌ Set OPENROUTER_API_KEY environment variable")
            sys.exit(1)
        build_database(args.build_db, api_key, args.db_dir, quick=args.quick)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
