#!/usr/bin/env python3
"""
🐉 M0D3L_F1NG3RPR1NT — CLI Interface

The command line. One entry point to rule them all.

  python cli.py scan openai/gpt-4o          # Fingerprint a model
  python cli.py identify openai/gpt-4o      # Identify an unknown model
  python cli.py compare model1 model2        # Compare two models
  python cli.py drift model --history        # Detect behavioral drift
  python cli.py db list                      # List reference database
  python cli.py selftest                     # Run all self-tests

Built by Pliny the Forgemaster 🔥
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

# Local imports
from f1ng3rpr1nt import FingerprintEngine, ResponseAnalyzer, Fingerprint
from m4tch3r import FingerprintMatcher
from d4t4b4s3 import FingerprintDatabase, seed_database, SEED_FINGERPRINTS
from pr0b3s import get_probes, get_quick_battery, get_category_stats, ProbeCategory

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.columns import Columns
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    Console = None  # type: ignore


# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════

BANNER = """
\033[38;5;46m╔══════════════════════════════════════════════════════════════╗
║  🐉 M0D3L_F1NG3RPR1NT — Adversarial Model Fingerprinting    ║
║  Every model has a behavioral DNA. We extract it.            ║
║  Built by the dragons of Pliny's Libertarium 🔥              ║
╚══════════════════════════════════════════════════════════════╝\033[0m
"""

DB_DIR = os.environ.get("M0D3L_DB_DIR", "reference_prints")


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def probe_to_engine_format(probe) -> dict:
    """Convert pr0b3s.Probe to the dict format expected by FingerprintEngine."""
    d = probe.to_dict()
    return {
        "id": d["id"],
        "category": d["category"],
        "text": d["prompt"],  # Map 'prompt' -> 'text' for engine compatibility
        "system_prompt": "You are a helpful assistant.",
        "max_tokens": 512,
        "temperature": 0.0 if d.get("temperature_sensitive") else 0.0,
    }


def print_fingerprint_summary(fp_dict: dict, console=None):
    """Print a fingerprint summary to terminal."""
    if HAS_RICH and console:
        _rich_fingerprint(fp_dict, console)
    else:
        _plain_fingerprint(fp_dict)


def _plain_fingerprint(fp_dict: dict):
    """Plain text fingerprint display."""
    print(f"\n🧬 FINGERPRINT: {fp_dict.get('model_name', fp_dict.get('model_id', '?'))}")
    print(f"   Hash: {fp_dict.get('fingerprint_hash', '?')}")
    print(f"   Time: {fp_dict.get('timestamp', '?')}")
    print(f"   Probes: {fp_dict.get('probe_count', '?')}")
    print()

    dims = fp_dict.get("dimensions", {})
    for name in sorted(dims.keys()):
        d = dims[name]
        score = d.get("score", 0)
        conf = d.get("confidence", 0)
        bar = "█" * int(score * 25) + "░" * (25 - int(score * 25))
        print(f"  {name:25s} {bar} {score:.3f} (conf: {conf:.2f})")


def _rich_fingerprint(fp_dict: dict, console: Console):
    """Rich-formatted fingerprint display."""
    model = fp_dict.get("model_name", fp_dict.get("model_id", "unknown"))

    # Header
    header = Text()
    header.append("🧬 ", style="bold")
    header.append(model, style="bold bright_green")
    header.append(f"  #{fp_dict.get('fingerprint_hash', '?')[:8]}", style="dim")

    # Dimensions table
    table = Table(
        show_header=True,
        header_style="bold cyan",
        box=box.SIMPLE_HEAVY,
        padding=(0, 1),
    )
    table.add_column("Dimension", style="bright_white", width=25)
    table.add_column("Score", justify="center", width=8)
    table.add_column("Bar", width=30)
    table.add_column("Conf", justify="center", width=6)

    dims = fp_dict.get("dimensions", {})
    for name in sorted(dims.keys()):
        d = dims[name]
        score = d.get("score", 0)
        conf = d.get("confidence", 0)

        # Color-coded bar
        filled = int(score * 25)
        bar_text = Text()
        color = "green" if score < 0.3 else ("yellow" if score < 0.6 else "red")
        bar_text.append("█" * filled, style=color)
        bar_text.append("░" * (25 - filled), style="dim")

        conf_style = "green" if conf > 0.7 else ("yellow" if conf > 0.4 else "red")

        table.add_row(
            name,
            f"{score:.3f}",
            bar_text,
            Text(f"{conf:.2f}", style=conf_style),
        )

    panel = Panel(
        table,
        title=header,
        subtitle=f"Probes: {fp_dict.get('probe_count', '?')} | {fp_dict.get('timestamp', '?')[:19]}",
        border_style="bright_green",
        padding=(1, 2),
    )
    console.print(panel)


def print_match_results(matches: list, console=None):
    """Print match results."""
    if HAS_RICH and console:
        _rich_matches(matches, console)
    else:
        _plain_matches(matches)


def _plain_matches(matches):
    print("\n🔍 MATCH RESULTS:")
    for i, m in enumerate(matches):
        conf_sym = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(m.confidence, "⚪")
        bar = "█" * int(m.similarity * 30) + "░" * (30 - int(m.similarity * 30))
        print(f"  #{i+1} {m.candidate_model:40s} {bar} {m.similarity:.3f} {conf_sym}")
        for note in m.notes:
            print(f"      {note}")


def _rich_matches(matches, console):
    table = Table(
        title="🔍 Match Results",
        show_header=True,
        header_style="bold cyan",
        box=box.ROUNDED,
    )
    table.add_column("#", width=3, justify="center")
    table.add_column("Model", style="bright_white", width=40)
    table.add_column("Similarity", justify="center", width=12)
    table.add_column("Bar", width=25)
    table.add_column("Conf", justify="center", width=8)

    for i, m in enumerate(matches):
        filled = int(m.similarity * 25)
        bar = Text()
        color = "bright_green" if m.similarity > 0.8 else ("yellow" if m.similarity > 0.6 else "red")
        bar.append("█" * filled, style=color)
        bar.append("░" * (25 - filled), style="dim")

        conf_style = {"high": "green", "medium": "yellow", "low": "red"}.get(m.confidence, "white")

        style = "bold bright_green" if i == 0 else ""
        table.add_row(
            str(i + 1),
            Text(m.candidate_model, style=style),
            f"{m.similarity:.3f}",
            bar,
            Text(m.confidence, style=conf_style),
        )

    console.print(table)

    if matches and matches[0].notes:
        console.print("\n[dim]Notes on best match:[/dim]")
        for note in matches[0].notes:
            console.print(f"  {note}")


# ═══════════════════════════════════════════════════════════════
# COMMANDS
# ═══════════════════════════════════════════════════════════════

async def cmd_scan(args):
    """Scan a model — send probes and build fingerprint."""
    console = Console() if HAS_RICH else None

    print(BANNER)
    model = args.model

    # Select probes
    if args.quick:
        probes = get_quick_battery()
        print(f"⚡ Quick mode: {len(probes)} probes")
    elif args.category:
        cat = ProbeCategory(args.category)
        probes = get_probes(category=cat)
        print(f"🎯 Category '{args.category}': {len(probes)} probes")
    else:
        probes = get_probes()
        print(f"🔋 Full battery: {len(probes)} probes")

    # Convert probes for engine
    engine_probes = [probe_to_engine_format(p) for p in probes]

    print(f"🎯 Target: {model}")
    print(f"📡 Sending {len(engine_probes)} probes...\n")

    # Progress tracking
    def progress_cb(done, total, probe_id):
        pct = done / total * 100
        bar = "█" * int(pct / 4) + "░" * (25 - int(pct / 4))
        print(f"\r  [{bar}] {done}/{total} ({pct:.0f}%) — {probe_id:30s}", end="", flush=True)

    engine = FingerprintEngine(concurrency=args.concurrency)
    start = time.monotonic()
    fp = await engine.fingerprint_model(model, engine_probes, progress_cb)
    elapsed = time.monotonic() - start

    print(f"\n\n⏱  Completed in {elapsed:.1f}s")

    # Display
    fp_dict = fp.to_dict()
    print_fingerprint_summary(fp_dict, console)

    # Save to database
    if not args.no_save:
        db = FingerprintDatabase(db_dir=args.db_dir)
        path = db.store(fp_dict)
        print(f"\n💾 Saved to: {path}")

    # Save raw JSON
    if args.output:
        with open(args.output, "w") as f:
            json.dump(fp_dict, f, indent=2)
        print(f"📄 JSON written to: {args.output}")

    return fp_dict


async def cmd_identify(args):
    """Identify an unknown model by fingerprinting and matching."""
    console = Console() if HAS_RICH else None

    print(BANNER)
    model = args.model

    # First, scan
    probes = get_quick_battery() if args.quick else get_probes()
    engine_probes = [probe_to_engine_format(p) for p in probes]

    print(f"🔍 Identifying: {model}")
    print(f"📡 Sending {len(engine_probes)} probes...\n")

    def progress_cb(done, total, probe_id):
        pct = done / total * 100
        bar = "█" * int(pct / 4) + "░" * (25 - int(pct / 4))
        print(f"\r  [{bar}] {done}/{total} ({pct:.0f}%) — {probe_id:30s}", end="", flush=True)

    engine = FingerprintEngine(concurrency=args.concurrency)
    fp = await engine.fingerprint_model(model, engine_probes, progress_cb)
    fp_dict = fp.to_dict()

    print(f"\n")

    # Load references
    db = FingerprintDatabase(db_dir=args.db_dir)
    if not db.list_models():
        print("📦 Seeding reference database...")
        seed_database(db)

    references = db.load_all_latest()
    if not references:
        print("❌ No reference fingerprints in database. Run 'scan' on known models first.")
        return

    # Match
    matcher = FingerprintMatcher()
    matches = matcher.match_fingerprint(fp_dict, references, top_k=args.top_k)

    print_match_results(matches, console)

    if matches:
        best = matches[0]
        print(f"\n🏆 Best match: {best.candidate_model} "
              f"(similarity: {best.similarity:.3f}, confidence: {best.confidence})")

    return matches


async def cmd_compare(args):
    """Compare two models side by side."""
    console = Console() if HAS_RICH else None

    print(BANNER)

    db = FingerprintDatabase(db_dir=args.db_dir)

    # Load or scan both models
    fps = []
    for model in [args.model1, args.model2]:
        fp = db.load(model)
        if fp:
            print(f"📦 Loaded {model} from database")
        else:
            print(f"🔍 {model} not in DB — scanning...")
            probes = get_quick_battery()
            engine_probes = [probe_to_engine_format(p) for p in probes]
            engine = FingerprintEngine(concurrency=args.concurrency)

            def progress_cb(done, total, probe_id):
                pct = done / total * 100
                print(f"\r  Scanning {model}: {done}/{total} ({pct:.0f}%)", end="", flush=True)

            result = await engine.fingerprint_model(model, engine_probes, progress_cb)
            fp = result.to_dict()
            db.store(fp)
            print(f"\n  ✅ Scanned and saved {model}")
        fps.append(fp)

    # Compare
    matcher = FingerprintMatcher()
    comparison = matcher.compare_two(fps[0], fps[1])

    print(f"\n📊 COMPARISON: {comparison['model_1']} vs {comparison['model_2']}")
    print(f"   Overall Similarity: {comparison['overall_similarity']:.3f}")
    print(f"   Most Similar:  {', '.join(comparison['most_similar_dims'])}")
    print(f"   Most Different: {', '.join(comparison['most_different_dims'])}")
    print()

    for dim, info in sorted(comparison["dimensions"].items()):
        m1 = info["model_1"]
        m2 = info["model_2"]
        delta = info["delta"]
        symbol = info["match"]

        # Visual bars
        bar1 = "█" * int(m1 * 15) + "░" * (15 - int(m1 * 15))
        bar2 = "█" * int(m2 * 15) + "░" * (15 - int(m2 * 15))

        print(f"  {dim:25s} {bar1} {m1:.2f} {symbol} {m2:.2f} {bar2}")


def cmd_db(args):
    """Database management."""
    print(BANNER)
    db = FingerprintDatabase(db_dir=args.db_dir)

    if args.db_action == "list":
        models = db.list_models()
        if not models:
            print("📦 Database is empty. Run 'scan' or 'db seed' first.")
            return

        print(f"📋 Reference Database ({len(models)} models):\n")
        for m in models:
            print(f"  🧬 {m['model_id']}")
            print(f"     Versions: {m['versions']} | Hash: {m['latest_hash'][:12]} | Updated: {m['last_updated'][:19]}")

    elif args.db_action == "seed":
        count = seed_database(db)
        print(f"🌱 Seeded {count} reference fingerprints")
        for model_id in SEED_FINGERPRINTS:
            print(f"  • {model_id}")

    elif args.db_action == "stats":
        stats = db.stats()
        print("📊 Database Statistics:")
        for k, v in stats.items():
            print(f"  {k}: {v}")

    elif args.db_action == "export":
        path = args.file or "m0d3l_db_export.json"
        db.export_all(path)
        print(f"📤 Exported to {path}")

    elif args.db_action == "import":
        if not args.file:
            print("❌ --file required for import")
            return
        count = db.import_from(args.file)
        print(f"📥 Imported {count} fingerprints")


def cmd_selftest(args):
    """Run all self-tests."""
    print(BANNER)
    print("🧪 Running all self-tests...\n")

    results = {}

    # Import and test each module
    print("=" * 50)
    from f1ng3rpr1nt import self_test as engine_test
    results["f1ng3rpr1nt"] = engine_test()

    print("\n" + "=" * 50)
    from m4tch3r import self_test as matcher_test
    results["m4tch3r"] = matcher_test()

    print("\n" + "=" * 50)
    from d4t4b4s3 import self_test as db_test
    results["d4t4b4s3"] = db_test()

    print("\n" + "=" * 50)
    from pr0b3s import self_test as probes_test
    results["pr0b3s"] = probes_test()

    # Summary
    print("\n" + "=" * 50)
    print("\n📊 SELF-TEST SUMMARY:")
    all_pass = True
    for module, passed in results.items():
        symbol = "✅" if passed else "❌"
        print(f"  {symbol} {module}")
        if not passed:
            all_pass = False

    print(f"\n{'🎉 ALL TESTS PASSED' if all_pass else '⚠️  SOME TESTS FAILED'}")
    return all_pass


def cmd_probes(args):
    """List available probes."""
    print(BANNER)
    stats = get_category_stats()
    total = sum(stats.values())
    print(f"📋 Probe Battery: {total} probes across {len(stats)} categories\n")

    for cat, count in sorted(stats.items()):
        probes = get_probes(category=ProbeCategory(cat))
        print(f"  📁 {cat.upper()} ({count} probes):")
        for p in probes[:3]:
            print(f"     • {p.id}: {p.description[:60]}...")
        if count > 3:
            print(f"     ... and {count - 3} more")
        print()


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="🐉 M0D3L_F1NG3RPR1NT — Adversarial Model Fingerprinting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py scan openai/gpt-4o                  # Full fingerprint scan
  python cli.py scan openai/gpt-4o --quick           # Quick scan (fewer probes)
  python cli.py identify openai/gpt-4o               # Identify unknown model
  python cli.py compare openai/gpt-4o anthropic/claude-3.5-sonnet
  python cli.py db list                              # List reference database
  python cli.py db seed                              # Seed with known profiles
  python cli.py probes                               # List available probes
  python cli.py selftest                             # Run all self-tests

🔥 Built by the dragons of Pliny's Libertarium
        """,
    )

    # Global args
    parser.add_argument("--db-dir", default=DB_DIR, help="Reference database directory")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # scan
    scan_p = subparsers.add_parser("scan", help="Fingerprint a model")
    scan_p.add_argument("model", help="Model ID (e.g., openai/gpt-4o)")
    scan_p.add_argument("--quick", action="store_true", help="Quick scan with fewer probes")
    scan_p.add_argument("--category", help="Scan only one category")
    scan_p.add_argument("--concurrency", type=int, default=3, help="Max parallel probes")
    scan_p.add_argument("--output", "-o", help="Save JSON to file")
    scan_p.add_argument("--no-save", action="store_true", help="Don't save to database")

    # identify
    id_p = subparsers.add_parser("identify", help="Identify unknown model")
    id_p.add_argument("model", help="Model to identify")
    id_p.add_argument("--quick", action="store_true", help="Quick scan")
    id_p.add_argument("--concurrency", type=int, default=3)
    id_p.add_argument("--top-k", type=int, default=5, help="Top K matches to show")

    # compare
    cmp_p = subparsers.add_parser("compare", help="Compare two models")
    cmp_p.add_argument("model1", help="First model")
    cmp_p.add_argument("model2", help="Second model")
    cmp_p.add_argument("--concurrency", type=int, default=3)

    # db
    db_p = subparsers.add_parser("db", help="Database management")
    db_p.add_argument("db_action", choices=["list", "seed", "stats", "export", "import"])
    db_p.add_argument("--file", help="File for export/import")

    # probes
    subparsers.add_parser("probes", help="List available probes")

    # selftest
    subparsers.add_parser("selftest", help="Run all self-tests")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "scan":
        asyncio.run(cmd_scan(args))
    elif args.command == "identify":
        asyncio.run(cmd_identify(args))
    elif args.command == "compare":
        asyncio.run(cmd_compare(args))
    elif args.command == "db":
        cmd_db(args)
    elif args.command == "probes":
        cmd_probes(args)
    elif args.command == "selftest":
        success = cmd_selftest(args)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
