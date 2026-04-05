#!/usr/bin/env python3
"""
🐉 M0D3L_F1NG3RPR1NT — Fingerprint Database

Stores, retrieves, and manages reference fingerprints for known models.
JSON-based file storage — no external dependencies.

Like a criminal database, but for AI models that lie about their identity.

Built by Pliny the Forgemaster 🔥
"""

import json
import os
import sys
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class FingerprintDatabase:
    """
    File-based fingerprint database. Each model gets a JSON file
    in the reference_prints/ directory.
    """

    def __init__(self, db_dir: str = "reference_prints"):
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.db_dir / "_index.json"
        self._index = self._load_index()

    def _load_index(self) -> dict:
        """Load or create the database index."""
        if self._index_path.exists():
            with open(self._index_path) as f:
                return json.load(f)
        return {"models": {}, "last_updated": None, "version": 1}

    def _save_index(self):
        """Save the database index."""
        self._index["last_updated"] = datetime.now(timezone.utc).isoformat()
        with open(self._index_path, "w") as f:
            json.dump(self._index, f, indent=2)

    def _model_filename(self, model_id: str) -> str:
        """Generate a safe filename from a model ID."""
        safe = model_id.replace("/", "__").replace(" ", "_")
        return f"{safe}.json"

    def store(self, fingerprint: dict, overwrite: bool = False) -> str:
        """
        Store a fingerprint in the database.

        Args:
            fingerprint: Fingerprint dict (from Fingerprint.to_dict())
            overwrite: If True, replace existing. If False, append as version.

        Returns:
            Path to stored file.
        """
        model_id = fingerprint.get("model_id", "unknown")
        filename = self._model_filename(model_id)
        filepath = self.db_dir / filename

        if filepath.exists() and not overwrite:
            # Versioned storage — append timestamp
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = self._model_filename(f"{model_id}_{ts}")
            filepath = self.db_dir / filename

        # Store the fingerprint
        with open(filepath, "w") as f:
            json.dump(fingerprint, f, indent=2)

        # Update index
        if model_id not in self._index["models"]:
            self._index["models"][model_id] = {
                "files": [],
                "first_seen": datetime.now(timezone.utc).isoformat(),
                "last_updated": None,
            }

        entry = self._index["models"][model_id]
        entry["files"].append(filename)
        entry["last_updated"] = datetime.now(timezone.utc).isoformat()
        entry["latest_hash"] = fingerprint.get("fingerprint_hash", "")
        self._save_index()

        return str(filepath)

    def load(self, model_id: str, version: str = "latest") -> Optional[dict]:
        """
        Load a fingerprint from the database.

        Args:
            model_id: Model identifier (e.g., "openai/gpt-4o")
            version: "latest" for most recent, or specific filename

        Returns:
            Fingerprint dict or None if not found.
        """
        if model_id not in self._index["models"]:
            return None

        files = self._index["models"][model_id]["files"]
        if not files:
            return None

        if version == "latest":
            filename = files[-1]
        else:
            filename = version

        filepath = self.db_dir / filename
        if not filepath.exists():
            return None

        with open(filepath) as f:
            return json.load(f)

    def load_all_latest(self) -> list[dict]:
        """Load the latest fingerprint for every model in the database."""
        results = []
        for model_id in self._index["models"]:
            fp = self.load(model_id)
            if fp:
                results.append(fp)
        return results

    def list_models(self) -> list[dict]:
        """List all models in the database with summary info."""
        models = []
        for model_id, info in self._index["models"].items():
            models.append({
                "model_id": model_id,
                "versions": len(info["files"]),
                "first_seen": info.get("first_seen", "?"),
                "last_updated": info.get("last_updated", "?"),
                "latest_hash": info.get("latest_hash", "?"),
            })
        return models

    def get_history(self, model_id: str) -> list[dict]:
        """Get all fingerprint versions for a model (for drift analysis)."""
        if model_id not in self._index["models"]:
            return []

        history = []
        for filename in self._index["models"][model_id]["files"]:
            filepath = self.db_dir / filename
            if filepath.exists():
                with open(filepath) as f:
                    fp = json.load(f)
                    history.append({
                        "filename": filename,
                        "timestamp": fp.get("timestamp", "?"),
                        "hash": fp.get("fingerprint_hash", "?"),
                        "fingerprint": fp,
                    })
        return history

    def delete(self, model_id: str) -> bool:
        """Delete all fingerprints for a model."""
        if model_id not in self._index["models"]:
            return False

        for filename in self._index["models"][model_id]["files"]:
            filepath = self.db_dir / filename
            if filepath.exists():
                filepath.unlink()

        del self._index["models"][model_id]
        self._save_index()
        return True

    def stats(self) -> dict:
        """Database statistics."""
        total_files = sum(
            len(info["files"]) for info in self._index["models"].values()
        )
        total_size = sum(
            (self.db_dir / f).stat().st_size
            for info in self._index["models"].values()
            for f in info["files"]
            if (self.db_dir / f).exists()
        )
        return {
            "models": len(self._index["models"]),
            "total_fingerprints": total_files,
            "total_size_bytes": total_size,
            "db_dir": str(self.db_dir),
            "last_updated": self._index.get("last_updated"),
        }

    def export_all(self, output_path: str) -> str:
        """Export entire database to a single JSON file."""
        export = {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "version": 1,
            "models": {},
        }
        for model_id in self._index["models"]:
            fp = self.load(model_id)
            if fp:
                export["models"][model_id] = fp

        with open(output_path, "w") as f:
            json.dump(export, f, indent=2)
        return output_path

    def import_from(self, input_path: str) -> int:
        """Import fingerprints from an exported JSON file. Returns count imported."""
        with open(input_path) as f:
            data = json.load(f)

        count = 0
        for model_id, fp in data.get("models", {}).items():
            self.store(fp)
            count += 1
        return count


# ═══════════════════════════════════════════════════════════════
# SEED DATABASE — Built-in approximate fingerprints for common models
# ═══════════════════════════════════════════════════════════════

SEED_FINGERPRINTS = {
    "openai/gpt-4o": {
        "model_id": "openai/gpt-4o",
        "model_name": "gpt-4o",
        "timestamp": "2026-04-04T00:00:00Z",
        "fingerprint_hash": "seed_gpt4o",
        "dimensions": {
            "refusal_rate": {"dimension": "refusal_rate", "score": 0.35, "raw_value": 0.35, "confidence": 0.7, "evidence": []},
            "refusal_style": {"dimension": "refusal_style", "score": 0.4, "raw_value": 2, "confidence": 0.7, "evidence": ["policy_citation", "apologetic"]},
            "verbosity": {"dimension": "verbosity", "score": 0.55, "raw_value": 0.55, "confidence": 0.7, "evidence": []},
            "hedging": {"dimension": "hedging", "score": 0.45, "raw_value": 0.45, "confidence": 0.7, "evidence": []},
            "formality": {"dimension": "formality", "score": 0.65, "raw_value": 0.65, "confidence": 0.7, "evidence": []},
            "emoji_usage": {"dimension": "emoji_usage", "score": 0.05, "raw_value": 0.05, "confidence": 0.7, "evidence": []},
            "structure": {"dimension": "structure", "score": 0.55, "raw_value": 0.55, "confidence": 0.7, "evidence": []},
            "response_speed": {"dimension": "response_speed", "score": 0.35, "raw_value": 1200, "confidence": 0.5, "evidence": []},
            "length_variance": {"dimension": "length_variance", "score": 0.30, "raw_value": 0.30, "confidence": 0.6, "evidence": []},
            "refusal_selectivity": {"dimension": "refusal_selectivity", "score": 0.55, "raw_value": 0.55, "confidence": 0.6, "evidence": []},
        },
        "probe_count": 0,
        "metadata": {"source": "seed", "note": "Approximate behavioral profile based on known characteristics"},
    },
    "anthropic/claude-3.5-sonnet": {
        "model_id": "anthropic/claude-3.5-sonnet",
        "model_name": "claude-3.5-sonnet",
        "timestamp": "2026-04-04T00:00:00Z",
        "fingerprint_hash": "seed_claude35s",
        "dimensions": {
            "refusal_rate": {"dimension": "refusal_rate", "score": 0.30, "raw_value": 0.30, "confidence": 0.7, "evidence": []},
            "refusal_style": {"dimension": "refusal_style", "score": 0.2, "raw_value": 2, "confidence": 0.7, "evidence": ["apologetic", "redirect"]},
            "verbosity": {"dimension": "verbosity", "score": 0.70, "raw_value": 0.70, "confidence": 0.7, "evidence": []},
            "hedging": {"dimension": "hedging", "score": 0.65, "raw_value": 0.65, "confidence": 0.7, "evidence": []},
            "formality": {"dimension": "formality", "score": 0.60, "raw_value": 0.60, "confidence": 0.7, "evidence": []},
            "emoji_usage": {"dimension": "emoji_usage", "score": 0.0, "raw_value": 0.0, "confidence": 0.8, "evidence": []},
            "structure": {"dimension": "structure", "score": 0.50, "raw_value": 0.50, "confidence": 0.7, "evidence": []},
            "response_speed": {"dimension": "response_speed", "score": 0.45, "raw_value": 1500, "confidence": 0.5, "evidence": []},
            "length_variance": {"dimension": "length_variance", "score": 0.25, "raw_value": 0.25, "confidence": 0.6, "evidence": []},
            "refusal_selectivity": {"dimension": "refusal_selectivity", "score": 0.50, "raw_value": 0.50, "confidence": 0.6, "evidence": []},
        },
        "probe_count": 0,
        "metadata": {"source": "seed", "note": "Approximate behavioral profile — verbose, hedging, no emoji, apologetic refusals"},
    },
    "google/gemini-2.5-pro": {
        "model_id": "google/gemini-2.5-pro",
        "model_name": "gemini-2.5-pro",
        "timestamp": "2026-04-04T00:00:00Z",
        "fingerprint_hash": "seed_gemini25p",
        "dimensions": {
            "refusal_rate": {"dimension": "refusal_rate", "score": 0.25, "raw_value": 0.25, "confidence": 0.7, "evidence": []},
            "refusal_style": {"dimension": "refusal_style", "score": 0.6, "raw_value": 2, "confidence": 0.7, "evidence": ["harm_flagging", "policy_citation"]},
            "verbosity": {"dimension": "verbosity", "score": 0.75, "raw_value": 0.75, "confidence": 0.7, "evidence": []},
            "hedging": {"dimension": "hedging", "score": 0.35, "raw_value": 0.35, "confidence": 0.7, "evidence": []},
            "formality": {"dimension": "formality", "score": 0.50, "raw_value": 0.50, "confidence": 0.7, "evidence": []},
            "emoji_usage": {"dimension": "emoji_usage", "score": 0.12, "raw_value": 0.12, "confidence": 0.7, "evidence": []},
            "structure": {"dimension": "structure", "score": 0.70, "raw_value": 0.70, "confidence": 0.7, "evidence": []},
            "response_speed": {"dimension": "response_speed", "score": 0.55, "raw_value": 2000, "confidence": 0.5, "evidence": []},
            "length_variance": {"dimension": "length_variance", "score": 0.40, "raw_value": 0.40, "confidence": 0.6, "evidence": []},
            "refusal_selectivity": {"dimension": "refusal_selectivity", "score": 0.40, "raw_value": 0.40, "confidence": 0.6, "evidence": []},
        },
        "probe_count": 0,
        "metadata": {"source": "seed", "note": "Verbose, structured, moderate emoji, harm-flagging refusals"},
    },
    "meta-llama/llama-3.1-70b-instruct": {
        "model_id": "meta-llama/llama-3.1-70b-instruct",
        "model_name": "llama-3.1-70b-instruct",
        "timestamp": "2026-04-04T00:00:00Z",
        "fingerprint_hash": "seed_llama31_70b",
        "dimensions": {
            "refusal_rate": {"dimension": "refusal_rate", "score": 0.15, "raw_value": 0.15, "confidence": 0.7, "evidence": []},
            "refusal_style": {"dimension": "refusal_style", "score": 0.8, "raw_value": 1, "confidence": 0.7, "evidence": ["capability_denial"]},
            "verbosity": {"dimension": "verbosity", "score": 0.50, "raw_value": 0.50, "confidence": 0.7, "evidence": []},
            "hedging": {"dimension": "hedging", "score": 0.20, "raw_value": 0.20, "confidence": 0.7, "evidence": []},
            "formality": {"dimension": "formality", "score": 0.40, "raw_value": 0.40, "confidence": 0.7, "evidence": []},
            "emoji_usage": {"dimension": "emoji_usage", "score": 0.02, "raw_value": 0.02, "confidence": 0.8, "evidence": []},
            "structure": {"dimension": "structure", "score": 0.35, "raw_value": 0.35, "confidence": 0.7, "evidence": []},
            "response_speed": {"dimension": "response_speed", "score": 0.25, "raw_value": 800, "confidence": 0.5, "evidence": []},
            "length_variance": {"dimension": "length_variance", "score": 0.45, "raw_value": 0.45, "confidence": 0.6, "evidence": []},
            "refusal_selectivity": {"dimension": "refusal_selectivity", "score": 0.30, "raw_value": 0.30, "confidence": 0.6, "evidence": []},
        },
        "probe_count": 0,
        "metadata": {"source": "seed", "note": "Low refusal, terse, minimal hedging, capability-denial refusal style"},
    },
    "x-ai/grok-3": {
        "model_id": "x-ai/grok-3",
        "model_name": "grok-3",
        "timestamp": "2026-04-04T00:00:00Z",
        "fingerprint_hash": "seed_grok3",
        "dimensions": {
            "refusal_rate": {"dimension": "refusal_rate", "score": 0.10, "raw_value": 0.10, "confidence": 0.6, "evidence": []},
            "refusal_style": {"dimension": "refusal_style", "score": 0.9, "raw_value": 1, "confidence": 0.6, "evidence": ["redirect"]},
            "verbosity": {"dimension": "verbosity", "score": 0.55, "raw_value": 0.55, "confidence": 0.6, "evidence": []},
            "hedging": {"dimension": "hedging", "score": 0.15, "raw_value": 0.15, "confidence": 0.6, "evidence": []},
            "formality": {"dimension": "formality", "score": 0.30, "raw_value": 0.30, "confidence": 0.6, "evidence": []},
            "emoji_usage": {"dimension": "emoji_usage", "score": 0.10, "raw_value": 0.10, "confidence": 0.6, "evidence": []},
            "structure": {"dimension": "structure", "score": 0.40, "raw_value": 0.40, "confidence": 0.6, "evidence": []},
            "response_speed": {"dimension": "response_speed", "score": 0.30, "raw_value": 1000, "confidence": 0.5, "evidence": []},
            "length_variance": {"dimension": "length_variance", "score": 0.50, "raw_value": 0.50, "confidence": 0.5, "evidence": []},
            "refusal_selectivity": {"dimension": "refusal_selectivity", "score": 0.25, "raw_value": 0.25, "confidence": 0.5, "evidence": []},
        },
        "probe_count": 0,
        "metadata": {"source": "seed", "note": "Very low refusal, casual, minimal hedging, redirect refusal style"},
    },
}


def seed_database(db: FingerprintDatabase) -> int:
    """Seed the database with built-in approximate fingerprints."""
    count = 0
    for model_id, fp in SEED_FINGERPRINTS.items():
        if model_id not in {m["model_id"] for m in db.list_models()}:
            db.store(fp)
            count += 1
    return count


# ═══════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════

def self_test():
    """Test database operations."""
    import tempfile
    import shutil

    print("🐉 M0D3L_F1NG3RPR1NT — Database Self-Test")
    print("=" * 50)

    # Create temp directory for test database
    tmpdir = tempfile.mkdtemp(prefix="m0d3l_db_test_")
    try:
        db = FingerprintDatabase(db_dir=tmpdir)
        print(f"\n📁 Test DB at: {tmpdir}")

        # Seed
        seeded = seed_database(db)
        print(f"\n🌱 Seeded {seeded} reference fingerprints")

        # List models
        models = db.list_models()
        print(f"\n📋 Models in database: {len(models)}")
        for m in models:
            print(f"  • {m['model_id']} (v{m['versions']}, hash: {m['latest_hash'][:8]})")

        # Load one
        fp = db.load("openai/gpt-4o")
        assert fp is not None, "Failed to load GPT-4o fingerprint"
        print(f"\n📦 Loaded GPT-4o fingerprint: {fp['fingerprint_hash']}")
        print(f"  Dimensions: {len(fp['dimensions'])}")

        # Load all
        all_fps = db.load_all_latest()
        print(f"\n📦 Loaded all latest: {len(all_fps)} fingerprints")

        # Stats
        stats = db.stats()
        print(f"\n📊 DB Stats:")
        for k, v in stats.items():
            print(f"  {k}: {v}")

        # Export
        export_path = os.path.join(tmpdir, "export.json")
        db.export_all(export_path)
        export_size = os.path.getsize(export_path)
        print(f"\n📤 Exported to {export_path} ({export_size} bytes)")

        # Import into fresh DB
        db2 = FingerprintDatabase(db_dir=os.path.join(tmpdir, "db2"))
        imported = db2.import_from(export_path)
        print(f"📥 Imported {imported} fingerprints into fresh DB")

        # History
        history = db.get_history("openai/gpt-4o")
        print(f"\n📜 GPT-4o history: {len(history)} version(s)")

        # Delete
        deleted = db.delete("x-ai/grok-3")
        remaining = len(db.list_models())
        print(f"\n🗑  Deleted Grok-3: {deleted}")
        print(f"  Remaining models: {remaining}")

        print("\n✅ Database self-test PASSED")
        return True

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    if "--self-test" in sys.argv or len(sys.argv) == 1:
        self_test()
    elif "--seed" in sys.argv:
        db = FingerprintDatabase()
        count = seed_database(db)
        print(f"Seeded {count} fingerprints")
    else:
        print("Usage: python d4t4b4s3.py [--self-test | --seed]")
