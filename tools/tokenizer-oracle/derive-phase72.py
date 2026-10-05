#!/usr/bin/env python3
"""Derive oracle inputs from the SHA-pinned, locally downloaded Phase 7.2 artifacts."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).with_name("fixtures")


def derive(name, fixture_name, trim):
    manifest = json.loads((ROOT / "tools/tier-b" / f"{name}.json").read_text())
    source = ROOT / ".tier-b" / name / "tokenizer.json"
    data = source.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    assert digest == manifest["files"]["tokenizer.json"]["sha256"]
    tokenizer = json.loads(data)
    trim(tokenizer)
    path = FIXTURES / fixture_name
    path.write_text(json.dumps(tokenizer, ensure_ascii=False, separators=(",", ":")) + "\n")
    return {"repository": manifest["repository"], "revision": manifest["revision"],
            "source_sha256": digest, "source_bytes": len(data),
            "derived_bytes": path.stat().st_size, "license": "Apache-2.0",
            "derivation": "tools/tokenizer-oracle/derive-phase72.py"}


if __name__ == "__main__":
    provenance_path = Path(__file__).with_name("provenance.json")
    provenance = json.loads(provenance_path.read_text())
    sources = {}
    sources["phase72-wordpiece.tokenizer.json"] = derive(
        "bert-base-uncased-sst2", "phase72-wordpiece.tokenizer.json", lambda t: None)
    sources["phase72-precompiled.tokenizer.json"] = derive(
        "flan-t5-small", "phase72-precompiled.tokenizer.json",
        lambda t: t.update(added_tokens=[], truncation=None, padding=None))
    tokenizer = json.loads((FIXTURES / "phase72-precompiled.tokenizer.json").read_text())
    charsmap = tokenizer["normalizer"]["normalizers"][0]["precompiled_charsmap"]
    sources["phase72-precompiled.tokenizer.json"]["charsmap_base64_sha256"] = hashlib.sha256(
        charsmap.encode()).hexdigest()
    pair = json.loads((FIXTURES / "phase72-wordpiece.tokenizer.json").read_text())
    pair["truncation"] = {"strategy": "OnlySecond", "max_length": 5,
                          "direction": "Right", "stride": 0}
    path = FIXTURES / "phase72-only-second.tokenizer.json"
    path.write_text(json.dumps(pair, separators=(",", ":")) + "\n")
    sources[path.name] = {**sources["phase72-wordpiece.tokenizer.json"],
                          "derived_bytes": path.stat().st_size,
                          "modification": "OnlySecond / max_length=5 / Right / stride=0"}
    first = json.loads((FIXTURES / "phase72-wordpiece.tokenizer.json").read_text())
    first["truncation"] = {"strategy": "OnlyFirst", "max_length": 5,
                           "direction": "Right", "stride": 0}
    path = FIXTURES / "phase72-only-first.tokenizer.json"
    path.write_text(json.dumps(first, separators=(",", ":")) + "\n")
    sources[path.name] = {**sources["phase72-wordpiece.tokenizer.json"],
                          "derived_bytes": path.stat().st_size,
                          "modification": "OnlyFirst / max_length=5 / Right / stride=0"}
    replacement = json.loads((FIXTURES / "wordpiece.tokenizer.json").read_text())
    replacement["normalizer"] = {"type": "Replace", "pattern": {"Regex": r"\b"},
                                  "content": "|"}
    (FIXTURES / "phase72-replace.tokenizer.json").write_text(
        json.dumps(replacement, separators=(",", ":")) + "\n")
    provenance["phase72Derivation"] = sources
    for path in FIXTURES.glob("phase72*.json"):
        if not path.name.endswith(".expected.json"):
            provenance["fixtureSources"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
