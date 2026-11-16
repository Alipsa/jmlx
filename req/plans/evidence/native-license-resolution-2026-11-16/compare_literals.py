#!/usr/bin/env python3
"""Reproduce the recorded function comparisons and exact comment-list projections."""

import argparse
from collections import Counter
from decimal import Decimal
import hashlib
import http.client
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import urllib.request

if sys.version_info < (3, 11):
    raise SystemExit("Python 3.11+ is required")

FLOAT_LITERAL_PATTERN = r"(?<![\w.])[+-]?(?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?[fF]?"
ANSWER_IDS = {"stackoverflow-erf": 35148199, "stackoverflow-erfinv": 49743348}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def response_json(raw):
    document = json.loads(raw)
    if not isinstance(document, dict) or "error_id" in document:
        raise ValueError("unexpected Stack Exchange response")
    return document


def comment_projection(raw):
    document = response_json(raw)
    if document.get("has_more") is not False or not isinstance(document.get("items"), list):
        raise ValueError("incomplete or unexpected Stack Exchange comment list")
    records = []
    for item in document["items"]:
        if not isinstance(item, dict) or not isinstance(item.get("body"), str):
            raise ValueError("unexpected Stack Exchange comment record")
        owner = item.get("owner", {})
        if not isinstance(owner, dict):
            raise ValueError("unexpected Stack Exchange comment owner")
        records.append({"comment_id": item["comment_id"], "creation_date": item["creation_date"],
                        "body": item["body"], "ownerUserId": owner.get("user_id")})
    # Keep the API items order and raw body HTML. No trailing newline is hashed.
    return json.dumps(records, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


class CodeBlocks(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.depth = 0
        self.blocks = []
        self.current = []

    def handle_starttag(self, tag, attrs):
        if tag == "code":
            if not self.depth:
                self.current = []
            self.depth += 1

    def handle_endtag(self, tag):
        if tag == "code" and self.depth:
            self.depth -= 1
            if not self.depth:
                self.blocks.append("".join(self.current))

    def handle_data(self, text):
        if self.depth:
            self.current.append(text)


def answer_code(raw, expected_body_digest):
    document = response_json(raw)
    items = document.get("items")
    if (not isinstance(items, list) or len(items) != 1 or not isinstance(items[0], dict)
            or not isinstance(items[0].get("body"), str)):
        raise ValueError("unexpected Stack Exchange answer")
    body = items[0]["body"]
    if digest(body.encode("utf-8")) != expected_body_digest:
        raise ValueError("Stack Overflow answer body differs from native audit")
    parser = CodeBlocks()
    parser.feed(body)
    if not parser.blocks:
        raise ValueError("Stack Overflow answer has no code block")
    return max(parser.blocks, key=len).encode("utf-8")


def function_text(text, name):
    # Deliberately bounded to the captured inputs, not a general C++ parser.
    match = re.search(r"\b" + re.escape(name) + r"\s*\([^)]*\)\s*\{", text)
    if not match:
        raise ValueError(f"function definition missing: {name}")
    start = text.index("{", match.start())
    depth, end = 1, start + 1
    while depth and end < len(text):
        if text[end] == "{":
            depth += 1
        elif text[end] == "}":
            depth -= 1
        end += 1
    if depth:
        raise ValueError(f"unbalanced function: {name}")
    return text[match.start():end]


def literals(text):
    text = re.sub(r"/\*.*?\*/|//[^\n]*", "", text, flags=re.S)
    return [str(Decimal(token.rstrip("fF"))) for token in re.findall(FLOAT_LITERAL_PATTERN, text)]


def read_source(url, expected_digest, project, inputs, fetch_missing):
    if not isinstance(url, str):
        raise ValueError("source url must be a string")
    cached = inputs / (digest(url.encode("utf-8")) + "-" + expected_digest)
    fetched = not cached.is_file()
    if fetched:
        if not fetch_missing:
            raise ValueError(f"missing input {cached}; use --fetch-missing to read {url}")
        fetch_url = url.replace("https://github.com/", "https://raw.githubusercontent.com/", 1)
        fetch_url = fetch_url.replace("/blob/", "/", 1)
        with urllib.request.urlopen(fetch_url, timeout=45) as stream:
            raw = stream.read()
    else:
        raw = cached.read_bytes()
    data = project(raw)
    if digest(data) != expected_digest:
        raise ValueError(f"projected source digest differs: {url}")
    if fetched:
        inputs.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=inputs, delete=False) as stream:
            temporary = Path(stream.name)
        try:
            temporary.write_bytes(raw)
            os.replace(temporary, cached)
        finally:
            temporary.unlink(missing_ok=True)
    return data


def comparison_checks(records):
    values = {record["id"]: record["decimalFloatingLiterals"] for record in records}
    return {
        "metalInverseLiteralSequenceMatchesStackOverflow": values["mlx-metal-erfinv"] == values["stackoverflow-erfinv"],
        "cpuInverseLiteralMultisetMatchesStackOverflow": Counter(values["mlx-cpu-erfinv"]) == Counter(values["stackoverflow-erfinv"]),
        "metalErfLiteralMultisetIsSubsetOfStackOverflow": not (Counter(values["mlx-metal-erf"]) - Counter(values["stackoverflow-erf"])),
        "bsdErfLiteralSequenceMatchesShipped": values["bsd-forum-erf"] == values["mlx-metal-erf"],
        "bsdInverseLiteralSequenceMatchesShipped": values["bsd-forum-erfinv"] == values["mlx-metal-erfinv"],
    }


def verify(root, inputs, fetch_missing=False):
    folder = Path(__file__).resolve().parent
    research = json.loads((folder / "research.json").read_bytes())
    expected = json.loads((folder / "code-comparison.json").read_bytes())
    audit = json.loads((root / "req/phase8-4-native-license-audit.json").read_bytes())
    audit_sources = {entry["source"]: entry["sha256"] for entry in audit["sourceEvidence"]}
    sources = research["sources"]
    if not isinstance(sources, list):
        raise ValueError("research sources must be a list")
    for source in sources:
        if not isinstance(source, dict) or not isinstance(source.get("id"), str):
            raise ValueError("research source must have a string id")
    records, comments = [], []
    for entry in expected["records"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            raise ValueError("comparison record must have a string id")
        if entry["id"].startswith("bsd-forum-"):
            source = next((item for item in sources if item["id"] == entry["id"]), None)
            if source is None:
                raise ValueError(f"research source missing for comparison record: {entry['id']}")
            data = (root / source["retainedPath"]).read_bytes()
            if digest(data) != source["sha256"] or digest(data) != entry["inputSha256"]:
                raise ValueError(f"retained BSD sample differs: {entry['id']}")
        elif entry["id"] in ANSWER_IDS:
            answer = ANSWER_IDS[entry["id"]]
            url = f"https://api.stackexchange.com/2.3/answers/{answer}?site=stackoverflow&filter=withbody"
            data = read_source(url, entry["inputSha256"],
                               lambda raw: answer_code(raw, audit_sources[url]), inputs, fetch_missing)
        else:
            data = read_source(entry["source"], entry["inputSha256"], lambda raw: raw, inputs, fetch_missing)
        function = function_text(data.decode("utf-8"), entry["function"])
        actual = dict(entry, functionSha256=digest(function.encode("utf-8")),
                      decimalFloatingLiterals=literals(function))
        if actual != entry:
            raise ValueError(f"function comparison record differs: {entry['id']}")
        records.append(actual)
    checks = comparison_checks(records)
    if checks != expected["checks"]:
        raise ValueError("literal comparison results differ from the recorded five checks")
    for entry in sources:
        if entry["id"].endswith("-answer-comments"):
            data = read_source(entry["url"], entry["sha256"], comment_projection, inputs, fetch_missing)
            count = len(json.loads(data))
            if count != entry["count"]:
                raise ValueError(f"comment count differs: {entry['id']}")
            comments.append({"id": entry["id"], "sha256": digest(data), "count": count})
    return {"schema": 1, "matchesRecordedResults": True, "functionRecords": len(records),
            "checks": checks, "commentProjections": comments}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--fetch-missing", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.root, args.inputs, args.fetch_missing)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, TypeError, http.client.HTTPException) as error:
        raise SystemExit(f"license comparison: {error}") from error
