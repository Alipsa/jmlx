#!/usr/bin/env python3
"""Prepare independent real 27B prompt candidates; no Java measurements inform selection."""
import argparse
import hashlib
import json
from pathlib import Path
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    original = json.loads((root/"evidence/27b-inputs.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    prompts = [
        ("text_copy", "Return exactly: one two three four five six seven eight nine ten."),
        ("text_count", "Count from one to twenty, using words separated by spaces."),
        ("text_sky", "Explain why the sky is blue in three sentences."),
        ("text_planet", "Write three sentences about the planet Earth."),
        ("text_repeat", "Repeat the word hello ten times, separated by spaces."),
        ("text_digits", "Return exactly the sequence: 1 2 3 4 5 6 7 8 9 10."),
    ]
    cases = [original["cases"][0]]
    for name, content in prompts:
        rendered = tokenizer.apply_chat_template([dict(role="user", content=content)],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        cases.append(dict(name=name, user_content=content, rendered_prompt=rendered,
                          input_ids=[tokenizer(rendered, add_special_tokens=False)["input_ids"]]))
    cases.append(original["cases"][1])
    args.output.write_text(json.dumps(dict(kind="reference-only candidates; original failed prompt retained",
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), cases=cases), indent=2)+"\n")


if __name__ == "__main__":
    main()
