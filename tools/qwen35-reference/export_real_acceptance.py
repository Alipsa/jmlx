#!/usr/bin/env python3
"""Fix real-artifact bounds, prompt/margin checks and stock logits before Java acceptance."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, Qwen3_5Config, Qwen3_5ForConditionalGeneration


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8*1024*1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", choices=("hf-08b", "mlx-27b"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--captures", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--diagnostic", type=Path)
    parser.add_argument("--select-cases", default="text,image")
    args = parser.parse_args()
    if args.manifest.exists():
        raise ValueError("Preserve the previous declaration before intentionally replacing it")
    root = Path(__file__).resolve().parent
    repo = root.parent.parent
    hf = args.artifact == "hf-08b"
    prefix = "08b" if hf else "27b"
    inventory = json.loads((root/"evidence"/(args.artifact+"-inventory.json")).read_text())
    inputs_file = (args.inputs or root/"evidence"/(prefix+"-inputs.json")).resolve()
    inputs = json.loads(inputs_file.read_text())
    reference_file = (args.reference or root/"evidence"/("08b-hf.json" if hf else "27b-float32-stock.json")).resolve()
    reference = json.loads(reference_file.read_text())
    assert digest(inputs_file) == reference["inputs_sha256"]
    capture = root/"evidence/08b-hf.npz" if hf else args.captures/reference["captures"]["file"]
    expected_hash = reference["logits_sha256"] if hf else reference["captures"]["sha256"]
    assert digest(capture) == expected_hash
    assert digest(args.model/"config.json") == inventory["files"]["config.json"]["sha256"]
    assert digest(args.model/"tokenizer.json") == inventory["files"]["tokenizer.json"]["sha256"]
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    configuration = Qwen3_5Config.from_pretrained(args.model, local_files_only=True)
    configuration._attn_implementation = "eager"
    # Official constructor with meta tensors: no model weights or reference math are replaced.
    with torch.device("meta"):
        position_model = Qwen3_5ForConditionalGeneration(configuration)
    stock = np.load(capture, allow_pickle=False)
    aligned = None
    diagnostic_file = (args.diagnostic or root/"evidence/27b-float32-diagnostic.json").resolve()
    if not hf:
        diagnostic = json.loads(diagnostic_file.read_text())
        diagnostic_capture = args.captures/diagnostic["captures"]["file"]
        assert digest(diagnostic_capture) == diagnostic["captures"]["sha256"]
        assert reference["floating_parameter_upcast"] and diagnostic["floating_parameter_upcast"]
        aligned = np.load(diagnostic_capture, allow_pickle=False)
    vectors, cases, attribution = {}, [], []
    normalization_max = 0.0
    sources = {item["name"]: item for item in inputs["cases"]}
    references = {item["name"]: item for item in reference["cases"]}
    for selected in args.select_cases.split(","):
        source, ref = sources[selected], references[selected]
        assert source["name"] == ref["name"]
        name = source["name"]
        suffix = "" if hf else "_logits"
        phase_names = [name+"_prefill"]+[name+f"_decode{i}" for i in range(8)]
        case_vectors, contributions, gaps, ids = [], [], [], []
        for phase in phase_names:
            array = stock[phase+suffix]
            assert array.dtype == np.float32 and array.shape == (1, 248320)
            vectors[phase] = array
            case_vectors.append(phase)
            order = np.sort(array[0])
            gaps.append(float(order[-1]-order[-2]))
            ids.append(int(array.argmax()))
            if aligned is not None:
                contribution = float(np.max(np.abs(array-aligned[phase+suffix])))
                normalization_max = max(normalization_max, contribution)
                contributions.append(contribution)
                attribution.append(dict(phase=phase, normalization_contribution_max_abs=contribution,
                                        stock_greedy=int(array.argmax()), diagnostic_greedy=int(aligned[phase+suffix].argmax())))
        history = [s["input_id"] for s in ref["decode" if hf else "decode_steps"]]
        assert history == ids[:8]
        expanded = source["input_ids"][0]
        original = []
        for token in expanded:
            if token != 248056 or not original or original[-1] != token:
                original.append(token)
        rendered = source["rendered_prompt"]
        original_rendered = rendered
        sentinel = "<|image_pad|>"
        while sentinel+sentinel in original_rendered:
            original_rendered = original_rendered.replace(sentinel+sentinel, sentinel)
        position_ids, delta = position_model.model.get_rope_index(
            torch.tensor([expanded]), mm_token_type_ids=(torch.tensor([expanded]) == 248056).long(),
            image_grid_thw=torch.tensor(source["image_grid_thw"]) if source.get("image_grid_thw") else None)
        cases.append(dict(name=name, original_ids=original, expanded_ids=expanded,
                          rendered_prompt=original_rendered, expanded_rendered_prompt=rendered,
                          image=name == "image", logits=case_vectors, greedy_ids=ids[:8],
                          gaps=gaps, normalization_contributions=contributions,
                          reference_decoded=ref.get("decoded"), positions=position_ids[:, 0].tolist(),
                          position_delta=int(delta.item()),
                          expected_text=tokenizer.decode(ids[:8], skip_special_tokens=True)))
        if "user_content" in source:
            cases[-1]["user_content"] = source["user_content"]
    # Real bound is independent of strict tiny 1e-4; HF itself measured 3.19e-4 cached/fresh.
    arithmetic = .001
    epsilon = arithmetic+normalization_max
    margin = .0001
    assert all(gap > 2*epsilon+margin for case in cases for gap in case["gaps"]), "Preserve ineligible cases; select an eligible reference prompt before Java"
    target = root/"goldens"/f"real-{args.artifact}-stock-logits.npz"
    np.savez_compressed(target, **vectors)  # Unmodified stock only: no diagnostic goldens.
    image = "tools/image-oracle/fast-comparison/fixtures/model-rgb-256x256.png"
    repository, revision = ("Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17") if hf else (
        "mlx-community/Qwen3.8-27B-4bit", "10c35caafbb80f7dc6a7a432cdd11af10a6d4818")
    files = {name: dict(sha256=spec["sha256"], max_bytes=spec["bytes"]) for name, spec in inventory["files"].items()}
    record = dict(repository=repository, revision=revision, task="qwen35-generation", license="Apache-2.0",
        license_evidence=f"https://huggingface.co/{repository}/blob/{revision}/README.md",
        max_total_bytes=sum(spec["max_bytes"] for spec in files.values()), files=files,
        recorded_pin="mlxMetalVersion=0.32.4\nmlxcCommit=a26f2d584a14ec619ae8db9a09be03becd47598a",
        resource_limits=dict(native_peak_bytes=8_000_000_000 if hf else 32_000_000_000,
                             sampled_phys_footprint_bytes=12_000_000_000 if hf else 48_000_000_000,
                             basis="predeclared storage/upcast and transient budget; regression gate, not a guarantee against OS OOM"),
        images=[dict(path=image, sha256=digest(repo/image), license="MIT", provenance="Existing digest-pinned synthetic project RGB fixture")],
        reference=dict(status="predeclared before Java acceptance; source bounds never inferred from Java",
            epsilon=epsilon, arithmetic_epsilon=arithmetic, margin=margin,
            normalization_contribution_max_abs=normalization_max,
            precision="HF CPU FLOAT32" if hf else "unmodified mlx-vlm 0.6.8 GPU, floating parameters explicitly FLOAT32; packed U32 unchanged",
            logits_file=dict(path=str(target.relative_to(repo)), sha256=digest(target)),
            source_record=dict(path=str(reference_file.relative_to(repo)), sha256=digest(reference_file)),
            input_record=dict(path=str(inputs_file.relative_to(repo)), sha256=digest(inputs_file)),
            generator_sha256=digest(__file__), cases=cases,
            raw_row_budget=4096, effective_max_pixels=1048576,
            normalization_attribution=attribution))
    if not hf:
        record["reference"]["diagnostic_record"] = dict(path=str(diagnostic_file.relative_to(repo)), sha256=digest(diagnostic_file), role="attribution only; never goldens")
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(record, indent=2)+"\n")
    print(args.manifest, "epsilon", epsilon, "minimum gap", min(g for c in cases for g in c["gaps"]))


if __name__ == "__main__":
    main()
