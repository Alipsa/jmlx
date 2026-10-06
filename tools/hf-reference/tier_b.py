#!/usr/bin/env python3
"""Generate candidate Phase 7.2 references from pinned, locally verified Hub artifacts."""
import argparse
import hashlib
import inspect
import json
import platform
from pathlib import Path

import sentence_transformers
import torch
import transformers
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, BertForSequenceClassification, T5ForConditionalGeneration


def generate(directory, manifest_path):
    if (transformers.__version__ != "4.57.6"
            or sentence_transformers.__version__ != "5.1.2"
            or torch.__version__ not in ("2.6.0", "2.6.0+cpu")):
        raise RuntimeError("Install the reviewed hash-locked reference environment")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    manifest = json.loads(manifest_path.read_text())
    for name, spec in manifest["files"].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != spec["sha256"]:
            raise RuntimeError(f"artifact hash mismatch: {name}")
    task = manifest["task"]
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    case = {}
    if task == "embedding":
        text = "The cat sits outside on a sunny afternoon."
        model = SentenceTransformer(str(directory), device="cpu", local_files_only=True,
                                    model_kwargs={"attn_implementation": "eager"})
        model.float().eval()
        embedding = model.encode([text], convert_to_tensor=True)[0]
        case = {"text": text, "source_ids": tokenizer(text)["input_ids"],
                "embedding": embedding.tolist()}
    elif task == "classification":
        text = "This movie was delightful, beautifully acted and wonderfully entertaining."
        model = BertForSequenceClassification.from_pretrained(directory, local_files_only=True,
                                                              attn_implementation="eager").cpu().float().eval()
        with torch.no_grad():
            logits = model(**tokenizer(text, return_tensors="pt")).logits[0]
        top = torch.topk(logits, 2)
        case = {"text": text, "source_ids": tokenizer(text)["input_ids"],
                "logits": logits.tolist(), "class_id": top.indices[0].item(),
                "top_two_ids": top.indices.tolist(), "gap": (top.values[0] - top.values[1]).item()}
    elif task == "seq2seq":
        text = "translate English to German: The house is wonderful."
        model = T5ForConditionalGeneration.from_pretrained(directory, local_files_only=True,
                                                         attn_implementation="eager").cpu().float().eval()
        encoded = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            result = model.generate(**encoded, max_new_tokens=8, do_sample=False,
                                    return_dict_in_generate=True, output_scores=True)
        history = result.sequences[0].tolist()
        logits = [score[0] for score in result.scores]
        gaps = []
        top_ids = []
        for score in logits:
            top = torch.topk(score, 2)
            gaps.append((top.values[0] - top.values[1]).item())
            top_ids.append(top.indices.tolist())
        case = {"text": text, "source_ids": encoded["input_ids"][0].tolist(),
                "target_ids": history[1:], "max_new_tokens": 8,
                "histories": [history[:i + 1] for i in range(len(logits))],
                "logits": [score.tolist() for score in logits], "gaps": gaps,
                "top_two_ids": top_ids,
                "finish_reason": "EOS" if history[-1] == model.config.eos_token_id else "MAX_TOKENS"}
    else:
        raise RuntimeError(f"unsupported reference task: {task}")
    manifest["reference"] = {
        "status": "candidate: final real-checkpoint measurements pending",
        "epsilon": 1e-4, "margin": 1e-4, "prompt_special_tokens": "ADD",
        "precision": "CPU float32 eager; Java TF32 off",
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__,
                     "sentence_transformers": sentence_transformers.__version__},
        "requirements_lock_sha256": hashlib.sha256(Path(__file__).with_name("requirements.lock").read_bytes()).hexdigest(),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_sha256": {
            cls.__name__: hashlib.sha256(Path(inspect.getsourcefile(cls)).read_bytes()).hexdigest()
            for cls in (BertForSequenceClassification, T5ForConditionalGeneration,
                        sentence_transformers.models.Transformer,
                        sentence_transformers.models.Pooling,
                        sentence_transformers.models.Normalize)
        },
        "cases": [case],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(task, "reference generated", "minimum gap", min(case.get("gaps", [case.get("gap", 1)])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    generate(args.directory, args.manifest)
