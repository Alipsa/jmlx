#!/usr/bin/env python3
"""Independent unmodified HF hybrid stack text logits, state and cached/fresh forwards."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import Qwen3_5ForConditionalGeneration
from transformers.models.qwen3_5 import modeling_qwen3_5 as hf
from probe_hf import tiny_config


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(20261104)
    root = Path(__file__).resolve().parent
    arrays, cases, fixtures = {}, [], []
    with torch.no_grad():
        for name in ("hf", "hf-tied-equal", "hf-eight"):
            directory = root / "fixtures" / name
            if name != "hf":
                if directory.exists():
                    raise ValueError(f"Refusing to overwrite variant checkpoint: {directory}")
                config = tiny_config()
                if name == "hf-tied-equal":
                    config.tie_word_embeddings = True
                    config.text_config.tie_word_embeddings = True
                    config.text_config.linear_num_value_heads = config.text_config.linear_num_key_heads
                else:
                    config.text_config.num_hidden_layers = 8
                    config.text_config.layer_types = ["full_attention" if (i + 1) % 4 == 0 else
                                                       "linear_attention" for i in range(8)]
                model = Qwen3_5ForConditionalGeneration(config).eval().bfloat16().float()
                model.save_pretrained(directory)
                for filename in ("tokenizer.json", "tokenizer_config.json", "preprocessor_config.json",
                                 "processor_config.json", "chat_template.jinja"):
                    source = root / "fixtures/hf" / filename
                    if source.exists():
                        shutil.copyfile(source, directory / filename)
            model = Qwen3_5ForConditionalGeneration.from_pretrained(
                directory, dtype=torch.float32, attn_implementation="eager").eval()
            text = model.config.text_config
            fixtures.append(dict(name=name, files={p.name: digest(p) for p in sorted(directory.iterdir())
                                                    if p.is_file()}))
            for length in (4, 65, 129):
                stem = f"{name}.t{length}"
                ids = torch.tensor([[1, 3, 4, 5]] if length == 4 else
                                   [[1] + [3+i % 90 for i in range(length-1)]], dtype=torch.int32)
                output = model(input_ids=ids, use_cache=True)
                cache = output.past_key_values
                arrays[stem+".ids"] = ids
                arrays[stem+".logits"] = output.logits

                def capture(prefix):
                    for i, kind in enumerate(text.layer_types):
                        if kind == "linear_attention":
                            arrays[f"{prefix}.layer{i}.state"] = cache.get_recurrent_state(i).clone()
                            arrays[f"{prefix}.layer{i}.convolution"] = cache.get_conv_state(i)[:, :, -(text.linear_conv_kernel_dim-1):].transpose(1, 2).clone()
                        else:
                            arrays[f"{prefix}.layer{i}.keys"] = cache.layers[i].keys.clone()
                            arrays[f"{prefix}.layer{i}.values"] = cache.layers[i].values.clone()

                capture(stem)
                fresh_error = 0.0
                for step, token in enumerate((6, 7)):
                    prefix = stem+f".decode{step}"
                    decode_ids = torch.tensor([[token]], dtype=torch.int32)
                    decoded = model(input_ids=decode_ids, past_key_values=cache, use_cache=True)
                    arrays[prefix+".ids"] = decode_ids
                    arrays[prefix+".logits"] = decoded.logits
                    ids = torch.cat((ids, decode_ids), dim=1)
                    fresh = model(input_ids=ids, use_cache=False).logits[:, -1:]
                    arrays[prefix+".fresh_logits"] = fresh
                    fresh_error = max(fresh_error, (fresh-decoded.logits).abs().max().item())
                    capture(prefix)
                assert fresh_error < 1e-4, (stem, fresh_error)
                cases.append(dict(name=stem, fixture=name, length=length,
                                  cached_fresh_max_abs=fresh_error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().contiguous().clone() for k, v in arrays.items()}, args.output,
              metadata={"reference": "unmodified HF 5.22.1 CPU FLOAT32; BF16-rounded weights"})
    args.output.with_suffix(".json").write_text(json.dumps(dict(
        kind="HF text acceptance goldens; absolute epsilon 1e-4 fixed before Java comparison",
        source_sha256=digest(__file__), hf_source_sha256=digest(hf.__file__),
        safetensors_sha256=digest(args.output), seed=20261104, cases=cases, fixtures=fixtures), indent=2)+"\n")
    print(args.output)


if __name__ == "__main__":
    main()
