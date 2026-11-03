#!/usr/bin/env python3
"""WP0 smoke/config probes; these results are not acceptance goldens."""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
import platform
from pathlib import Path

import torch
from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
from transformers.models.qwen3_5 import modeling_qwen3_5 as impl


def tiny_config():
    return Qwen3_5Config(
        text_config=dict(
            vocab_size=128, hidden_size=128, intermediate_size=256,
            num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
            head_dim=64, linear_num_key_heads=2, linear_num_value_heads=6,
            linear_key_head_dim=32, linear_value_head_dim=64,
            linear_conv_kernel_dim=4, max_position_embeddings=4096,
            rope_parameters=dict(rope_type="default", rope_theta=10000000,
                                 partial_rotary_factor=.25, mrope_section=[3, 3, 2],
                                 mrope_interleaved=True),
            pad_token_id=0, bos_token_id=1, eos_token_id=2,
            tie_word_embeddings=False),
        vision_config=dict(
            depth=2, hidden_size=64, intermediate_size=128, num_heads=4,
            patch_size=16, temporal_patch_size=2, spatial_merge_size=2,
            out_hidden_size=128, num_position_embeddings=64,
            deepstack_visual_indexes=[]),
        image_token_id=120, video_token_id=121, vision_start_token_id=118,
        vision_end_token_id=119, tie_word_embeddings=False)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--artifact-config", action="append", default=[], type=Path)
    parser.add_argument("--export-checkpoint", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(20261102)
    config = tiny_config()
    config._attn_implementation = "eager"
    model = Qwen3_5ForConditionalGeneration(config).float().eval()
    # Match the plan's stored BF16 values, executing in CPU float32.
    model.bfloat16().float()
    if args.export_checkpoint:
        from tokenizers import Tokenizer, models
        from transformers import PreTrainedTokenizerFast, Qwen3VLProcessor, Qwen2VLImageProcessorPil, Qwen3VLVideoProcessor
        vocab = {f"t{i}": i for i in range(128)}
        for index, token in {0: "<pad>", 1: "<bos>", 2: "<eos>",
                             3: " ", 4: "\n", 127: "<unk>",
                             **{i+5: chr(i+33) for i in range(94)},
                             118: "<|vision_start|>", 119: "<|vision_end|>",
                             120: "<|image_pad|>", 121: "<|video_pad|>"}.items():
            del vocab[f"t{index}"]
            vocab[token] = index
        backend = Tokenizer(models.BPE(vocab, merges=[], unk_token="<unk>"))
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend, pad_token="<pad>", bos_token="<bos>",
            eos_token="<eos>", unk_token="<unk>",
            additional_special_tokens=["<|vision_start|>", "<|vision_end|>",
                                       "<|image_pad|>", "<|video_pad|>"])
        processor = Qwen3VLProcessor(
            image_processor=Qwen2VLImageProcessorPil(
                patch_size=16, temporal_patch_size=2, merge_size=2,
                image_mean=[.5]*3, image_std=[.5]*3,
                size={"shortest_edge": 1024, "longest_edge": 1048576}),
            tokenizer=tokenizer, video_processor=Qwen3VLVideoProcessor(),
            chat_template="{{ messages[0]['content'] }}")
        model.save_pretrained(args.export_checkpoint)
        processor.save_pretrained(args.export_checkpoint)
    record = dict(
        kind="WP0 smoke probes, not acceptance goldens", seed=20261102,
        platform=platform.platform(), python=platform.python_version(), device="cpu",
        torch_threads=1, precision="BF16-rounded weights, FLOAT32 execution",
        packages={p: importlib.metadata.version(p) for p in
                  ("torch", "torchvision", "transformers", "numpy", "Pillow",
                   "safetensors", "tokenizers", "Jinja2")},
        sources={"modeling_qwen3_5.py": digest(impl.__file__),
                 "configuration_qwen3_5.py": digest(inspect.getfile(type(config))),
                 "probe_hf.py": digest(__file__)},
        kernel_substitutions={p: importlib.util.find_spec(p) is not None
                              for p in ("fla", "causal_conv1d", "kernels")},
        config=config.to_dict(), cases=[], artifact_configs=[])
    assert not any(record["kernel_substitutions"].values())
    assert model.model.language_model.config._attn_implementation == "eager"
    assert model.model.visual.config._attn_implementation == "eager"
    for path in args.artifact_config:
        parsed = Qwen3_5Config.from_dict(json.loads(path.read_text()))
        record["artifact_configs"].append(dict(
            path=path.name, sha256=digest(path),
            vision_model_type=parsed.vision_config.model_type,
            layer_types=parsed.text_config.layer_types,
            rope_parameters=parsed.text_config.rope_parameters))
    with torch.no_grad():
        for image in (False, True):
            ids = torch.tensor([[1, 118, 120, 119, 3, 4]] if image else [[1, 3, 4, 5]])
            kwargs = dict(pixel_values=torch.randn(4, 1536),
                          image_grid_thw=torch.tensor([[1, 2, 2]]),
                          mm_token_type_ids=torch.tensor([[0, 0, 1, 0, 0, 0]])) if image else {}
            output = model(input_ids=ids, use_cache=True, **kwargs)
            decode = model(input_ids=torch.tensor([[6]]),
                           past_key_values=output.past_key_values, use_cache=True)
            if image:
                kwargs["mm_token_type_ids"] = torch.tensor([[0, 0, 1, 0, 0, 0, 0]])
            fresh = model(input_ids=torch.cat((ids, torch.tensor([[6]])), dim=1),
                          use_cache=False, **kwargs)
            error = (decode.logits[:, -1] - fresh.logits[:, -1]).abs().max().item()
            assert error < 1e-4, error
            record["cases"].append(dict(
                name="image" if image else "text", input_ids=ids.tolist(),
                prefill_shape=list(output.logits.shape),
                decode_vs_fresh_max_abs=error,
                pixel_values=kwargs["pixel_values"].tolist() if image else None,
                prefill_logits=output.logits[:, -1].tolist(),
                decode_logits=decode.logits[:, -1].tolist()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
