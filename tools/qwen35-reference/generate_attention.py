#!/usr/bin/env python3
"""Independent HF gated attention, partial MRoPE and both hybrid block variants."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import DynamicCache, Qwen3_5ForConditionalGeneration
from transformers.models.qwen3_5 import modeling_qwen3_5 as hf
from probe_hf import tiny_config


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(20261103)
    tensors = {}
    records = []
    with torch.no_grad():
        for name, hidden in (("base", 128), ("wide", 96)):
            config = tiny_config()
            config.text_config.hidden_size = hidden
            config.text_config.intermediate_size = 2*hidden
            config.vision_config.out_hidden_size = hidden
            config._attn_implementation = "eager"
            model = Qwen3_5ForConditionalGeneration(config).eval()
            for module in model.modules():
                if isinstance(module, hf.Qwen3_5RMSNorm):
                    module.weight.copy_(torch.randn_like(module.weight)*.05)
            model.bfloat16().float()
            text = model.model.language_model
            attention = text.layers[3].self_attn
            rotary = hf.Qwen3_5TextRotaryEmbedding(config.text_config)
            x = torch.randn(1, 5, hidden)
            token = torch.randn(1, 1, hidden)
            positions = torch.tensor([[[0, 1, 2, 3, 4]], [[0, 2, 1, 5, 3]], [[0, 4, 2, 1, 6]]], dtype=torch.int32)
            decode_positions = torch.tensor([[[9]], [[10]], [[11]]], dtype=torch.int32)
            cos, sin = rotary(x, positions)
            decode_cos, decode_sin = rotary(token, decode_positions)
            mask = torch.zeros(1, 1, 5, 5).masked_fill(torch.ones(5, 5, dtype=torch.bool).triu(1), -float("inf"))
            for key, value in dict(input=x, decode_input=token, positions=positions,
                                    decode_positions=decode_positions, cos=cos, sin=sin).items():
                tensors[name+"."+key] = value.contiguous()
            for kind, module, index in (("attention", attention, 3), ("full_block", text.layers[3], 3),
                                        ("linear_block", text.layers[0], 0)):
                prefix = name+"."+kind+"."
                for key, value in module.state_dict().items():
                    tensors[prefix+"weights."+key] = value.detach().contiguous()
                cache = DynamicCache(config=config.text_config)
                if kind == "attention":
                    output, _ = module(x, (cos, sin), mask, past_key_values=cache)
                else:
                    output = module(x, position_embeddings=(cos, sin), attention_mask=mask if index==3 else None, past_key_values=cache)
                tensors[prefix+"output"] = output.clone()
                if index == 3:
                    tensors[prefix+"keys"] = cache.layers[3].keys.clone()
                    tensors[prefix+"values"] = cache.layers[3].values.clone()
                else:
                    tensors[prefix+"state"] = cache.get_recurrent_state(0).clone()
                    tensors[prefix+"convolution"] = cache.get_conv_state(0)[:, :, -3:].transpose(1, 2).contiguous()
                if kind == "attention":
                    decode, _ = module(token, (decode_cos, decode_sin), None, past_key_values=cache)
                else:
                    decode = module(token, position_embeddings=(decode_cos, decode_sin), past_key_values=cache)
                tensors[prefix+"decode_output"] = decode.clone()
                if index == 3:
                    tensors[prefix+"decode_keys"] = cache.layers[3].keys.clone()
                    tensors[prefix+"decode_values"] = cache.layers[3].values.clone()
                else:
                    tensors[prefix+"decode_state"] = cache.get_recurrent_state(0).clone()
                    tensors[prefix+"decode_convolution"] = cache.get_conv_state(0)[:, :, -3:].transpose(1, 2).contiguous()
            projected = attention.q_proj(x).reshape(1, 5, 2, 128)
            query, gate = torch.chunk(projected, 2, dim=-1)
            query = attention.q_norm(query).transpose(1, 2)
            key = attention.k_norm(attention.k_proj(x).reshape(1, 5, 1, 64)).transpose(1, 2)
            qr, kr = hf.apply_rotary_pos_emb(query, key, cos, sin)
            for label, value in dict(query_normalized=query, key_normalized=key,
                                     query_rotated=qr, key_rotated=kr, gate=gate).items():
                tensors[name+"."+label] = value.contiguous()
            assert torch.equal(query[..., 16:], qr[..., 16:])
            records.append(dict(name=name, hidden=hidden, attention_width=128, head_dim=64,
                                rotary_dim=16, heads=2, kv_heads=1, mrope_section=[3, 3, 2]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file({key: value.clone() for key, value in tensors.items()}, args.output,
              metadata={"reference": "HF 5.22.1 CPU FLOAT32, BF16-rounded weights"})
    args.output.with_suffix(".json").write_text(json.dumps(dict(
        seed=20261103, source_sha256=digest(__file__), hf_source_sha256=digest(hf.__file__),
        safetensors_sha256=digest(args.output), cases=records), indent=2)+"\n")
    print(args.output)


if __name__ == "__main__":
    main()
