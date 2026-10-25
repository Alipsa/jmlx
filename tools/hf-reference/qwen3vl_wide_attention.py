#!/usr/bin/env python3
"""Generate wide-attention HF float32 logits without duplicating the base checkpoint.

Uses the pinned qwen3vl-model checkpoint, changing only text query heads to 8
(hidden 256, attention width 512) and q_proj/o_proj to asymmetric index-coded
weights. Integer residues divided by 4096 are exactly representable in BF16,
so Java reconstructs the same weights without sharing a random-number library.
Every head and projection has distinct values; the reference uses Transformers
4.57.6 eager CPU attention and full-history recomputation for decode goldens.
Run manually under tools/hf-reference/.venv; ordinary Gradle checks only digests.
"""
from pathlib import Path
import json

import torch
from safetensors.torch import load_file
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
from qwen3vl_model import (
    gate_versions, slow_processor, expand_prompt, TEXT_PROMPT, ONE_IMAGE_PROMPT,
    EXPECTED_GRIDS, sha256_file,
)


def projection(rows, columns, salt):
    row = torch.arange(rows, dtype=torch.int64)[:, None]
    col = torch.arange(columns, dtype=torch.int64)[None, :]
    return ((row * 17 + col * 31 + row * col * 7 + salt * 13) % 251 - 125).float() / 4096


def main():
    environment = gate_versions()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    root = Path(__file__).resolve().parent
    checkpoint = root / 'goldens/checkpoints/qwen3vl-model'
    cfg = Qwen3VLConfig.from_pretrained(checkpoint)
    cfg.text_config.num_attention_heads = 8
    cfg.text_config.dtype = torch.float32
    cfg._attn_implementation = 'eager'
    ref = Qwen3VLForConditionalGeneration(cfg)
    ref._tie_or_clone_weights(ref.lm_head, ref.model.language_model.embed_tokens)
    weights = {k: v.float() for k, v in load_file(checkpoint / 'model.safetensors').items()}
    for layer in range(2):
        prefix = f'model.language_model.layers.{layer}.self_attn.'
        weights[prefix + 'q_proj.weight'] = projection(512, 256, 11 + 2 * layer)
        weights[prefix + 'o_proj.weight'] = projection(256, 512, 12 + 2 * layer)
    missing, extra = ref.load_state_dict(weights, strict=False)
    assert missing == ['lm_head.weight'] and not extra, (missing, extra)
    ref.eval()
    batch = slow_processor().preprocess([str(checkpoint / 'images/a.png')], return_tensors='pt')
    grid = [EXPECTED_GRIDS['a']]
    expanded = expand_prompt(ONE_IMAGE_PROMPT, grid)
    cases = []
    with torch.no_grad():
        for name, ids, images in [('text', TEXT_PROMPT, []), ('image', ONE_IMAGE_PROMPT, ['a.png'])]:
            history = expanded if images else ids
            kwargs = dict(pixel_values=batch['pixel_values'], image_grid_thw=batch['image_grid_thw']) if images else {}
            prefill = ref(input_ids=torch.tensor([history]), **kwargs).logits[0]
            token = int(prefill[-1].argmax())
            decoded = ref(input_ids=torch.tensor([history + [token]]), **kwargs).logits[0, -1]
            cases.append(dict(name=name + '_prefill', unexpanded_ids=ids, images=images,
                              logits=(prefill[-1] if images else prefill).tolist()))
            cases.append(dict(name=name + '_decode', unexpanded_ids=ids, images=images,
                              token_id=token, logits=decoded.tolist()))
    result = dict(generated_by=Path(__file__).name, environment=environment,
                  checkpoint_sha256=sha256_file(checkpoint / 'model.safetensors'),
                  generator_sha256=sha256_file(Path(__file__)),
                  weights='((row*17 + col*31 + row*col*7 + salt*13) % 251 - 125) / 4096',
                  hidden=256, query_heads=8, head_dim=64, cases=cases)
    target = root / 'goldens/qwen3vl-wide-attention.json'
    target.write_text(json.dumps(result, indent=2) + '\n')
    print(f'{target}: {sha256_file(target)}')


if __name__ == '__main__':
    main()
