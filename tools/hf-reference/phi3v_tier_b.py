#!/usr/bin/env python3
"""Generate reviewed CPU float32 Phi-3.5-Vision 4B Tier-B candidates; never called by Gradle.

Mirrors qwen3vl_tier_b.py's review contract for the pinned
microsoft/Phi-3.5-vision-instruct snapshot: the artifact's bundled
trust-remote-code processor and model (transformers 4.57.6 has no native
phi3_v), the raw model logits at every expected greedy history, per-image
processed-pixel hashes, the sentinel-expanded input IDs, and a little-endian
float32 logits binary (one full vocab vector per greedy step, per case) whose
digest is pinned in the manifest and in provenance.json.

Every greedy step is a fresh, full, uncached forward (use_cache=False): the
bundled modeling_phi3_v.py cannot run a cached forward under the pinned
transformers (findings section 4). The greedy loop stops at the token 32000
(the tokenizer's 13-character endoftext literal: special, eos and pad;
findings section 7.3), so 32000 is the last entry of target_ids exactly when
finish_reason is EOS.
"""
import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import PIL
from PIL import Image
import torch
import torchvision
import transformers
from transformers import LlamaTokenizerFast
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.models.clip.configuration_clip import CLIPVisionConfig

# Measured Java-vs-reference maximum error on the pinned artifact is
# 1.02996826171875e-4 (TF32 off, pinned mlx-metal 0.31.2 / mlx-c fba4470,
# macOS 27.0.1; stable across repeated runs), so epsilon keeps about 2x headroom
# over it; margin stays 1e-4 (the recorded minimum greedy gap is 0.003166, about
# 6x the 2*epsilon + margin = 5e-4 eligibility bound).
EPSILON = 2e-4
MARGIN = 1e-4
STATUS = 'eligible after measured real-checkpoint error; local macOS 27 run passed; macOS 26 CI pending'

# The stop token (findings section 7.3): 32000 is the 13-character endoftext
# literal, special and both eos and pad. 32007 (<|end|>) is emitted by the
# model before it stops and must not be a stop.
STOP_TOKEN_ID = 32000


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def gate_versions():
    if (transformers.__version__ != '4.57.6'
            or torchvision.__version__ != '0.21.0'
            or PIL.__version__ != '12.3.0'
            or np.__version__ != '2.5.3'
            or torch.__version__ not in ('2.6.0', '2.6.0+cpu')
            or platform.python_version_tuple()[:2] != ('3', '12')):
        raise RuntimeError('Install the reviewed hash-locked reference environment')


def generate(directory, manifest_path):
    gate_versions()
    transformers.logging.set_verbosity_error()
    manifest = json.loads(manifest_path.read_text())
    manifest.pop('local_evidence', None)  # Regeneration invalidates previous measurements.
    if manifest['task'] != 'phi3v-generation':
        raise RuntimeError('Expected a phi3v-generation manifest')
    for name, spec in manifest['files'].items():
        if sha(directory / name) != spec['sha256']:
            raise RuntimeError(f'artifact hash mismatch: {name}')
    repository = Path(__file__).resolve().parents[2]
    images = []
    for spec in manifest['images']:
        path = repository / spec['path']
        if sha(path) != spec['sha256']:
            raise RuntimeError(f'image hash mismatch: {path}')
        image = Image.open(path)
        image.load()  # Raw PIL, no EXIF transpose, preliminary RGB conversion or ImageCms.
        images.append(image)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    # ------------------------------------------------------------- bundled code
    # transformers 4.57.6 has no native phi3_v: the model, its config and the
    # processor are the artifact's trust-remote-code bundle, whose file SHAs
    # are verified against the manifest above and re-recorded below.
    config_cls = get_class_from_dynamic_module(
        'configuration_phi3_v.Phi3VConfig', str(directory))
    model_cls = get_class_from_dynamic_module(
        'modeling_phi3_v.Phi3VForCausalLM', str(directory))
    model_mod = sys.modules[model_cls.__module__]

    # The pristine bundled CLIP constant (the real ViT-L/14@336 tower config)
    # is read by Phi3ImageEmbedding at build time and its None
    # attn_implementation resolves to sdpa in place (findings section 5):
    # assert the pristine values, then swap an eager copy in for this process.
    pristine = model_mod.CLIP_VIT_LARGE_PATCH14_336_CONFIG
    if (pristine.hidden_size != 1024 or pristine.intermediate_size != 4096
            or pristine.num_hidden_layers != 24 or pristine.image_size != 336
            or pristine.patch_size != 14 or pristine.num_attention_heads != 16
            or pristine.num_channels != 3):
        raise RuntimeError(f'pristine CLIP constant drifted: {pristine.to_dict()}')
    model_mod.CLIP_VIT_LARGE_PATCH14_336_CONFIG = CLIPVisionConfig(
        **{**pristine.to_dict(), 'attn_implementation': 'eager'})

    cfg = config_cls.from_pretrained(str(directory))
    if (cfg.architectures != ['Phi3VForCausalLM'] or cfg.model_type != 'phi3_v'
            or cfg.num_hidden_layers != 32 or cfg.hidden_size != 3072
            or cfg.num_attention_heads != 32 or cfg.num_key_value_heads != 32
            or cfg.intermediate_size != 8192 or cfg.vocab_size != 32064
            or cfg.rope_theta != 10000
            or cfg.original_max_position_embeddings != 4096
            or cfg.max_position_embeddings != 131072
            or cfg.rope_scaling.get('type') != 'su'
            or len(cfg.rope_scaling.get('short_factor', [])) != 48
            or len(cfg.rope_scaling.get('long_factor', [])) != 48):
        raise RuntimeError(f'artifact config drifted: {cfg.to_dict()}')
    cfg._attn_implementation = 'eager'
    model = model_cls(cfg)
    if model.config._attn_implementation != 'eager':
        raise RuntimeError(f'model attention implementation is {model.config._attn_implementation}')
    if model.model.vision_embed_tokens.img_processor.config._attn_implementation != 'eager':
        raise RuntimeError('CLIP attention implementation is not eager')

    # The 592 stored tensors (all bfloat16) up-cast to float32, exactly the
    # Java loader's stored-bf16 -> float32 path. The shared embedding is stored
    # once under model.embed_tokens.weight; the model registers the vision
    # alias key from the same tensor, so the strict load re-supplies it.
    from safetensors import torch as st

    index = json.loads((directory / 'model.safetensors.index.json').read_text())
    state = {}
    for shard in sorted(set(index['weight_map'].values())):
        state.update(st.load_file(directory / shard))
    if len(state) != 592:
        raise RuntimeError(f'expected 592 stored tensors, found {len(state)}')
    if any(value.dtype != torch.bfloat16 for value in state.values()):
        raise RuntimeError('a stored tensor is not bfloat16')
    state = {key: value.float() for key, value in state.items()}
    state['model.vision_embed_tokens.wte.weight'] = state['model.embed_tokens.weight']
    model.load_state_dict(state, strict=True)
    model.eval()

    # ------------------------------------------------------------ processor
    proc_cls = get_class_from_dynamic_module(
        'processing_phi3_v.Phi3VProcessor', str(directory))
    proc_mod = sys.modules[proc_cls.__module__]
    image_processor = proc_mod.Phi3VImageProcessor.from_pretrained(str(directory))
    if int(image_processor.num_crops) != 4:
        raise RuntimeError(f'processor options drifted: num_crops={image_processor.num_crops}')
    preprocessor = json.loads((directory / 'preprocessor_config.json').read_text())
    if ([float(x) for x in image_processor.image_mean] != preprocessor['image_mean']
            or [float(x) for x in image_processor.image_std] != preprocessor['image_std']
            or image_processor.do_convert_rgb is not True):
        raise RuntimeError('processor mean/std/convert_rgb drifted')
    tokenizer = LlamaTokenizerFast.from_pretrained(str(directory))
    processor = proc_cls(image_processor=image_processor, tokenizer=tokenizer)

    cases, step_blocks = [], []
    with torch.no_grad():
        for specification in manifest['requests']:
            text = tokenizer.apply_chat_template(
                specification['messages'], tokenize=False, add_generation_prompt=True)
            selected = [images[i] for i in specification['image_indices']]
            # Per-image processed pixels: the full num_crops+1 slot row, zero
            # padding included (exactly the bytes the Java processor returns),
            # plus the HD geometry and token count, in selection order.
            pixel_shas, image_sizes, img_tokens = [], [], []
            for image in selected:
                single = image_processor(images=[image], return_tensors='pt')
                pixels = torch.as_tensor(single['pixel_values'], dtype=torch.float32)
                if pixels.shape[0] != 1:
                    raise RuntimeError('single-image batch must be 1')
                pixel_shas.append(
                    hashlib.sha256(pixels[0].numpy().tobytes()).hexdigest())
                image_sizes.append(
                    [int(x) for x in single['image_sizes'][0].tolist()])
                img_tokens.append(int(single['num_img_tokens'][0]))
            inputs = processor(text=text, images=selected, return_tensors='pt')
            pixel_values = torch.as_tensor(inputs['pixel_values'], dtype=torch.float32)
            sizes_tensor = torch.as_tensor(inputs['image_sizes'], dtype=torch.long)
            if sizes_tensor.tolist() != image_sizes:
                raise RuntimeError('per-image and batched geometries disagree')
            if (int(inputs['attention_mask'].min()) != 1
                    or int(inputs['attention_mask'].max()) != 1):
                raise RuntimeError('attention mask must be all ones')
            history = inputs['input_ids'][0].tolist()
            target, logits_gaps, top_ids, blocks = [], [], [], []
            for step in range(specification['max_new_tokens']):
                output = model(
                    input_ids=torch.tensor([history], dtype=torch.long),
                    attention_mask=torch.ones(1, len(history), dtype=torch.long),
                    pixel_values=pixel_values, image_sizes=sizes_tensor,
                    use_cache=False)
                scores = output.logits[0, -1].to(torch.float32)
                top = torch.topk(scores, 2)
                token = int(top.indices[0])
                target.append(token)
                blocks.append(scores.cpu().numpy().astype(np.float32, copy=False))
                logits_gaps.append(float(top.values[0] - top.values[1]))
                top_ids.append([int(v) for v in top.indices.tolist()])
                history = history + [token]
                if token == STOP_TOKEN_ID:
                    break
            cases.append({**specification, 'rendered_prompt': text,
                          'unexpanded_ids': tokenizer(
                              text, add_special_tokens=False)['input_ids'],
                          'expanded_ids': inputs['input_ids'][0].tolist(),
                          'image_sizes': image_sizes,
                          'num_img_tokens': img_tokens,
                          'pixel_values_sha256': pixel_shas,
                          'target_ids': target,
                          'gaps': logits_gaps, 'top_two_ids': top_ids,
                          # The exact decoded-output assertion the plan requires:
                          # the Java suite's result.generatedText() (its
                          # incremental decoder runs with skip_special_tokens=True
                          # on this same tokenizer.json) must equal this string
                          # byte-for-byte.
                          'expected_text': tokenizer.decode(
                              target, skip_special_tokens=True),
                          'finish_reason':
                              'EOS' if target[-1] == STOP_TOKEN_ID else 'MAX_TOKENS'})
            step_blocks.append(blocks)
    logits_path = Path(__file__).with_name('goldens') / 'phi3v-tier-b-logits.bin'
    blob = b''.join(block.tobytes() for blocks in step_blocks for block in blocks)
    if len(blob) != sum(block.nbytes for blocks in step_blocks for block in blocks):
        raise RuntimeError('logits serialization length mismatch')
    logits_path.write_bytes(blob)
    manifest['reference'] = {
        'status': STATUS,
        'epsilon': EPSILON, 'margin': MARGIN,
        'stop_token_id': STOP_TOKEN_ID,
        'precision': 'CPU float32 eager (bundled trust-remote-code); Java TF32 off',
        'image_loading':
            'raw PIL.Image.open -> load -> bundled Phi3VImageProcessor per image',
        'versions': {'python': platform.python_version(), 'torch': torch.__version__,
                     'torchvision': torchvision.__version__,
                     'transformers': transformers.__version__,
                     'pillow': PIL.__version__, 'numpy': np.__version__},
        'requirements_lock_sha256': sha(Path(__file__).with_name('requirements.lock')),
        'generator_sha256': sha(Path(__file__)),
        'source_sha256': {name: sha(directory / name) for name in
                          ('configuration_phi3_v.py', 'modeling_phi3_v.py',
                           'processing_phi3_v.py', 'preprocessor_config.json')},
        'logits_file': {'path': str(logits_path.relative_to(repository)),
                        'sha256': sha(logits_path),
                        'layout': 'little-endian float32; per case, per greedy step, '
                                  'one full vocab-size vector per step'},
        'cases': cases}
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    provenance_path = Path(__file__).with_name('provenance.json')
    provenance = json.loads(provenance_path.read_text())
    paths = [manifest_path.resolve(), Path(__file__).resolve()]
    paths += [(repository / image['path']).resolve() for image in manifest['images']]
    provenance['files'][logits_path.name] = sha(logits_path)
    provenance['phi3v_tier_b_files'] = {str(path.relative_to(repository)): sha(path)
                                        for path in paths}
    provenance_path.write_text(json.dumps(provenance, indent=2) + '\n')
    print('minimum gap:', min(g for c in cases for g in c['gaps']), flush=True)
    print('max_new_tokens used:', [len(c['target_ids']) for c in cases], flush=True)
    print('greedy ids:', [c['target_ids'] for c in cases], flush=True)
    print('expanded lengths:', [len(c['expanded_ids']) for c in cases], flush=True)
    print('logits bytes:', logits_path.stat().st_size, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    generate(args.directory, args.manifest)
