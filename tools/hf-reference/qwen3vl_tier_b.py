#!/usr/bin/env python3
"""Generate reviewed CPU float32 Qwen3-VL 2B Tier-B candidates; never called by Gradle.

Mirrors vision_tier_b.py's review contract for the pinned Qwen/Qwen3-VL-2B-Instruct
snapshot: the locked slow processor, the raw model logits at every expected greedy
history, per-image processed-pixel hashes, expansion/rope-delta facts, and a
little-endian float32 logits binary (one full vocab vector per greedy step, per
case) whose digest is pinned in the manifest and in provenance.json.
"""
import argparse
import hashlib
import inspect
import json
import platform
from pathlib import Path

import numpy as np
import PIL
from PIL import Image
import torch
import transformers
from transformers import Qwen2TokenizerFast
from transformers import Qwen3VLForConditionalGeneration
from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
    Qwen2VLImageProcessor,
)
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
from transformers.models.qwen3_vl.video_processing_qwen3_vl import (
    Qwen3VLVideoProcessor,
)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def generate(directory, manifest_path):
    if (transformers.__version__ != '4.57.6' or PIL.__version__ != '12.3.0'
            or np.__version__ != '2.5.3' or torch.__version__ not in ('2.6.0', '2.6.0+cpu')
            or platform.python_version_tuple()[:2] != ('3', '12')):
        raise RuntimeError('Install the reviewed hash-locked reference environment')
    manifest = json.loads(manifest_path.read_text())
    manifest.pop('local_evidence', None)  # Regeneration invalidates previous measurements.
    if manifest['task'] != 'qwen3-vl-generation':
        raise RuntimeError('Expected a qwen3-vl-generation manifest')
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
    # AutoProcessor is not usable in the locked CPU venv: the artifact declares a
    # video processor and the venv has no torchvision. The WP1 probe (qwen3vl_probe.py)
    # records the consequences in detail; this generator reuses its construction:
    # explicit min_pixels/max_pixels, because 4.57.6's slow Qwen2VLImageProcessor
    # __init__ overwrites the artifact-declared size dict with 3136/1003520 defaults,
    # and a class-check override for the video processor, which is image-inert.
    preprocessor = json.loads((directory / 'preprocessor_config.json').read_text())
    declared_min = int(preprocessor['size']['shortest_edge'])
    declared_max = int(preprocessor['size']['longest_edge'])
    img_proc = Qwen2VLImageProcessor.from_pretrained(
        str(directory), min_pixels=declared_min, max_pixels=declared_max)
    if (img_proc.min_pixels, img_proc.max_pixels) != (declared_min, declared_max):
        raise RuntimeError('the artifact-declared pixel window must be in force')
    tokenizer = Qwen2TokenizerFast.from_pretrained(str(directory))
    video_proc = Qwen3VLVideoProcessor.from_pretrained(str(directory))
    tokenizer_config = json.loads((directory / 'tokenizer_config.json').read_text())
    chat_template = tokenizer_config['chat_template']
    # The hub also ships chat_template.json; the two must agree for the recorded
    # rendered prompts to be template-stable (WP0 verified them byte-identical).
    if json.loads((directory / 'chat_template.json').read_text())['chat_template'] != chat_template:
        raise RuntimeError('chat_template.json disagrees with tokenizer_config.json')

    class TierBProcessor(Qwen3VLProcessor):
        def check_argument_for_proper_class(self, name, arg):
            if name == 'video_processor':
                return Qwen3VLVideoProcessor
            return super().check_argument_for_proper_class(name, arg)

    processor = TierBProcessor(image_processor=img_proc, tokenizer=tokenizer,
                               video_processor=video_proc, chat_template=chat_template)
    if type(processor.image_processor).__name__ != 'Qwen2VLImageProcessor':
        raise RuntimeError('The pinned slow Qwen2VLImageProcessor is required')
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        directory, local_files_only=True, dtype=torch.float32, attn_implementation='eager'
    ).eval()
    if model.dtype != torch.float32:
        raise RuntimeError('The reference must run CPU float32')
    llm = model.model
    eos = model.config.text_config.eos_token_id
    cases, step_blocks = [], []
    with torch.no_grad():
        for specification in manifest['requests']:
            text = processor.apply_chat_template(
                specification['messages'], add_generation_prompt=True)
            selected = [images[i] for i in specification['image_indices']]
            # Per-image processed pixels and grids, in selection order: the combined
            # processor call concatenates exactly these per-image tensors, and the
            # Java side re-processes each image individually.
            pixel_shas, grids = [], []
            # The pixel tensor depends only on the image, so each per-image call
            # uses its own one-image rendered prompt (the request's text has one
            # placeholder per request image, and a single-image call must match it).
            single_prompt = processor.apply_chat_template(
                [{'role': 'user', 'content': [
                    {'type': 'image'},
                    {'type': 'text', 'text': 'Describe this image.'}]}],
                add_generation_prompt=True)
            for image in selected:
                single = processor(text=single_prompt, images=[image], return_tensors='pt')
                pixels = torch.as_tensor(single['pixel_values'], dtype=torch.float32)
                pixel_shas.append(
                    hashlib.sha256(pixels.numpy().tobytes()).hexdigest())
                grids.append([int(v) for v in single['image_grid_thw'][0].tolist()])
            inputs = processor(text=text, images=selected, return_tensors='pt')
            pixel_values = torch.as_tensor(inputs['pixel_values'], dtype=torch.float32)
            image_grid_thw = torch.as_tensor(inputs['image_grid_thw'], dtype=torch.long)
            history = inputs['input_ids']
            attention_mask = torch.ones_like(history)
            _, rope_deltas = llm.get_rope_index(
                history, image_grid_thw, None, attention_mask=attention_mask)
            delta = int(rope_deltas.flatten()[0])
            target, logits_gaps, top_ids, blocks = [], [], [], []
            for step in range(specification['max_new_tokens']):
                attention_mask = torch.ones_like(history)
                output = model(
                    input_ids=history, attention_mask=attention_mask,
                    pixel_values=pixel_values, image_grid_thw=image_grid_thw,
                    use_cache=False)
                scores = output.logits[0, -1].to(torch.float32)
                top = torch.topk(scores, 2)
                token = int(top.indices[0])
                target.append(token)
                blocks.append(scores.cpu().numpy().astype(np.float32, copy=False))
                logits_gaps.append(float(top.values[0] - top.values[1]))
                top_ids.append([int(v) for v in top.indices.tolist()])
                history = torch.cat((history, history.new_tensor([[token]])), dim=1)
                if token == eos:
                    break
            cases.append({**specification, 'rendered_prompt': text,
                          'unexpanded_ids': processor.tokenizer(
                              text, add_special_tokens=False)['input_ids'],
                          'expanded_ids': inputs['input_ids'][0].tolist(),
                          'grids': grids, 'delta': delta, 'target_ids': target,
                          'gaps': logits_gaps, 'top_two_ids': top_ids,
                          'pixel_values_sha256': pixel_shas,
                          # The exact decoded-output assertion the plan requires: the Java
                          # suite's result.generatedText() (its incremental decoder runs with
                          # skip_special_tokens=True on this same tokenizer.json) must equal
                          # this string byte-for-byte.
                          'expected_text': tokenizer.decode(
                              target, skip_special_tokens=True),
                          'finish_reason': 'EOS' if target[-1] == eos else 'MAX_TOKENS'})
            step_blocks.append(blocks)
    logits_path = Path(__file__).with_name('goldens') / 'qwen3vl-tier-b-logits.bin'
    blob = b''.join(block.tobytes() for blocks in step_blocks for block in blocks)
    if len(blob) != sum(block.nbytes for blocks in step_blocks for block in blocks):
        raise RuntimeError('logits serialization length mismatch')
    logits_path.write_bytes(blob)
    manifest['reference'] = {
        # Measured Java-vs-reference maximum error on the pinned artifact is 1.023e-4
        # (TF32 off, pinned mlx-metal 0.31.2 / mlx-c fba4470, macOS 27.0.1), so epsilon keeps
        # about 2x headroom over it; margin stays 1e-4 (the recorded minimum greedy gap is
        # 0.1028, orders of magnitude above the 2*epsilon + margin eligibility bound).
        'status': 'eligible after measured real-checkpoint error; local macOS 27 run passed; macOS 26 CI pending',
        'epsilon': 2e-4, 'margin': 1e-4,
        'precision': 'CPU float32 eager; Java TF32 off',
        'image_loading': 'raw PIL.Image.open -> load -> slow Qwen2VLImageProcessor per image',
        'versions': {'python': platform.python_version(), 'torch': torch.__version__,
                     'transformers': transformers.__version__, 'pillow': PIL.__version__,
                     'numpy': np.__version__},
        'requirements_lock_sha256': sha(Path(__file__).with_name('requirements.lock')),
        'generator_sha256': sha(Path(__file__)),
        'source_sha256': {cls.__name__: sha(Path(inspect.getsourcefile(cls)))
                          for cls in (Qwen3VLProcessor, type(processor.image_processor),
                                      Qwen3VLForConditionalGeneration)},
        'logits_file': {'path': str(logits_path.relative_to(repository)),
                        'sha256': sha(logits_path),
                        'layout': 'little-endian float32; per case, per greedy step, one full vocab-size vector per step'},
        'cases': cases}
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    provenance_path = Path(__file__).with_name('provenance.json')
    provenance = json.loads(provenance_path.read_text())
    paths = [manifest_path.resolve(), Path(__file__).resolve()]
    paths += [(repository / image['path']).resolve() for image in manifest['images']]
    provenance['files'][logits_path.name] = sha(logits_path)
    provenance['qwen3vl_tier_b_files'] = {str(path.relative_to(repository)): sha(path)
                                          for path in paths}
    provenance_path.write_text(json.dumps(provenance, indent=2) + '\n')
    print('minimum gap:', min(g for c in cases for g in c['gaps']), flush=True)
    print('max_new_tokens used:', [len(c['target_ids']) for c in cases], flush=True)
    print('greedy ids:', [c['target_ids'] for c in cases], flush=True)
    print('logits bytes:', logits_path.stat().st_size, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    generate(args.directory, args.manifest)
