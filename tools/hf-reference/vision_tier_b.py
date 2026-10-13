#!/usr/bin/env python3
"""Generate reviewed CPU float32 SmolVLM Tier-B candidates; never called by Gradle."""
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
from transformers import AutoProcessor, Idefics3ForConditionalGeneration


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
    if manifest['task'] != 'image-text-generation':
        raise RuntimeError('Expected an image-text-generation manifest')
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
    processor = AutoProcessor.from_pretrained(directory, local_files_only=True, use_fast=False)
    if type(processor.image_processor).__name__ != 'Idefics3ImageProcessor':
        raise RuntimeError('The pinned slow processor is required')
    model = Idefics3ForConditionalGeneration.from_pretrained(
        directory, local_files_only=True, attn_implementation='eager', torch_dtype=torch.float32
    ).cpu().float().eval()
    cases = []
    with torch.no_grad():
        for specification in manifest['requests']:
            text = processor.apply_chat_template(specification['messages'], add_generation_prompt=True)
            selected = [images[i] for i in specification['image_indices']]
            inputs = processor(text=text, images=[selected], return_tensors='pt')
            count = specification['max_new_tokens']
            # Raw model logits at each expected decoder history; no generation processors.
            history = inputs['input_ids']
            target, logits, gaps, top_ids = [], [], [], []
            for step in range(count):
                output = model(input_ids=history, attention_mask=torch.ones_like(history),
                               pixel_values=inputs['pixel_values'],
                               pixel_attention_mask=inputs['pixel_attention_mask'], use_cache=False)
                scores = output.logits[0, -1]
                top = torch.topk(scores, 2)
                token = top.indices[0].item()
                target.append(token)
                logits.append(scores.tolist())
                gaps.append((top.values[0] - top.values[1]).item())
                top_ids.append(top.indices.tolist())
                history = torch.cat((history, torch.tensor([[token]])), dim=1)
                if token == model.config.text_config.eos_token_id:
                    break
            cases.append({**specification, 'rendered_prompt': text,
                          'unexpanded_ids': processor.tokenizer(text, add_special_tokens=False)['input_ids'],
                          'expanded_ids': inputs['input_ids'][0].tolist(), 'target_ids': target,
                          'logits': logits, 'gaps': gaps, 'top_two_ids': top_ids,
                          'pixel_values_sha256': hashlib.sha256(inputs['pixel_values'].numpy().tobytes()).hexdigest(),
                          'finish_reason': 'EOS' if target[-1] == model.config.text_config.eos_token_id else 'MAX_TOKENS'})
    logits_path = Path(__file__).with_name('goldens') / 'smolvlm-tier-b-logits.json'
    logits_path.write_text(json.dumps({'cases': [case.pop('logits') for case in cases]}, separators=(',', ':')) + '\n')
    manifest['reference'] = {
        'status': 'candidate: measured Java error and macOS 26 acceptance pending',
        'epsilon': 1e-4, 'margin': 1e-4,
        'precision': 'CPU float32 eager; Java TF32 off',
        'image_loading': 'raw PIL.Image.open -> load -> slow Idefics3ImageProcessor',
        'versions': {'python': platform.python_version(), 'torch': torch.__version__,
                     'transformers': transformers.__version__, 'pillow': PIL.__version__, 'numpy': np.__version__},
        'requirements_lock_sha256': sha(Path(__file__).with_name('requirements.lock')),
        'generator_sha256': sha(Path(__file__)),
        'source_sha256': {cls.__name__: sha(Path(inspect.getsourcefile(cls)))
                          for cls in (type(processor), type(processor.image_processor), type(model))},
        'logits_file': {'path': str(logits_path.relative_to(repository)), 'sha256': sha(logits_path)},
        'cases': cases}
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    provenance_path = Path(__file__).with_name('provenance.json')
    provenance = json.loads(provenance_path.read_text())
    paths = [manifest_path.resolve(), Path(__file__).resolve()]
    paths += [(repository / image['path']).resolve() for image in manifest['images']]
    provenance['files'][logits_path.name] = sha(logits_path)
    provenance['vision_tier_b_files'] = {str(path.relative_to(repository)): sha(path) for path in paths}
    provenance_path.write_text(json.dumps(provenance, indent=2) + '\n')
    print('minimum gap:', min(g for c in cases for g in c['gaps']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    generate(args.directory, args.manifest)
