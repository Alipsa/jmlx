# Offline Hugging Face reference goldens

This is a generate-only Python tool. Gradle and CI read the committed JSON and
safetensors files and verify their hashes; they do not install or run Torch.
The references come from Hugging Face `transformers` on CPU, in float32, with
eager attention. `generate.py` seeds Torch, constructs tiny two-layer models,
saves each checkpoint with `save_pretrained`, and records full prompt logits and
two greedy decode steps. `rope.json` records Hugging Face inverse frequencies,
attention scaling, and rotary output at several offsets.

Python 3.12 is required. Regeneration is a reviewed, manual step:

```sh
cd tools/hf-reference
python3.12 -m venv .venv
.venv/bin/python -m pip install --require-hashes \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  -r requirements.lock
.venv/bin/python generate.py --family all --out goldens
cd ../..
./gradlew verifyHfReferenceGoldens
```

`--family` also accepts one family name or `rope` for a targeted regeneration.
Phase 7.2 has a separate seed-preserving entry point:

```sh
tools/hf-reference/.venv/bin/python tools/hf-reference/generate.py \
  --family phase72 --out tools/hf-reference/goldens
```

It produces BERT encoder/sequence/token heads and tied-ReLU/untied-gated T5 checkpoints,
full/cached logits, encoder states, buckets and greedy outputs. Source hashes and executable
semantic probes are recorded in `phase72-semantics.json` and provenance. The universal hash
lock retains Linux Torch `2.6.0+cpu`, uses Torch `2.6.0` on Darwin, and adds the actual
`sentence-transformers==5.1.2` pipeline. Existing decoder goldens are unchanged.

Generate reviewed candidates for locally downloaded real artifacts with:

```sh
tools/hf-reference/.venv/bin/python tools/hf-reference/tier_b.py \
  --directory .tier-b/minilm-l6-v2 --manifest tools/tier-b/minilm-l6-v2.json
```

The command also accepts classification and seq2seq manifests. It verifies artifact hashes,
checks installed versions, and records CPU float32 eager vectors/logits, source hashes and
top-two gaps. Regeneration resets acceptance. Rerun Java Tier-B, measure the real error and
verify `gap > 2 * epsilon + margin` before recording final eligibility. A multi-label
threshold would require `abs(logit) > epsilon + margin`. Python execution stays opt-in.
To regenerate only the additional Mistral window-boundary references from the committed
checkpoint without changing `mistral.json`, run
`tools/hf-reference/.venv/bin/python tools/hf-reference/generate.py --family mistral --window-cases --out tools/hf-reference/goldens`
from the repository root. The new file stays within the fixture's 128-position context limit.
The generator checks every saved tensor key against a Python-owned Hub-style
manifest. Keep that manifest in sync with `ArchitectureMappings.tensorPlan`;
the Java golden tests are the final cross-check. A mismatch stops generation.

`provenance.json` records versions, the Transformers source commit and RoPE
source hash, host OS and architecture, and SHA-256 of **every** file under
`goldens/`. A dependency repin should produce a reviewed golden diff, including
checkpoint and numerical changes. Regeneration on another host may produce
different float32 values and needs the same review. Do not edit JSON or weights
by hand.

### Chat references

Run `HF_HUB_OFFLINE=1 .venv/bin/python generate.py --chat --family all --out goldens`
from the repository root (or supply `--tokenizer-root`). This reads only the committed synthetic
family bundles under `jmlx-tokenizer/src/test/resources/families`. It writes
`goldens/chat-{mistral,gemma,phi3,mixtral}.json`, pins their hashes in `provenance.json`, and
records hashes of the input tokenizer bundles in `chat_sources`. The existing model and RoPE
goldens are left unchanged.

### SmolVLM real-artifact reference

Generate a reviewed candidate from the SHA-verified local pinned artifact:

```sh
tools/hf-reference/.venv/bin/python tools/hf-reference/vision_tier_b.py \
  --directory .tier-b/smolvlm-256m-instruct \
  --manifest tools/tier-b/smolvlm-256m-instruct.json
```

This uses raw `PIL.Image.open -> load` and the explicit slow Idefics3 image processor, CPU float32
and eager attention. It records structured chat, original/expanded IDs, greedy histories, full raw
logits, gaps, pixel/source/template/checkpoint hashes and environment versions. Compact full logits
live in `goldens/smolvlm-tier-b-logits.json`; the manifest pins that file's digest. It updates the
additive vision Tier-B provenance entries without rewriting existing synthetic goldens. The root
`verifyHfReferenceGoldens` verifies both the new golden and manifest/image/generator digests with no
Python execution. Regeneration resets numerical acceptance: rerun Java, review the changed outputs
and record the measured error/memory before setting local eligibility again.

### Qwen3-VL processed-patch fixtures (Phase 7.3b WP2)

The two WP1 probe images (256x256 and 64x32) are committed through the pinned
transformers 4.57.6 slow `Qwen2VLImageProcessor` file-to-processor pipeline --
`exif_transpose`, `convert("RGB")`, `smart_resize`, BICUBIC resize, rescale and
normalize, and the merge-block patch flatten -- at the jmlx planned-effective
bounds `min_pixels=65536` / `max_pixels=1048576` (the artifact declares
`max_pixels=16777216`). They live under
`tools/image-oracle/fixtures/qwen3vl/patches/` as two little-endian IEEE-754
float32 bins of shape `[rawPatchRows, 1536]` (256 and 288 rows) plus
`manifest.json`, which records the schema, per-patch row counts, `gridThw` and
the source-image digests.

The fixtures are owned by `tools/image-oracle`, never by this tool: only its
runner produces and verifies them (`./gradlew generateImageOracleFixtures`
rewrites, `./gradlew verifyImageOracleFixtures` verifies). Their digests are
pinned twice: in `tools/image-oracle/provenance.json` (enforced by the root
`verifyImageOracleFixtureDigests`) and in this `provenance.json` under
`qwen3vl_processed_patches` (enforced by the root `verifyHfReferenceGoldens`
with the same set-equality and path-traversal guards as the vision Tier-B
files).

Cross-tool agreement (verified 2026-10-18): the two bin digests are exactly the
`pixel_values.sha256` of the `rgb_256x256` and `rgb_64x32` cases in
`goldens/qwen3-vl-2b-wp1-probe.json`, which was produced on the pinned 2B
artifact at the declared max bound of 16,777,216 -- bit-level agreement across
the bound choice, because neither geometry is affected by `max_pixels`: the
256x256 image sits exactly at `min_pixels` with no scaling, and the 64x32 image
takes the upscale branch under either bound.

Consumption contract for the Phase 7.3b WP5+ model-math tests: feed these bins
directly as `pixel_values` (with the manifest's grids) without re-preprocessing,
so a model-math error cannot be concealed by a preprocessing drift.
