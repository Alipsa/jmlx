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

### Qwen3-VL interleaved multi-axis RoPE cross-check (Phase 7.3b WP4)

`qwen3vl_rope_interleaved_check.py` and its Java twin
`qwen3vl_rope_interleaved_check.jsh` are an opt-in pair, never invoked by
Gradle, that writes no golden: the Python probe runs the real pinned
`Qwen3VLTextRotaryEmbedding` plus `apply_rotary_pos_emb` (transformers 4.57.6,
CPU float32, single-threaded, deterministic) on fixed, index-coded asymmetric
cases -- prefill, prefill at the pinned maximum coordinate (89), decode at the
all-axes-equal shape, decode at the negative-delta coordinate (26), per-row
batched positions, and small sections that exercise every interleaved `f % 3`
owner -- and prints a deterministic full-precision float32 case table to
stdout. It exits non-zero if its internal gates fail: the pinned snapshot's
`text_config` must carry theta 5,000,000 / head_dim 128 /
`mrope_section [24, 20, 20]` / `mrope_interleaved true` / rope type default;
each module's `inv_freq` must be bit-exact against the documented float32
derivation order; each case's explicit owner-gather recomputation must be
bit-exact against the module's own `apply_interleaved_mrope` output; the
equal-axes case's 2-D position-ids path must equal its 3-D path; the q and k
outputs must be bit-equal; and a double run must reproduce the table's
SHA-256.

The Java twin reads that table through `JMLX_ROPE_REFERENCE` (jshell has no
argv), runs the frozen `se.alipsa.jmlx.nn.MultiAxisRope` component over every
case, and requires a max absolute error at or below 1e-5 against the
reference rotate-half output (plus bit-identical rotate self-consistency).
The pinned transformers output is authoritative: a divergence means the Java
component is fixed, not the probe. Run both from the repository root:

```sh
./gradlew :jmlx-core:jar :jmlx-ffi:jar
tools/hf-reference/.venv/bin/python \
  tools/hf-reference/qwen3vl_rope_interleaved_check.py \
  > build/qwen3vl-rope-reference.txt
JMLX_LIBRARY_PATH=$PWD/native/install/lib \
JMLX_ROPE_REFERENCE=$PWD/build/qwen3vl-rope-reference.txt \
  "$JDK25_HOME/bin/jshell" --execution local -s -J--enable-native-access=ALL-UNNAMED \
  --class-path jmlx-core/build/libs/jmlx-core-0.5.0-SNAPSHOT.jar:jmlx-ffi/build/libs/jmlx-ffi-0.5.0-SNAPSHOT.jar \
  tools/hf-reference/qwen3vl_rope_interleaved_check.jsh
```

where `$JDK25_HOME` is the Java 25 toolchain home the build uses. The Python
probe exits non-zero on any gate failure; the Java twin exits 0 when every
case is within 1e-5, 1 when any case exceeds the threshold or a rotate is not
self-consistent, 2 on a missing or unrecognized table, and 3 on any other
failure. Each per-case max error is printed. `--execution local` runs the
snippets inside the jshell process so the `System.exit` verdict is the probe's
exit code (the default child-engine mode swallows it).

### Wide Qwen3-VL attention regression

`qwen3vl_wide_attention.py` reuses the pinned `qwen3vl-model` checkpoint and
changes query heads from 4 to 8 (hidden 256, attention width 512). It replaces
q_proj and o_proj with distinct asymmetric index-coded weights, exactly
representable in BF16. This avoids duplicating the checkpoint or depending on
matching Python/Java random generators. The four CPU float32 eager-reference
cases cover text and image prefill plus full-history decode recomputation;
Java compares its cached decode against those logits at the existing 1e-4
tolerance in `Qwen3VlModelFloatGoldenTest`. The generator and its output are
digest-pinned and checked by `verifyHfReferenceGoldens`.

Regenerate manually in the locked environment and review the golden/pin diff:

```sh
tools/hf-reference/.venv/bin/python tools/hf-reference/qwen3vl_wide_attention.py
```

### Phase 7.4 audio/diffusion decision probes

`whisper_logmel_probe.py` and `diffusion_unet_probe.py` back the Phase 7.4
capability decision (`req/plans/phase7-4-decision.md`). They are opt-in, never
invoked by Gradle, and write no golden: the audio probe synthesizes a
deterministic 30 s 16 kHz waveform and runs the Whisper log-mel front end
through the locked transformers 4.57.6; the diffusion probe forwards a tiny
SD-shaped UNet and VAE decoder in torch and maps every exercised primitive to
jmlx's surface. Each prints a JSON findings document that the decision record
pastes verbatim. Both scripts are digest-pinned in `provenance.json` and enforced by
root `verifyHfReferenceGoldens`, so an edit to a probe fails the build rather than silently
invalidating the decision record's evidence; the universal hash lock is unchanged and no new
dependency was added. Run manually from the repository root:

```sh
tools/hf-reference/.venv/bin/python tools/hf-reference/whisper_logmel_probe.py
tools/hf-reference/.venv/bin/python tools/hf-reference/diffusion_unet_probe.py
```
