# JMLX requirements: Z-Image Base and FLUX.2 Klein Base 4B

**Status:** Proposed roadmap specification  
**Date:** 2026-11-02  
**Target:** JMLX (Java 25+, MLX on macOS/arm64)  
**Checkpoints:** `Tongyi-MAI/Z-Image` and `black-forest-labs/FLUX.2-klein-base-4B`

**Where this is scheduled.** P0 and P1 are Phase 9 milestone 9.8 in `req/full-roadmap.md`. That
milestone is decided by `req/plans/phase9-8-decision.md` and follows 9.7 (Qwen-Image-2.1). P2
(FKB-003, ART-003, ART-004) is deferred to later milestones. The decision record pins the
checkpoint revisions and settles the module home: a new `jmlx-diffusion` module, as §2 suggests.

**Which document wins.** Where this document and the roadmap or the decision record differ, the
roadmap and the decision record are authoritative. The milestone plan
(`req/plans/phase9-8-plan.md`) turns these requirements into work packages.

## 1. Objective

Extend JMLX with a **pure Java, in-process image-generation API** backed by native MLX, supporting the two named *undistilled/base* checkpoints. The immediate product use case is generating painterly artwork—especially Chinese landscape painting (shanshui), fantasy landscapes, and visionary art—from a Groovy-FX desktop application. JMLX itself must have no UI dependency.

The initial implementation should be proven on a MacBook Pro M5 Max with 128 GB unified memory; a quantized low-memory mode should then be validated on a MacBook Air M4 with **24 GB**. Python/Diffusers/MFLUX can be used as development oracles, but must not be required at runtime.

### Scope boundary

- **In scope:** text-to-image, base-model sampling and guidance, multi-seed variations, quantized inference, LoRA inference, metrics, metadata and cancellation; FLUX.2 image-conditioned editing in a later phase.
- **Out of scope for first release:** training or fine-tuning LoRAs, GUI components, HTTP server, video generation, automatic prompt enhancement, proprietary hosted models.
- **Separate model identities:** `Z-Image` is not `Z-Image-Turbo`; `FLUX.2-klein-base-4B` is not the distilled four-step `FLUX.2-klein-4B`. Never silently swap checkpoints or sampling schedules.

## 2. Architecture and API boundaries

Suggested modules (subject to JMLX's existing module conventions):

- Existing `jmlx-ffi` / `jmlx-core`: missing MLX tensor/native primitives, memory lifecycle, GPU evaluation.
- Existing `jmlx-models`: shared Qwen3 text encoder and model-discovery infrastructure.
- New `jmlx-diffusion`: image pipeline abstractions, schedulers, seeded latent generation, progress/cancellation, image/latent converters.
- Model-specific implementations: `ZImagePipeline` and `Flux2KleinPipeline` in `jmlx-models` or a distinct model adapter module.

The generic interfaces should be small. Different model families may require different schedulers, image token layouts, text conditioning and attention rules.

### Illustrative API contract (not existing code)

```java
try (ImagePipeline pipeline = ImagePipelines.load(
    "Tongyi-MAI/Z-Image", LoadOptions.builder().quantization(Q4).build())) {
  ImageResult result = pipeline.generate(ImageRequest.builder()
      .prompt("Shanshui ink wash landscape, misty mountains, ancient pines")
      .width(768).height(768)
      .steps(40).guidanceScale(4.0)
      .seed(42)
      .build());
  result.image().save(Path.of("shanshui.png"));
}
```

Step and guidance values above are **illustrative**, not universal defaults. The implementation must resolve defaults/allowed values from the particular model and expose a capability descriptor.

## 3. Requirements and acceptance criteria

Priority convention: **P0** = functional reference-parity text-to-image; **P1** = app-ready inference on 24 GB; **P2** = richer artistic workflow and editing.

### Shared foundation

**DIFF-001 [P0] — Feature/operator gap analysis**

- Pin exact upstream repository revisions and checkpoint revisions for both models.
- Produce a matrix of required native MLX operations versus JMLX's existing FFM and tensor APIs.
- Audit convolution, VAE up/downsampling, normalization, attention, positional embeddings, modulation, packed/masked sequences, concatenation, dtype conversions and random sampling.
- Add only missing primitives, using MLX-backed execution and deterministic resource ownership.
- **Accept:** the gap matrix is checked in; each new primitive has shape/dtype/numerical tests against a pinned oracle.

**DIFF-002 [P0] — Compatible model artifacts**

- Load pinned Hugging Face Diffusers configuration, tokenizer data, safetensors indexes/shards, transformer, text encoder, VAE and scheduler metadata.
- Validate tensor names, shapes and dtypes; surface explicit errors on incompatible checkpoints.
- Support local model directories and deterministic cache locations; make network download opt-in.
- **Accept:** both official full-precision checkpoints load on the M5 Max without Python, with an inventory of resolved component sizes.

**DIFF-003 [P0] — Shared Qwen3 text-encoder support**

- Both selected models use Qwen3-family conditioning, but conditioning pipelines and embedding projections must be implemented per-model.
- Reproduce correct prompt tokenization/chat templates where applicable, padding, truncation, attention masks, required hidden states and output projections.
- Reuse JMLX's existing Qwen primitives where numerically compatible; do not reuse LLM next-token inference as image conditioning by assumption.
- **Accept:** fixed prompts yield reference-comparable token IDs and conditioning tensors for **both** checkpoints.

**DIFF-004 [P0] — Diffusion/flow scheduler framework**

- Support model-defined sigma/timestep schedules, step shifting, integration/update rules, seeded initial noise, and optional classifier-free guidance (CFG).
- Validate guidance policy and default step range per checkpoint; support base-model multi-pass guidance where required.
- Keep RNG and scheduler metadata in the result for replay; define determinism tolerance for Metal/MLX version drift.
- **Accept:** fixed-seed intermediate latent trajectory and scheduler outputs match reference within documented tolerance.

**DIFF-005 [P0] — VAE and image encoding/decoding**

- Implement each checkpoint's required VAE model/config and latent scaling, dtype conversion, image normalization, output color format and PNG export.
- Support arbitrary supported aspect ratios and valid divisibility constraints; validate rather than silently changing dimensions.
- Add optional tiled/chunked VAE operation for bounded memory when needed.
- **Accept:** fixed latent tensors decode comparably to the oracle and save valid, correctly sized RGB images.

**DIFF-006 [P0] — Java generation API and native lifecycle**

- Stable request/result objects with prompt, negative prompt (if applicable), seed, dimensions, steps, guidance, model identity/revision, timings and output image.
- Reusable loaded pipeline, deterministic close, no JavaFX dependency, no Python process.
- Provide capability introspection so unsupported features fail before GPU load.
- **Accept:** an external Java 25 example generates PNGs from both models, then closes all native resources.

### Z-Image Base specific

**ZIMG-001 [P0] — Implement the Z-Image base transformer**

- Implement the model's scalable single-stream diffusion transformer (S3-DiT) architecture, including text/image token handling, attention, positional encoding, modulation and output projection according to the pinned reference.
- Verify full forward passes at latent sequence lengths corresponding to 512, 768 and 1024 px images.
- **Accept:** selected internal layers and final denoiser outputs match a reference at full precision.

**ZIMG-002 [P0] — Base checkpoint guidance and denoising**

- Implement **base** (undistilled) inference, including its model-specific timestep schedule and CFG support; do not assume Turbo's 8-NFE schedule or guidance limitations.
- Allow configurable step count and guidance; validate sensible defaults from the official model/pipeline rather than hardcoding a cross-model constant.
- **Accept:** reproducible text-to-image for fixed prompt/seed with roughly the reference's 28–50-step base workflow and guidance enabled.

**ZIMG-003 [P1] — Z-Image artistic prompt regression suite**

- Create a versioned test corpus for ink wash (水墨画), blue-green shanshui (青绿山水), literati painting, fantastical landscapes and visionary imagery.
- Track prompt adherence, composition, non-photographic texture, negative space and color palette with archived samples; compare Q4 to full precision.
- Keep subjective aesthetic evaluation separate from deterministic numerical parity.
- **Accept:** representative gallery and repeatable review protocol included in release validation.

### FLUX.2 Klein Base 4B specific

**FKB-001 [P0] — Implement the base transformer and conditioning**

- Implement the official FLUX.2 Klein **Base 4B** architecture, its Qwen3 text encoder conditioning, embedding projections, attention/modulation and output mappings.
- Respect the Base checkpoint's architecture flags and guidance conditioning; do not import distilled-model settings implicitly.
- **Accept:** per-block and full forward-pass parity on fixed text and latent inputs.

**FKB-002 [P0] — 50-step base sampling and CFG**

- Implement the undistilled model's guidance-enabled sampling and scheduler behavior; expose adjustable guidance and steps.
- Default policies must be checkpoint-specific; do not reuse distilled Klein's four-step, guidance-free behavior.
- **Accept:** generate a seeded image using the official base reference's nominal ~50-step workflow; intermediate latent comparisons pass.

**FKB-003 [P2] — Image-conditioned editing**

- Add one-or-more input images and prompt-guided editing if supported by the pinned Base pipeline. Reproduce conditioning, resolution/crop logic and input validation.
- Keep this optional in the public capability contract; text-to-image must not depend on editing support.
- **Accept:** fixed reference-image edit matches oracle conditioning tensors and produces a valid edited output without Python.

### Performance, quantization and user-facing execution

**DIFF-007 [P1] — Quantized inference on 24 GB**

- Support at least one end-to-end 4-bit model load path, quantizing **both major text-encoder and denoiser weights** as appropriate; retain precision-sensitive modules at documented dtypes.
- Choose and document the supported quantization representation. MLX affine Q4, GGUF K-quants and custom packed kernels are not interchangeable.
- Support explicit staging/offloading between conditioning, denoising and VAE and avoid unexpected CPU tensor copies.
- **Accept:** both models complete a 512×512 run on M4 Air 24 GB in an ordinary desktop session with no sustained swap thrashing, recording peak resident and Metal allocator memory. 768×768 is the next test; 1024×1024 is an optimization goal, not a P1 release blocker.

**DIFF-008 [P1] — Job progress, cancellation and GPU coordination**

- Provide async execution with stage and step progress, cancellation token and completion/failure notification.
- Run one active generation per GPU by default; serialize jobs without blocking the UI thread. Ensure a cancelled or failed job cannot poison the next run.
- **Accept:** cancel in the denoising stage and immediately run a second successful generation in the same JVM; no sustained growth in native allocations over repeated runs.

**DIFF-009 [P1] — Benchmarking and regression automation**

- Record cold/warm load, text conditioning, denoiser seconds/step, VAE, total latency, process/system memory, MLX allocations and output artifact metadata.
- Run fixed-seed numerical parity tests against pinned MFLUX/Diffusers implementations; make large-weight hardware tests optional in normal CI and explicit in Apple Silicon integration CI.
- **Accept:** a reproducible benchmark report for M5 Max (full precision and Q4) and M4 Air 24 GB (Q4), with versions, checkpoint hashes and generated image samples.

**DIFF-010 [P1] — Checkpoint identity and reproducibility**

- Persist model ID and revision, component hashes, precision/quantization, LoRAs, prompt, seed, sampler, schedule, guidance, dimensions and JMLX/MLX versions.
- Use a stable metadata schema that a desktop gallery can index; never silently apply a different checkpoint's defaults.
- **Accept:** metadata can reconstruct the request and provide a clear warning if the exact model version is unavailable.

**DIFF-011 [P1] — Packaging and external consumer contract**

- Publish model support independently of Groovy-FX. Document macOS/arm64, JDK, native MLX/Metal libraries and deployment requirements.
- Avoid bundling upstream model weights by default, while respecting Apache 2.0 attribution/NOTICE requirements.
- **Accept:** a fresh external Gradle Java project can load either checkpoint and generate an image from the public JMLX artifacts alone plus model weights.

### Artistic extensibility

**ART-001 [P1] — LoRA inference and style composition**

- Load compatible model-family LoRA adapters without modifying base model files; implement correct weight name mapping, rank/alpha scaling and supported target layers.
- Expose one or more adapters with individually configurable weights; reject adapters for wrong architecture or incompatible shape.
- Start with inference only. Full training, dynamic hot-swap performance and arbitrary multi-adapter composition can follow once single-adapter correctness is proven.
- **Accept:** one verified Z-Image style LoRA and one FLUX.2 Klein Base 4B style LoRA produce reference-comparable changes; scale 0 matches the unmodified base output within deterministic tolerance.

**ART-002 [P1] — Multi-seed variations and parameter sweeps**

- Support deterministic seed lists and multiple jobs while bounding GPU concurrency and RAM; preserve per-image metadata.
- Do not require SIMD/batched denoising to offer four-image variations: sequential requests are acceptable initially.
- **Accept:** four seeds produce four traceable images and can be resumed after a failed/cancelled job.

**ART-003 [P2] — Style preset contracts**

- Provide a serializable, model-neutral representation of style preset inputs: prompt prefix/suffix, negative prompt, recommended resolution/steps/guidance, LoRA references and strengths.
- Keep the actual style authoring and UI in the consuming application; expose capability validation and preset resolution through a thin API if useful.
- **Accept:** client application can reproduce a saved shanshui or visionary-art preset exactly given the same checkpoint and adapters.

**ART-004 [P2] — Model extensibility**

- Keep pipeline components modular enough for an additional unrelated image architecture without changing the basic generation request/result contracts.
- Preserve model-specific controls through typed options/capability descriptors rather than a growing collection of globally interpreted flags.
- **Accept:** adding a third model does not force a breaking change in existing consumer code.

## 4. Test and release gates

| Gate | Validation | Hardware |
| --- | --- | --- |
| G0 — Feasibility | Exact checkpoint/operator matrix, source revisions, memory estimate | Any developer Mac |
| G1 — Z-Image parity | Text-encoder, denoiser, scheduler and VAE parity; seeded PNG | M5 Max 128 GB |
| G2 — FLUX.2 Base parity | Same, including its Base CFG workflow | M5 Max 128 GB |
| G3 — Low memory | Q4 512² full generation with bounded memory | M4 Air 24 GB |
| G4 — App readiness | Async progress, cancel/retry, metadata, repeated jobs | M4 Air 24 GB |
| G5 — Style support | Compatible LoRAs and curated painterly corpus | Both |
| G6 — Optional editing | Reference-image conditioning and validation | M5 Max first |

For numerical correctness, agree on dtype-dependent tolerances based on reference experiments instead of promising bitwise-identical GPU output. For image quality, use a curated human-reviewed corpus **in addition to** numeric parity checks.

## 5. Recommended build order

1. **Audit** existing JMLX primitives, Qwen3 model support and official checkpoint formats (DIFF-001/002/003).
2. **Finish shared scheduler/VAE/public API** using small synthetic tests (DIFF-004/005/006).
3. **Port FLUX.2 Klein Base 4B first** (FKB-001/002), because the smaller transformer makes debugging and memory optimization easier.
4. **Port Z-Image Base** (ZIMG-001/002), validating that the abstractions are genuinely reusable.
5. **Implement end-to-end Q4 and low-memory execution** for both models (DIFF-007).
6. **Add asynchronous jobs, benchmark suite, reproducibility and packaging** (DIFF-008 through 011).
7. **Add single-model LoRA inference and artistic regression suite** (ART-001/002, ZIMG-003).
8. **Add FLUX.2 editing and advanced style application features** (FKB-003, ART-003/004).

### Definition of first usable release

An external Java program can load **each named Base checkpoint**, generate a fixed-seed 512×512 PNG using native MLX and no Python runtime, report generation progress, cancel a job safely, and repeat runs. On an M4 Air with 24 GB RAM, both models have a validated 4-bit low-memory path. The public API exposes enough metadata for a Groovy-FX gallery to reproduce images, and at least one art-style LoRA works per model.

## 6. Risks and decisions to resolve

1. **Actual JMLX coverage:** audit live code before assigning development estimates. Current JMLX operator and model implementation status may be ahead of published documentation.
2. **Quantization format:** prefer one canonical MLX Q4 representation initially; document whether prequantized community checkpoints load directly or need conversion.
3. **24 GB constraint:** model file size is not peak memory. Full Qwen3 conditioning, VAE activations, CFG and 768–1024 px latents may dominate.
4. **Reference compatibility:** pin reference versions, model revisions and test input preprocessing; upstream MFLUX/Diffusers move quickly.
5. **Licensing:** both selected official checkpoints are marked Apache 2.0, but review adapter/LoRA licenses independently before distribution.
6. **Style quality:** base models are selected for artistic flexibility, not guaranteed superiority on Chinese ink painting. Evaluate with explicit shanshui and visionary prompts.

## 7. Primary references

- Official Z-Image Base: https://huggingface.co/Tongyi-MAI/Z-Image
- Official FLUX.2 Klein Base 4B: https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B
- Official FLUX.2 implementation: https://github.com/black-forest-labs/flux2
- MFLUX Z-Image implementation: https://github.com/mflux-community/mflux/tree/main/src/mflux/models/z_image
- MFLUX FLUX.2 implementation: https://github.com/mflux-community/mflux/tree/main/src/mflux/models/flux2
- JMLX: https://github.com/Alipsa/jmlx

This is a requirements and validation specification, not an assertion that each listed feature is currently missing from JMLX.
