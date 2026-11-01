# JMLX roadmap: diffusion image generation and Qwen-Image-2.1

Status: requirements · 2026-11-01

**Where this is scheduled.** P0 and P1 (IMG-001 to IMG-012) are Phase 9 milestone 9.7 in
`req/full-roadmap.md`. That milestone is decided by `req/plans/phase9-7-decision.md` and follows
9.6 (SD 1.5-class diffusion). P2 (IMG-013 to IMG-016) is deferred to later milestones.

**Which document wins.** Where this document and the roadmap or the decision record differ, the
roadmap and the decision record are authoritative. The milestone plan
(`req/plans/phase9-7-plan.md`) turns these requirements into work packages.

## Goal and scope

Extend JMLX with a reusable Java 25+ API for local image generation on Apple Silicon using MLX.
The first target is **Qwen-Image-2.1** (`Qwen/Qwen-Image-2.1`, revision
`d26bb61231c349cf6b7896fa83353113880e1ba3`) on a MacBook Air M4 with 24 GB of unified memory.

The implementation must:

- run in-process in the JVM through JMLX/MLX, with no mandatory Python process or HTTP
  inference service;
- remain independent of any user interface, such as a Groovy-FX image studio.

Reuse the existing JMLX tensor, FFM, neural-network, quantization and model-loading facilities
where suitable. The requirements below describe **capabilities to audit and implement**, not
claims that they are all currently absent. Decision record §3 maps each one to what exists today.

| Priority | Meaning | Where it lands |
| --- | --- | --- |
| **P0** | First reproducible text-to-image vertical slice | milestone 9.7 |
| **P1** | Usable 24 GB application backend | milestone 9.7 |
| **P2** | Editing and advanced functions | later milestones |

## Architecture

The decision record §6 decides where the code lives:

- **`jmlx-core`, `jmlx-ffi`:** existing infrastructure, plus any generally useful tensor or
  native additions.
- **`jmlx-models`:** the Qwen-Image-2.1 pipeline, built on the scheduler, sampler and
  VAE-wrapper contracts that 9.6 introduces. No new published module is created. 9.6 and 9.7
  are the first two users of the model-independent scheduler and sampler interfaces. Splitting
  them into their own module later is a mechanical move, reconsidered when an external consumer
  needs diffusion without the model families, or a third family arrives (decision record §6).
- **UI front ends:** external consumers. JMLX has no JavaFX or Groovy dependency for this.

Do not couple the generic diffusion API to a particular Qwen checkpoint or to a UI.

## P0 — first end-to-end text-to-image

### IMG-001 — Audit and extend MLX tensor operations

- **Audit:** compare the operators, dtypes, shapes and layouts of the pinned reference
  implementation against `req/mlx-api-inventory.md`. These already exist from 7.1: conv and
  conv-transpose 1-D to 3-D, `pad`, `GroupNorm`, `Upsample` (nearest/linear) and masked
  attention.
- **Already settled:** `Upsample` `NEAREST` equals the VAE's `nearest-exact` interpolation at
  integer scale factors (decision record §3). A unit test pins this.
- **Fill gaps** through tested mlx-c/FFM bindings, or narrowly scoped native kernels. Do not
  emulate large GPU operations with Java-side tensor copies.
- **Acceptance:** shape, dtype and numerical parity tests for every newly added or changed
  operator against the pinned Python MLX oracle, including nontrivial image and sequence
  lengths.

### IMG-002 — Model artifact discovery and loading

- Load the Hugging Face-style component configurations, safetensors shards and tensor indexes of
  the transformer (BF16), the Qwen3-VL text encoder (BF16) and the VAE (stored F32).
- **Support three weight-residency paths:**
  - FLOAT32, the existing up-cast, for the strict gates;
  - BF16 resident as stored;
  - Q4 resident.
  
  Decision record §4 explains the choice.
- **Text-only encoder load.** Skip the vision tower and the unused `lm_head`, which
  text-to-image never uses.
- Validate tensor names, shapes, dtypes, quantization metadata and architecture version. Report
  unsupported checkpoints clearly.
- Load from local model directories only. Downloading stays an explicit step (the existing
  `downloadTierBArtifact` pattern).
- **Acceptance:** the pinned artifact loads without Python. Deterministic diagnostics for a
  missing shard, an unsupported quantization or a mismatched shape.

### IMG-003 — Qwen-Image-2.1 text conditioning

- **Reuse 7.3b's `Qwen3VlModel`** and `HfTokenizer`. The encoder is `qwen3_vl` with the
  Qwen3-VL-8B text shape.
- **New:** an output seam returning the **last decoder layer's hidden states before the final
  RMSNorm**. This is what the pinned pipeline feeds the transformer, not the normalized
  `last_hidden_state`.
- **Reproduce the reference prompt exactly:**
  - its raw template string with the fixed system message, not `apply_chat_template`;
  - its attention mask;
  - dropping the leading system-message tokens (`_drop_idx`).
- Do not treat the model as ordinary causal text generation.
- **Acceptance:** fixed-prompt conditioning tensors agree with the pinned reference within
  explicit, dtype-appropriate tolerances.

### IMG-004 — Qwen-Image-2.1 diffusion transformer

- **Implement:**
  - the 32 single-stream blocks (32 heads × 128, `mlp_ratio` 3);
  - the timestep, image and text embeddings;
  - `causal_condition` modulation, an extra modulation row for condition tokens taken at t = 0;
  - three-axis RoPE (`axes_dims_rope` [16, 56, 56]);
  - latent packing.
- **Attention is block-causal:** `(q_idx >= kv_idx) or same_image_block`. Text tokens are
  causal, and each image block is bidirectional inside itself. Audit the attention cost at the
  sequence lengths of 512–1024 px generation.
- **Acceptance:** selected intermediate block outputs and a full denoiser forward pass match the
  reference for fixed inputs.

### IMG-005 — Scheduler and sampling loop

- **Implement `FlowMatchEulerDiscreteScheduler` as configured:**
  - dynamic exponential shift, with `mu` from the image sequence length;
  - `base_shift` 0.5, `max_shift` 0.9, `shift_terminal` 0.02.
  
  It goes behind the 9.6 scheduler contract.
- **Defaults:** 40 steps and `true_cfg_scale` 1.0 (no guidance); implement this path first.
  Step count, seed, dimensions and guidance are configurable.
- **Seeded latent initialization uses MLX random.** The reference's torch-generated noise cannot
  be reproduced, so the two are checked separately.
- **Acceptance:**
  - **Teacher-forced per step:** starting from the reference's injected initial latent, each
    compared step is fed the reference's own latent at that step. The denoiser output matches
    the reference within the recorded ε (FLOAT32 path) or within the separately named BF16 and Q4
    bounds. Errors therefore don't accumulate across steps.
  - A free-running 40-step trajectory is reported, not gated.
  - Repeated Java runs with the same seed are reproducible on a pinned runtime and hardware
    configuration.

### IMG-006 — VAE decoder and image output

- **Implement the VAE decoder:**
  - its 64-channel latent;
  - de-normalization (`latents * latents_std + latents_mean`);
  - the residual decoder whose "causal 3-D" convolutions reduce to 2-D for single images;
  - `nearest-exact` upsampling and 16× spatial scaling;
  - optional tiling.
- **The decoder outputs 4 channels (RGBA). This is the native output path.** Handle the color
  channels, numeric range, aspect ratio and alpha semantics explicitly, verified against the
  reference.
- PNG writing is new: `jmlx-vision` decodes images but has no encoder.
- **Acceptance:** a reference latent decodes to a numerically comparable image. Java saves a
  valid RGBA PNG with the expected dimensions and channel values.

### IMG-007 — Public, synchronous Java pipeline API

- An immutable request (prompt, width, height, steps, seed, guidance) against a loaded model
  object.
- **The loaded model object owns the components under a memory policy:**
  - `RESIDENT` (default): all components stay loaded between requests.
  - `STAGED`: the encoder and the transformer are never loaded together. Each request encodes,
    releases the encoder, then loads the transformer (IMG-009).
  
  Either way, the caller holds one object.
- A result carrying image pixels plus complete generation metadata.
- Manage native resources deterministically (`AutoCloseable` and `MLXScope` ownership). Surface
  failures without leaking native allocations.
- **Acceptance:** a Java 25 integration test produces a PNG from a prompt entirely in-process.

## P1 — practical use on a 24 GB M4 Air

### IMG-008 — End-to-end low-bit model support

- Implement and verify a 4-bit inference path for **both** large components, the transformer
  **and** the Qwen3-VL text encoder. Keep tensors at higher precision where the reference needs
  them.
- **Format:** jmlx's existing MLX affine Q4 (`QuantizedLinear`, group 64), converted locally from
  the pinned BF16 artifact. GGUF K-quants and custom packed formats are out of scope. Converted
  weights are never published by JMLX (decision record §5).
- **Acceptance:**
  - The Q4 path matches the CPU float32 reference within its separately named, recorded bound,
    under the same teacher-forced comparison as IMG-005.
  - It completes on the 24 GB target (IMG-009).

### IMG-009 — Bounded peak memory and lifecycle

- Measure peak process memory across loading, encoding, denoising and decoding, including the
  MLX allocator's cache and temporaries, in the existing Tier-B memory-test style.
- **Staging is a memory-budget policy, not a requirement.**
  - At Q4, the transformer (~4.0 GB), encoder (~4.4–4.9 GB) and VAE fit resident together on
    24 GB.
  - `STAGED` never holds the encoder and the transformer at the same time. It encodes, releases
    the encoder, then loads the transformer, while the VAE stays loaded. This lowers peak
    weights to the larger component plus the VAE: about 5.75 GB at Q4, 16.2 GB at BF16, and
    32.4 GB on the FLOAT32 Tier-B run. The cost is re-reading both large components every job
    (decision record §4).
- Also: optional VAE tiling, no retained intermediate history, and deterministic disposal of
  temporaries.
- Provide explicit memory-budget options and actionable out-of-memory errors.
- **Acceptance:**
  - On the project's runner, a 512×512 run's measured process footprint stays under a recorded
    bound chosen for the 24 GB target. Checkpoint size on disk is not a substitute for peak
    memory.
  - One manual run on a 24 GB M4 Air is recorded, with its measured peak.

### IMG-010 — Asynchronous jobs, progress and cancellation

- **Asynchronous generation** with stage transitions (`LOADING`, `ENCODING`, `DENOISING`,
  `DECODING`, `COMPLETE`), step counters and cancellation.
- **One platform worker thread.** All MLX work runs on a single platform worker thread, as jmlx's
  threading rule requires and `BatchGenerationScheduler` does. Jobs are therefore serialized.
  Callers submit and cancel from any thread, and callbacks never need a particular (for example
  JavaFX) thread.
- **Acceptance:**
  - Cancel during denoising, then generate another image in the same JVM, with active memory
    back at its baseline.
  - Callbacks never block the worker.

### IMG-011 — Image quality, size and performance validation

- **A pinned test matrix:**
  - 512×512 and 768×768 first, then 1024×1024;
  - fixed prompts and seeds;
  - Q4 and BF16 outputs.
- **Hosts:**
  - The M5 Max development host records every cell, Q4 and BF16.
  - The 24 GB M4 Air records only the Q4 cells. Its 512² cell is the IMG-009 manual run. Larger
    cells are recorded where they complete, or else their failure and measured peak are
    recorded.
  - BF16 does not fit the 24 GB target and is not run there.
- **Measure:** cold load, warm load, conditioning, seconds per step, VAE decoding, total
  latency, peak memory, and output quality against the reference.
- Compare with MFLUX or Diffusers on the same model and device where practical.
- **Acceptance:** reproducible benchmark runs recorded under the `req/phase6-4-benchmark.md`
  conventions (clean commit, environment versions, warmups, spread). Representative regression
  images. Quality thresholds defined before any release.

### IMG-012 — Packaging and compatibility

- The native runtime, packaging and code signing ride Phase 8.4's `jmlx-native-macos-arm64`
  artifact. This milestone adds only what image generation needs beyond that.
- **Acceptance:** `tools/release-smoke` resolves the published-shape modules and generates an
  image from a **synthetic Tier-A checkpoint** (like its existing synthetic Mistral case) without
  a Python environment, on the hosted native CI job. The 33 GB real artifact cannot run there.

## P2 — editing, broader model support and ergonomics (deferred)

### IMG-013 — Image-to-image and reference-image editing

- Encode input images through the conditioning path, which uses the Qwen3-VL vision tower
  already in jmlx. Implement reference-image preprocessing, with up to 10 references as the
  model card states.
- Composite the alpha channel over white for the vision-encoder copy only, as the reference
  does.
- Support edit strength and image-size/aspect handling where applicable. Distinguish
  strength-based img2img from instruction or reference editing.
- **Acceptance:** fixed image-plus-prompt edits match the pinned reference at selected
  intermediate tensor checkpoints.

### IMG-014 — Masks, transparency and targeted editing

- External masks, local edits and inpainting, after validating the model's mask conventions end
  to end. RGBA output itself is P0 (IMG-006).
- **Acceptance:** an edit leaves the required unmasked regions unchanged within a documented
  tolerance, and preserves the intended alpha semantics.

### IMG-015 — Reusable diffusion abstractions

- Separate the scheduler, denoiser, conditioning, VAE and postprocessor interfaces so that
  further image models fit without pretending their tokenization, attention or schedulers are
  identical. The 9.6 and 9.7 pipelines are the first two users.
- **Acceptance:** a third, architecturally distinct image model is integrated without changing
  the public request/result contract. That is one of the triggers to revisit the module home
  (decision record §6).

### IMG-016 — Optional fast and advanced inference

- **Evaluate:**
  - Qwen-Image-2.1-Turbo (`fc13ba1`, also under the Qwen Research License, a few-step schedule
    shipped in its config);
  - quantized attention kernels, LoRA, decode tiling and ControlNet. Warm model reuse is
    already P1 (the `RESIDENT` policy, IMG-007).
  
  Prompt-embedding caching follows 9.2's prefix-cache design.
- Treat Turbo schedules and checkpoints as model-specific, not interchangeable with base
  sampling.
- **Acceptance:** each opt-in feature shows verified correctness and a memory or performance gain
  in repeatable tests.

## Nonfunctional requirements (all milestones)

- **Java-first:** public execution needs no Python. Python is used only for development,
  conversion and reference generation, in the hash-locked `tools/image-gen-reference/` tool
  (decision record §7), which is never Gradle-invoked.
- **Portability:** keep reusable APIs free of Apple specifics, while stating that the backend is
  macOS/arm64 MLX.
- **Resource safety:** no leaks across repeated generations, failures, cancellation or model
  reloads. Test long-lived processes.
- **Observability:** log model and component versions, quantization, seed, step count, timings
  and memory. Do not log prompts by default without the caller's consent.
- **Determinism:** store enough metadata to reproduce a run. Document allowable numerical drift
  across MLX versions and hardware.
- **Testing:**
  - A strict Tier-A synthetic golden is the binding gate:
    - Its generated weights are stored as BF16, as in 7.3b WP8 and 7.3c WP6. The strict gate
      runs through the FLOAT32 up-cast, and the same checkpoint covers BF16-resident loading.
    - Its tokenizer is a committed tiny fixture tokenizer with the template's special tokens and
      a minimal chat template, so the raw template and `_drop_idx` are exercised.
  - The real artifact is a pinned, opt-in Tier-B suite. Its goldens are capped at 10 MiB.
  - Strict float32 references use the `full-float32` tag; BF16 and Q4 comparisons carry
    separately named bounds.
- **Licensing:** Qwen-Image-2.1 is under the Qwen Research License Agreement (non-commercial
  research and evaluation only; notice on redistribution; attribution under §4.b).
  - Unlike SD 1.5's license, it forbids commercial use outright. The verified checkpoint
    therefore needs a separate license from the model owner for any commercial use, and the
    documentation says so.
  - No file from the artifact is committed: weights, configs, processor or tokenizer files, or
    converted weights. JMLX never bundles, publishes or redistributes them (decision record
    §5).

## Dependency order and gates

1. **Decision record** (`req/plans/phase9-7-decision.md`): license, module home, reference
   toolchain. Decided before the milestone plan.
2. **9.6 delivered:** the scheduler, sampler and VAE contracts that 9.7 extends.
3. **Reference and probes:** the reference environment, plus the probes listed in the decision
   record §8 (WP1).
4. **Numerical components:** IMG-001 to IMG-006, with intermediate parity tests and the Tier-A
   golden.
5. **Vertical slice:** IMG-007, one 512×512 image in Java. A larger development Mac is acceptable
   at this stage.
6. **24 GB target:** IMG-008 and IMG-009.
7. **Application-ready:** IMG-010 to IMG-012.
8. **Later milestones:** IMG-013 to IMG-016.

**First demonstrable success:** a single Java `main()` loads the pinned checkpoint, generates a
seeded 512×512 image through native MLX, saves PNG plus metadata, and passes the resource-cleanup
and reference-parity checks.

## References

- JMLX: https://github.com/Alipsa/jmlx
- Model and license, pinned at `d26bb61231c349cf6b7896fa83353113880e1ba3`:
  https://huggingface.co/Qwen/Qwen-Image-2.1
- Diffusers pipeline docs: https://huggingface.co/docs/diffusers/en/api/pipelines/qwenimage21.
  Pinned source: `diffusers==0.42.1`, identical to commit
  `86ef420b17363c28e36ebad6b58555e9aa8feda2` for the Qwen-Image-2.1 pipeline, transformer and
  VAE files. 0.41.0's transformer differs.
- MFLUX Qwen 2.1 implementation: https://github.com/mflux-community/mflux/tree/main/src/mflux/models/qwen21
  (not checked; pin a commit before use).
- Community packed-Q4 prototype (custom runner, not drop-in; provenance and revision unpinned,
  informational only): https://huggingface.co/Nurymanau/Qwen-Image-2.1-Q4_K_M-MLX

These references are implementation guides. They do not guarantee that JMLX can load their
checkpoints unchanged.
