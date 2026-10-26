# Opt-in pinned artifact downloads

Download an immutable manifest with the Java 21 buildSrc helper:

```sh
./gradlew downloadTierBArtifact \
  -PtierBManifest=tools/tier-b/qwen3-0.6b.json -PtierBTarget=.tier-b/qwen3
JMLX_TIER_B_MODEL_DIR=.tier-b/qwen3 JMLX_TIER_B_MANIFEST=tools/tier-b/qwen3-0.6b.json \
  ./gradlew :jmlx-models:tierBTest
```

The task always verifies cached file hashes. It declares no Gradle outputs and never stores model
files in the build cache. Nested paths are supported; traversal and symlinks are rejected. HEAD and
GET follow the JDK `Redirect.NORMAL` policy, with final-response size checks and bounded streaming.
Every file has a SHA-256 and size cap; the manifest also has an aggregate cap. Failed downloads
remove partial files. Ordinary builds do not download artifacts or require Python.

Downloader tests run with `./gradlew -p buildSrc check`, independently of root `check`, and use
in-memory responses and local HTTP servers. Python remains confined to optional reference tools
and the existing CI reporting script.

SmolVLM uses `smolvlm-256m-instruct.json` and task `image-text-generation`, selecting
`TierBVisionLanguageTest` in the TF32-off JVM. The image is committed under `images/`, MIT-licensed,
and hash-pinned independently of the downloaded Apache-2.0 checkpoint. No image URL is fetched.
Structured chat, rendered text, original/expanded IDs, all four greedy IDs, every history's full
logits, top-two IDs/gaps and local peak-memory evidence are pinned. Numerical comparisons and
stable-margin eligibility always run, regardless of host; macOS 26 acceptance is still pending.

```sh
./gradlew downloadTierBArtifact \
  -PtierBManifest=tools/tier-b/smolvlm-256m-instruct.json -PtierBTarget=.tier-b/smolvlm
JMLX_TIER_B_MODEL_DIR=.tier-b/smolvlm \
  JMLX_TIER_B_MANIFEST=tools/tier-b/smolvlm-256m-instruct.json \
  JMLX_TIER_B_MODEL_TYPE=idefics3 ./gradlew :jmlx-models:tierBTest
```

Qwen3-VL uses `qwen3vl-2b-instruct.json` and task `qwen3-vl-generation`, selecting
`TierBQwen3VlTest` in the TF32-off JVM. The request is the required multi-image prompt: two
images committed under `images/` (derived from the hash-pinned image-oracle fixtures) plus
"Describe both images.", 8 greedy tokens. The reference is
`tools/hf-reference/qwen3vl_tier_b.py` (CPU float32), whose full-vocabulary logit vectors at
every expected history step are pinned as the binary golden
`tools/hf-reference/goldens/qwen3vl-tier-b-logits.bin`; both images, the manifest, the
generator and the binary are digest-pinned in `tools/hf-reference/provenance.json` and
verified by `verifyHfReferenceGoldens`. Rendered/unexpanded/expanded prompt IDs, per-image
processed-pixel hashes, the reference's own top-two gaps and the Java run's measured margins
gate the exact greedy IDs and the end-to-end `generate`'s exact decoded output
(`expected_text`). Measured local evidence (allocator peak 9,828,890,788 B, sampled
footprint peak 11,979,116,480 B, load ~843 ms, generation ~247 ms, 4.27 GB artifact disk)
sizes the suite's bounds; macOS 26 acceptance is still pending.

```sh
./gradlew downloadTierBArtifact \
  -PtierBManifest=tools/tier-b/qwen3vl-2b-instruct.json -PtierBTarget=.tier-b/qwen3vl-2b
JMLX_TIER_B_MODEL_DIR=.tier-b/qwen3vl-2b \
  JMLX_TIER_B_MANIFEST=tools/tier-b/qwen3vl-2b-instruct.json \
  JMLX_TIER_B_TASK=qwen3-vl-generation \
  JMLX_TIER_B_MODEL_TYPE=qwen3_vl ./gradlew :jmlx-models:tierBTest
```

The test samples the test process's physical footprint (libproc's `ri_phys_footprint`, the figure
the runner's RAM must actually hold, unlike `ps -o rss` which misses MLX/Metal allocations) every
20 ms and reports the highest sample alongside the native allocator peak, asserting both stay
below bounds sized from the manifest's recorded local evidence so a memory regression fails the
suite explicitly. The workflow uploads JUnit numerical/memory evidence even when
the test fails.

In CI, the SmolVLM and qwen3vl rows of the `tier-b` workflow run on `macos-26-xlarge`, a
GitHub larger runner (the qwen3vl row needs the tier because the FLOAT32-up-cast 2B
checkpoint's measured footprint peak is 11.16 GiB, past the standard runner's 7 GB; its
suite footprint bound — 12,500,000,000 bytes, ~520 MB above the measured peak — detects
a regression but is not a guarantee the tier's 14 GB survives a large regression, so
the first hosted run is acceptance). GitHub's Actions billing docs state that larger
runners are always charged per minute, even in public repositories, and that
larger-runner usage is blocked until a payment method is set up on the account
(https://docs.github.com/en/billing/concepts/product-billing/github-actions). If the
repository's account has no billing configured, those rows fail to start while the other
rows are unaffected; fixing it is an account/billing setting, not a repository change.
