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
