# Phase 6 Tier-B artifact manifest

Tier-B checks use named real Hugging Face artifacts and are manual or scheduled only. They are not
required pull-request checks, never download credentials or multi-gigabyte weights implicitly, and
must record every field below before an artifact is accepted as compatibility evidence.

| Architecture | Repository candidate | Revision | File hashes | License / access | Expected output and runtime pin | Download / peak memory | Trigger owner | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Llama | pending a small licensed Llama-architecture artifact | — | — | pending | — | unmeasured | project maintainer | candidate pending |
| Qwen2 | [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | `7ae557604adf67be50417f59c2c2f167def9a775` | all consumed files in [`tools/tier-b/qwen2.5-0.5b-instruct.json`](../tools/tier-b/qwen2.5-0.5b-instruct.json) | Apache-2.0; public | `model_type=qwen2`; exact 16 IDs and full device/macOS/MLX pin in machine manifest | 995,137,433 bytes consumed; 1,100,000,000-byte cap; highest sampled test-JVM RSS 1,018,112 KiB | project maintainer | verified-with-real-artifact; [exact-token run](https://github.com/Alipsa/jmlx/actions/runs/36629740554) |
| Mistral | pending a ≤1B Mistral-architecture artifact | — | — | pending | — | unmeasured | project maintainer | candidate pending |
| Gemma v1 | [google/gemma-2b](https://huggingface.co/google/gemma-2b) | — | — | Gemma terms acceptance and token required | — | about 5.02 GB weights plus runtime; peak unmeasured | project maintainer | candidate pending |
| Phi-3 | [microsoft/Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) | — | — | MIT; public | — | index reports 7,642,159,104 bytes of weights, exceeding hosted runner RAM before runtime overhead | self-hosted runner owner pending | candidate pending |
| Mixtral | pending a small licensed Mixtral-architecture artifact | — | — | pending | — | unmeasured | project maintainer | candidate pending |

On 2026-09-29, [GitHub's hosted runner specification](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)
listed `macos-26` arm64 as an M1 runner with **7 GB RAM and 14 GB SSD**. The Phi-3 weight total above
comes from the [model's safetensors index](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/blob/main/model.safetensors.index.json).
The [pinned Qwen Instruct weight file](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/7ae557604adf67be50417f59c2c2f167def9a775/model.safetensors) is
988,097,824 bytes and publishes the SHA-256 recorded in the machine-readable manifest. The three
small consumed JSON files were downloaded at that revision and hashed locally. Gemma's [gated repository](https://huggingface.co/google/gemma-2b) requires accepting its
terms; its file listing shows a 4.95 GB shard and a 67.1 MB shard. Qwen's exact-token run used
`Apple M1 (Virtual)`, macOS 26.6.2, aarch64, mlx-metal 0.31.2, and mlx-c
`fba4470b89073180056c9ea46c443051375f7399`; it produced the 16 IDs in the machine
manifest. Its highest sampled **test JVM RSS** across two runs was 1,018,112 KiB. That
measurement does not include separate Gradle processes or all unified GPU allocations; the
successful hosted-runner load is the practical fit evidence. The remaining rows are candidate
evaluations, not Tier-B evidence.

The opt-in `tierBTest` source set accepts a local checkpoint. The scheduled Tier-B workflow
downloads only pinned public Qwen files with per-file and total hard caps, checks every SHA-256,
bootstraps native MLX, and asserts exact greedy IDs when its measured device/macOS/MLX pin
matches. On a different pin it reports a structural pass and does not claim token parity.

Adding a row requires an immutable repository revision and hashes for every consumed config,
tokenizer, index, and weight file. Record whether authentication or license acceptance is required,
the exact model metadata and prompt/token output asserted, the maximum download/cache size, and who
owns manual/scheduled execution. A passing Tier-A synthetic fixture is not Tier-B evidence.
