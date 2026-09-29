# Phase 6 Tier-B artifact manifest

Tier-B checks use named real Hugging Face artifacts and are manual or scheduled only. They are not
required pull-request checks, never download credentials or multi-gigabyte weights implicitly, and
must record every field below before an artifact is accepted as compatibility evidence.

| Architecture | Repository candidate | Revision | File hashes | License / access | Expected output and runtime pin | Download / peak memory | Trigger owner | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Llama | pending a small licensed Llama-architecture artifact | — | — | pending | — | unmeasured | project maintainer | candidate pending |
| Qwen2 | [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | `7ae557604adf67be50417f59c2c2f167def9a775` | all consumed files in [`tools/tier-b/qwen2.5-0.5b-instruct.json`](../tools/tier-b/qwen2.5-0.5b-instruct.json) | Apache-2.0; public | `model_type=qwen2`; exact token IDs and device/macOS/MLX pin pending first run | 988,097,824-byte weight; 1,100,000,000-byte total cap; peak resident memory unmeasured | project maintainer | candidate pending; scheduled structural probe |
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
terms; its file listing shows a 4.95 GB shard and a 67.1 MB shard. Qwen now has an immutable revision
and consumed-file hashes; no row has measured peak resident memory or a macOS/MLX-pinned output yet.
These are evaluations, not Tier-B evidence.

The opt-in `tierBTest` source set accepts a local checkpoint. The scheduled Tier-B workflow now
downloads only the pinned public Qwen candidate with per-file and total hard caps, checks every
SHA-256, bootstraps native MLX, and runs structural assertions. Its exact-token mode remains
unused until measured peak memory, generated IDs, and the device/macOS/MLX pin are recorded.

Adding a row requires an immutable repository revision and hashes for every consumed config,
tokenizer, index, and weight file. Record whether authentication or license acceptance is required,
the exact model metadata and prompt/token output asserted, the maximum download/cache size, and who
owns manual/scheduled execution. A passing Tier-A synthetic fixture is not Tier-B evidence.
