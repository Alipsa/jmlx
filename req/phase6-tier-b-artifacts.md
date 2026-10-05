# Phase 6 Tier-B artifact manifest

Tier-B checks use named real Hugging Face artifacts and are manual or scheduled only. They are not
required pull-request checks, never download credentials or multi-gigabyte weights implicitly, and
must record every field below before an artifact is accepted as compatibility evidence.

| Architecture | Repository candidate | Revision | File hashes | License / access | Expected output and runtime pin | Download / peak memory | Trigger owner | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Llama | [HuggingFaceTB/SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) | `12fd25f77366fa6b3b4b768ec3050bf629380bac` | all consumed files in [`tools/tier-b/smollm2-135m-instruct.json`](../tools/tier-b/smollm2-135m-instruct.json) | Apache-2.0; public | `model_type=llama`; exact 16 chat-generated IDs, chat text/IDs, and runtime pin in machine manifest | 271,169,733 bytes consumed; 300,000,000-byte cap; highest sampled test-JVM RSS 461,424 KiB | project maintainer | verified-with-real-artifact; [exact-token run](https://github.com/Alipsa/jmlx/actions/runs/36640308134) |
| Qwen2 | [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | `7ae557604adf67be50417f59c2c2f167def9a775` | all consumed files in [`tools/tier-b/qwen2.5-0.5b-instruct.json`](../tools/tier-b/qwen2.5-0.5b-instruct.json) | Apache-2.0; public | `model_type=qwen2`; exact 16 chat-generated IDs, chat text/IDs, and runtime pin in machine manifest | 995,137,433 bytes consumed; 1,100,000,000-byte cap; highest sampled test-JVM RSS 1,018,112 KiB | project maintainer | verified-with-real-artifact; [exact-token run](https://github.com/Alipsa/jmlx/actions/runs/36640308134) |
| Qwen3 | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | `c1899de289a04d12100db370d81485cdf75e47ca` | all consumed files in [`tools/tier-b/qwen3-0.6b.json`](../tools/tier-b/qwen3-0.6b.json) | Apache-2.0; public | `model_type=qwen3`; exact 16 chat-generated IDs, chat text/IDs, and runtime pin in machine manifest | 1,514,733,440 bytes consumed; 1,520,000,000-byte cap; highest sampled test-JVM RSS 867,520 KiB | project maintainer | verified-with-real-artifact; [observed run](https://github.com/Alipsa/jmlx/actions/runs/37301827462) |
| Mistral | [OuteAI/Lite-Mistral-150M-v2-Instruct](https://huggingface.co/OuteAI/Lite-Mistral-150M-v2-Instruct) | `6e2f90cbf312921289b3b8c29cb34cbdb6cb708e` | all consumed files in [`tools/tier-b/lite-mistral-150m-v2-instruct.json`](../tools/tier-b/lite-mistral-150m-v2-instruct.json) | Apache-2.0; public | `model_type=mistral`; exact 16 chat-generated IDs, chat text/IDs, and runtime pin in machine manifest | 627,886,087 bytes consumed; 650,000,000-byte cap; highest sampled test-JVM RSS 802,064 KiB | project maintainer | verified-with-real-artifact; [exact-token run](https://github.com/Alipsa/jmlx/actions/runs/36640308134) |
| Gemma v1 | [google/gemma-2b](https://huggingface.co/google/gemma-2b) | — | — | Gemma terms acceptance and token required | — | about 5.02 GB weights plus runtime; peak unmeasured | project maintainer | candidate pending |
| Phi-3 | [microsoft/Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) | — | — | MIT; public | — | index reports 7,642,159,104 bytes of weights, exceeding hosted runner RAM before runtime overhead | self-hosted runner owner pending | candidate pending |
| Mixtral | [M4-ai/TinyMistral-6x248M](https://huggingface.co/M4-ai/TinyMistral-6x248M) evaluated; smaller runnable candidate pending | — | — | Apache-2.0; public | — | 4.01 GB weight file; hosted-runner peak unmeasured, dense expert execution cost unmeasured | project maintainer | candidate pending |

On 2026-09-29, [GitHub's hosted runner specification](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)
listed `macos-26` arm64 as an M1 runner with **7 GB RAM and 14 GB SSD**. The Phi-3 weight total above
comes from the [model's safetensors index](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/blob/main/model.safetensors.index.json).
The [pinned Qwen Instruct weight file](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/7ae557604adf67be50417f59c2c2f167def9a775/model.safetensors) is
988,097,824 bytes and publishes the SHA-256 recorded in the machine-readable manifest. The three
small consumed JSON files were downloaded at that revision and hashed locally. Gemma's [gated repository](https://huggingface.co/google/gemma-2b) requires accepting its
terms; its file listing shows a 4.95 GB shard and a 67.1 MB shard. Qwen's exact-token run used
`Apple M1 (Virtual)`, macOS 26.6.2, aarch64, mlx-metal 0.31.2, and mlx-c
`fba4470b89073180056c9ea46c443051375f7399`; it produced the 16 IDs in the machine
manifest. Its highest sampled **test JVM RSS** across recorded runs was 1,018,112 KiB. That
measurement does not include separate Gradle processes or all unified GPU allocations; the
successful hosted-runner load is the practical fit evidence. Llama's exact-token run used the
same recorded Apple M1/macOS/MLX pin; the highest sampled test-JVM RSS across its recorded runs
was 461,424 KiB. Mistral's exact-token run used the same runner pin, asserted the 16 IDs in
its machine manifest, and sampled a peak test-JVM RSS of 802,064 KiB. Qwen3's hosted run logged the 16 IDs in its machine manifest on the Apple M1 (Virtual),
macOS 26.6.2 runner pin, with a peak sampled test-JVM RSS of 867,520 KiB; that run
predated this manifest's pin update, so it passed structurally, and the recorded pin now
enables the exact-ID assertion on subsequent matching hosted runs. A local Apple M5 Max run
on the same checkpoint asserted the identical 16 IDs on its recorded pin. The Tier-B generation
requests use the same rendered chat IDs checked before model loading. The remaining rows are
candidate evaluations, not Tier-B evidence.

The opt-in `tierBTest` source set accepts a local checkpoint. The scheduled Tier-B workflow
downloads only pinned public Llama, Qwen, Qwen3, and Mistral files with per-file and total hard caps,
checks every SHA-256,
bootstraps native MLX, and asserts exact greedy IDs when its measured device/macOS/MLX pin
matches. On a different pin it reports a structural pass and does not claim token parity.

Adding a row requires an immutable repository revision and hashes for every consumed config,
tokenizer, index, and weight file. Record whether authentication or license acceptance is required,
the exact model metadata and prompt/token output asserted, the maximum download/cache size, and who
owns manual/scheduled execution. A passing Tier-A synthetic fixture is not Tier-B evidence.

`tierBTest` explicitly sets `MLX_ENABLE_TF32=0` and declares that mode as a task input.
Exact greedy-ID comparisons therefore use full float32 independently of the launching shell.
The recorded artifacts were verified on the documented device; this does not establish
identical greedy IDs across GPU generations.
