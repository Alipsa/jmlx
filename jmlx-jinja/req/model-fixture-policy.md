# Model fixture licensing policy

This policy satisfies WP1a before the differential corpus imports any model chat-template text.
It is a repository intake policy, not legal advice. Every future fixture change must identify its
exact model repository, immutable revision, template path, and applicable license in `NOTICE`.

## Approved fixture forms

| Exact model repository and revision | Current license basis | Repository policy |
| --- | --- | --- |
| `Qwen/Qwen2.5-32B-Instruct` at `afb2829595f63efa3548e9d6b13aa66e61aa0f38` | Apache-2.0 | Template text may be retained after recording the template path, Apache-2.0 notice, and model-card attribution. |
| `mlx-community/Qwen3.8-27B-4bit` at `3e6447f082e89cc7f0bc6e5441afd38dfce760ff` | Apache-2.0 MLX conversion of `Qwen/Qwen3.8-27B` | Template text may be retained after recording the `chat_template.jinja` path, Apache-2.0 notice, upstream-base relationship, and model-card attribution. This is the primary retained Qwen fixture. |
| `mistralai/Mistral-7B-Instruct-v0.3` at `c170c708c41dac9275d15a8fff4eca08d52bab71` | Apache-2.0 | Template text may be retained after recording the template path, Apache-2.0 notice, and model-card attribution. |
| `stepfun-ai/step3` at `7bf55112c8b477c47f91ed7c5872a5a80015b099` | Apache-2.0 | Template text and self-authored rendered output may be retained after recording the `chat_template.json` path, Apache-2.0 notice, and model-card attribution. |
| All other Qwen models, including Qwen2.5 3B and 72B variants | Not preapproved | Do not retain template text or rendered output. A hash-only case is permitted; text or output requires a fixture-specific review. |
| All Llama models | Model-version-specific Llama Community License | Do not retain template text or rendered output by default. A hash-only case may retain reviewed source revision/path, the template SHA-256, either the successful-output SHA-256 or an error category, and self-authored test context. Text or output may be added only after a separate review confirms the applicable license, attribution, redistribution, naming, and acceptable-use terms. |
| Any repository-and-revision pair not listed above | Not preapproved | Do not retain template text or rendered output. A hash-only case following the Llama form is permitted; text or output requires a fixture-specific review that adds an exact repository-and-revision row here and records its notice requirements. |

## Current fixture set

The retained Mistral-7B-Instruct-v0.3, Qwen2.5-32B-Instruct, and primary Qwen3.8-27B-4bit MLX
template resources are approved Qwen/Mistral cases. Their pinned source records, Apache-2.0
notices, and model-card attribution are recorded in NOTICE in this same change; copied templates
and generated output are not implicitly covered by the upstream MIT notice. The model-bearing
upstream README and e2e fixture remain intentionally unvendored. The first Llama case may retain
only hash-only metadata, self-authored test context, and an error category where applicable.

The retained Step3 resource is the approved macro-heavy fixture. Its pinned source record,
Apache-2.0 notice, and model-card attribution were added with its first retained copy in NOTICE.

## Sources reviewed

Reviewed 2026-08-19:

- [Qwen/Qwen2.5-32B-Instruct license at `afb2829`](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/blob/afb2829595f63efa3548e9d6b13aa66e61aa0f38/LICENSE)
  — Apache-2.0.
- [Qwen/Qwen2.5-32B-Instruct model card at `afb2829`](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/blob/afb2829595f63efa3548e9d6b13aa66e61aa0f38/README.md)
  — attribution source.
- [mistralai/Mistral-7B-Instruct-v0.3 at `c170c70`](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/tree/c170c708c41dac9275d15a8fff4eca08d52bab71)
  — directly confirmed as a Git commit; its `README.md` metadata declares Apache-2.0.
- [mistralai/Mistral-7B-Instruct-v0.3 model card at `c170c70`](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/blob/c170c708c41dac9275d15a8fff4eca08d52bab71/README.md)
  — attribution source.
- [Meta Llama 3.1 Community License](https://www.llama.com/llama3_1/license/)
  — used only to establish the review-required default; no Llama material is retained.

Reviewed 2026-08-24:

- [stepfun-ai/step3 LICENSE at `7bf5511`](https://huggingface.co/stepfun-ai/step3/blob/7bf55112c8b477c47f91ed7c5872a5a80015b099/LICENSE)
  — 11,357-byte Apache-2.0 license text.
- [stepfun-ai/step3 model card at `7bf5511`](https://huggingface.co/stepfun-ai/step3/blob/7bf55112c8b477c47f91ed7c5872a5a80015b099/README.md)
  — §License states that both the code repository and model weights are Apache-2.0; attribution source.
- [stepfun-ai/step3 chat template at `7bf5511`](https://huggingface.co/stepfun-ai/step3/blob/7bf55112c8b477c47f91ed7c5872a5a80015b099/chat_template.json)
  — immutable retained source; its decoded `chat_template` field was reviewed for fixture intake.

Reviewed 2026-08-26:

- [mlx-community/Qwen3.8-27B-4bit at `3e6447f`](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit/tree/3e6447f082e89cc7f0bc6e5441afd38dfce760ff)
  — immutable verified commit; the model card metadata declares Apache-2.0 and identifies
  `Qwen/Qwen3.8-27B` as the base model for this MLX conversion.
- [mlx-community/Qwen3.8-27B-4bit model card at `3e6447f`](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit/blob/3e6447f082e89cc7f0bc6e5441afd38dfce760ff/README.md)
  — Apache-2.0 and attribution source.
- [mlx-community/Qwen3.8-27B-4bit chat template at `3e6447f`](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit/blob/3e6447f082e89cc7f0bc6e5441afd38dfce760ff/chat_template.jinja)
  — immutable retained source.
