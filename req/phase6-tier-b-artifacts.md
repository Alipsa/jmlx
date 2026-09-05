# Phase 6 Tier-B artifact manifest

Tier-B checks use named real Hugging Face artifacts and are manual or scheduled only. They are not
required pull-request checks, never download credentials or multi-gigabyte weights implicitly, and
must record every field below before an artifact is accepted as compatibility evidence.

| Architecture | Repository | Revision | File hashes | License | Access requirement | Expected metadata/output | Cache and size policy | Trigger owner | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Llama | — | — | — | — | — | — | opt-in cache; size limit pending | project maintainer | candidate pending |
| Qwen2 | — | — | — | — | — | — | opt-in cache; size limit pending | project maintainer | candidate pending |

Adding a row requires an immutable repository revision and hashes for every consumed config,
tokenizer, index, and weight file. Record whether authentication or license acceptance is required,
the exact model metadata and prompt/token output asserted, the maximum download/cache size, and who
owns manual/scheduled execution. A passing Tier-A synthetic fixture is not Tier-B evidence.
