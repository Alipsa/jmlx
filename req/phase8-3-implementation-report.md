# Phase 8.3 implementation and validation

Date: 2026-11-14. Implementation complete; PR and hosted CI review pending.
Plan: [phase8-3-plan.md](plans/phase8-3-plan.md).

## Delivered behavior

Two new Java 25 adapter modules expose local text chat, incremental streaming
and sentence embeddings through Spring AI 2.0.2 and LangChain4j 1.23.0.
Both start at 0.1.0-SNAPSHOT. Framework callers, including virtual threads,
borrow one shared InferenceClient. The application closes its
SchedulerInferenceClient after all views finish.

The existing BatchGenerationScheduler remains the sole MLX owner. Bounded,
package-private worker operations load and run a separate BERT encoder
between generation cohorts. Encoder weights belong to a child of the worker
root and encoding scratch scopes close per request. Encoder failures fail
their stages; shutdown rejects/cancels queued work and closes all weights.

Shared prompt/policy mapping lives in jmlx-models for reuse by the future
server. Chat uses the decoder's configured chat template and tokenizer;
embeddings use the sentence encoder's own tokenizer and pooling policy.
Framework response mappings retain actual token counts and finish reasons.

Streaming text uses serial delivery off the native worker, preserving flush
deltas. Each consumer has at most 64 queued deltas; slow or throwing callbacks
cancel that request. A client-wide delivery limit equals the scheduler's
admission capacity and remains held until delivery finishes, even if native
generation completed already. Spring streams are cold and disposal cancels
the request. LangChain4j context cancellation silences further callbacks,
including terminal callbacks. A throwing terminal observer cannot produce a
second terminal event.

## Local verification

On the local Apple Silicon host with the staged runtime and Java 25:

- Root ./gradlew check passed, including ordinary/default-precision tests,
  float32GoldenTest, threadProbeTest, formatting, lint, provenance guards and
  publication checks.
- SchedulerOperationTest: 3 native tests, none skipped. Verifies same-worker
  ownership, shared admission, exceptional completion on close, recovery from
  an operation failure, and active-memory release after evaluated BERT use.
- SchedulerInferenceClientTest: 5 native tests, none skipped. Covers concurrent
  chat/embeddings, stable repeated embedding values and scratch memory,
  throwing callbacks, bounded slow-consumer overflow, aggregate delivery
  admission, and recovery after encoder-load failure.
- The two adapter mapping suites: 5 pure-Java tests each, none skipped. Cover
  ordered roles, supported options, explicit rejection, Unicode streaming,
  cancellation/error delivery, usage and ordered embedding results.
- tools/release-smoke/run.sh ci --profile all passed against nine locally
  published Maven modules under a fresh consumer Gradle home. POM-only and
  Gradle-metadata resolution agreed on 45 runtime components. License/NOTICE,
  JPMS names, native payload, nine sources and nine Javadoc jars passed.
  Public-interface native tests verified both frameworks' chat, incremental
  streams, ordered normalized embeddings, virtual threads, cancellation,
  and active-memory release on client close.
- The separate ./gradlew -p buildSrc check passed.
- tools/release-smoke/selftest.sh passed, including new adapter version
  preflight checks.

Ubuntu ARM64, in an isolated source copy: adapter checks and the smoke runner
selftest passed. Gradle provisioned Java 25; no native inference is claimed
for Linux. Existing macOS build outputs were not reused in that copy.
No Windows-specific behavior changed or was tested.

## Publication and compatibility limits

The 8.2 dated scope amendment accepts the evidence deliverable for advancing
development, leaving the affected measurement gates unpassed. It does not
rewrite evidence or assert settled ratios for unstable or blocked cells.

The adapters support text chat and the existing BERT sentence encoder.
Tool calls/results, media, response formats and stop strings are rejected;
nonzero temperature requires a seed in the configured defaults. Model
switching is rejected. Embedding dimensions/model selection belong to the
client's encoder configuration. Overlong encoder input is rejected, not
silently truncated. Quarkus extension wiring and Spring Boot auto-configuration
are not verified by this milestone.

Both adapters join the combined release candidate list and full artifact
smoke, and have matching per-module release scripts. CI runs their mapping
checks on Linux and requires both new native suites to execute on macOS.
The pure-Java smoke profile remains the three existing modules. The combined
release's 8.1 expected-selection guard remains intact; 8.4 must update it to
the reviewed native/adapters tuple and implement mixed Central/new-artifact
candidate resolution before publication. No Central deployment, version
bump of existing modules, native-pin change or generated-binding change
occurred.
