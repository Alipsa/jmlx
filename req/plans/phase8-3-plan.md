# Phase 8.3 — Spring AI and LangChain4j adapters

Date: 2026-11-14. Status: implemented and locally verified; hosted CI and PR review pending.

## Scope and references

Implement two independently published Java 25 modules, `jmlx-spring-ai` and
`jmlx-langchain4j`, each starting at 0.1.0-SNAPSHOT. No publication or native-pin
change is part of this milestone. Phase 8.4 owns publication and the mixed
Central/candidate graph. The baseline limitations remain in
[the 8.2 report](../phase8-2-benchmark.md#7-open-items-and-limitations).

Pin Spring AI `spring-ai-model` 2.0.2 and LangChain4j `langchain4j-core` 1.23.0.
These are the stable versions checked on 2026-11-14, not the Spring AI 2.1 RC
or LangChain4j development snapshot. Compile and test the pinned versions on
Java 25; no claim is made for other framework versions or Quarkus extensions.
References: [Spring release](https://spring.io/blog/2026/10/23/spring-ai-2-1-0-RC1-available-now/),
[LangChain4j release](https://github.com/langchain4j/langchain4j/releases/tag/1.23.0).
Exact contracts are verified against their Central source jars.

## Shared runtime and ownership

Use `BatchGenerationScheduler` as the sole MLX owner. Extend it with bounded
worker operations, executed between generation cohorts, for BERT encoder
loading and inference. Operations share admission limits, finish exceptionally
on shutdown/failure, and publish Java values on the completion dispatcher;
no MLXArray or scope escapes to framework threads. Operations must not block
on scheduler stages or recursively submit synchronous work. Request-local
encoder scratch state is released before returning. The worker root owns
encoder weights, including on failed loading and shutdown.

A framework-neutral client in jmlx-models shares prompt creation, generation
policy, token counts, cancellation and embedding input/output across both
adapters and the future Phase 9.3 server. Chat uses the checkpoint's tokenizer
and chat template, including add-generation-prompt. BERT embeddings use
TextEncoderModels, the encoder checkpoint tokenizer and explicit pooling;
return owned float arrays, preserve input order and reject implicit truncation.
The encoder can be loaded alongside a decoder on the same scheduler worker.
Adapters borrow the client. The application closes the shared client once,
which closes the scheduler and releases both models. Closing one view must
not invalidate other framework views.

## Framework contracts

Provide Spring AI ChatModel (including StreamingChatModel) and EmbeddingModel;
provide LangChain4j ChatModel, StreamingChatModel and EmbeddingModel. Map
system/user/assistant text messages in order. Reject tool calls, tool results,
non-text media and unsupported generation parameters explicitly rather than
silently flattening or ignoring them. Generation defaults are explicit,
immutable and greedy; nonzero temperature requires a configured seed. Framework
request overrides map only to supported jmlx generation controls. Stop strings
are rejected until there is a shared incremental stop-string implementation.

Synchronous calls wait from caller threads, including virtual threads. Streaming
is incremental; flush deltas are retained, followed by exactly one terminal
response with usage and finish reason. Failure never yields a success terminal.
Spring streams are cold and cancellation propagates to the request handle.
LangChain4j exposes its context cancellation handle. Callback exceptions cancel
the request. Slow callbacks must not block MLX; bounded delivery must either
complete or fail/cancel on overflow, never accumulate an unbounded token buffer.

## Work packages and exit gate

1. Record the dated 8.2 scope amendment without rewriting evidence or promoting
   rejected/unstable cells to passing measurements.
2. Add shared runtime, with meaningful lifecycle tests: concurrent chat/encoder
   use on one worker, bounded admission, close/failure completion and memory
   release. Preserve existing scheduler guarantees and precision references.
3. Add the two adapters, pinned dependencies, publication metadata, matching
   release scripts, documentation and mapping/cancellation/error tests.
4. Extend release smoke so adapter tests consume Maven artifacts, including
   transitive dependencies, through both frameworks' public interfaces. Use
   the committed tiny Mistral and BERT fixtures; assert chat/stream equality,
   ordered embedding vectors, virtual-thread access and memory release.
5. Run module checks, root checks, buildSrc checks if modified, and native
   artifact smoke. Add mandatory native suites to CI's execution assertion.
   Record results and limitations before marking the milestone delivered.

Publication remains Phase 8.4. No embedding reference is fabricated from
language-model token embeddings; BERT's existing tested sentence encoder is
the embedding implementation.

## Implementation record

See [the implementation report](../phase8-3-implementation-report.md) for the
local test results and published-artifact evidence. Worker operations are
package-private; no general native callback API was added to the scheduler's
public surface. Delivery remains admitted until the consumer finishes, with
an aggregate bound equal to the scheduler's total admission capacity.

## Review amendment — request control and bounded embedding work

Overlong embedding inputs are rejected before scheduler submission, using the
minimum of BERT positional capacity and sentence-transformers max_seq_length.
All texts in a batch are tokenized and checked on the caller. Batches contain
at most 128 texts and yield the worker between individual text forwards.
EmbeddingCall exposes cancellation; queued cancelled operations skip native
work and an executing forward finishes at its next safe boundary. Encoder
weights still load lazily on the sole worker; native load failure is sticky
rather than retried on later requests.

Sentence-only BERT inference pools natively, reading back [hidden] rather than
[sequence, hidden]; the full-state API and its reference assertions remain.
Blocking framework methods use interruptible waits and cancel on interruption.
Both frameworks treat scheduler cancellation as an error, and unwrap underlying
runtime failures. Blocking chat avoids the streaming delivery thread.

Spring streaming waits on demand on the delivery thread, retaining the bounded
64-delta queue and aggregate admission bound. Completion/disposal wakeups are
event-driven. Arbitrary callbacks cannot safely be terminated: a callback that
never returns retains its slot even after native cancellation. This documented
constraint also applies to a subscription that never requests or disposes.

Constructor argument/configuration failure closes the transferred scheduler.
The Spring automatic module name is se.alipsa.jmlx.springai before publication.
Embedding metadata reports the configured encoder directory's basename.
Sampled requests reuse the configured seed deliberately; a nonzero temperature
that underflows float32 is rejected.
