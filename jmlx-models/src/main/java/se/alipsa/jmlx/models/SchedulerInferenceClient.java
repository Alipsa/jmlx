package se.alipsa.jmlx.models;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Path;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.tokenizer.ChatTemplateOptions;
import se.alipsa.jmlx.tokenizer.EncodingOptions;
import se.alipsa.jmlx.tokenizer.HfTokenizer;
import se.alipsa.jmlx.tokenizer.TokenizerEncoding;

/**
 * Owns one scheduler shared by framework views. Encoder weights load lazily once on its worker.
 * Close this client after all views finish. Do not mix it with direct MLX calls or another
 * scheduler. A batch contains at most 128 texts; all text lengths are checked on the caller before
 * submission. Each text runs as a separate worker operation so generation cohorts can run between
 * texts.
 */
public final class SchedulerInferenceClient implements InferenceClient {
  /** Most texts accepted in one embedding batch. Split larger ingestion batches at the caller. */
  public static final int MAX_EMBEDDING_TEXTS = 128;

  private final BatchGenerationScheduler scheduler;
  private final HfTokenizer tokenizer;
  private final String modelName;
  private final GenerationConfig defaults;
  private final Path encoderDirectory;
  private final Pooling pooling;
  private final Semaphore deliveries;
  private final HfTokenizer encoderTokenizer;
  private final TextEncoderModels.InputSpec encoderSpec;
  // The model is accessed only by the scheduler worker; the sticky load failure is also
  // checked by callers so another failed load never blocks generation.
  private TextEncoderModel encoder;
  private volatile Throwable encoderLoadFailure;

  /**
   * Takes ownership of the scheduler, including on argument/configuration failure. A null encoder
   * directory disables embeddings; null pooling uses artifact defaults. CPU tokenizer/config reads
   * happen here; native encoder loading remains on the worker. Failed native loads are sticky for
   * this client's lifetime. Create a new client after repairing its checkpoint.
   */
  public SchedulerInferenceClient(
      BatchGenerationScheduler scheduler,
      HfTokenizer tokenizer,
      String modelName,
      GenerationConfig defaults,
      Path encoderDirectory,
      Pooling pooling) {
    this.scheduler = Objects.requireNonNull(scheduler, "scheduler");
    try {
      this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer");
      this.modelName = Objects.requireNonNull(modelName, "modelName");
      this.defaults = Objects.requireNonNull(defaults, "defaults");
      if (modelName.isBlank()) {
        throw new IllegalArgumentException("modelName must not be blank");
      }
      if (defaults.maxNewTokens() > scheduler.maxNewTokensPerRequest()) {
        throw new IllegalArgumentException("default maxNewTokens exceeds the scheduler limit");
      }
      this.encoderDirectory =
          encoderDirectory == null ? null : encoderDirectory.toAbsolutePath().normalize();
      this.pooling = pooling;
      this.encoderTokenizer =
          encoderDirectory == null ? null : HfTokenizer.fromDirectory(encoderDirectory);
      this.encoderSpec =
          encoderDirectory == null ? null : TextEncoderModels.inputSpec(encoderDirectory);
      this.deliveries = new Semaphore(scheduler.admissionCapacity());
    } catch (IOException failure) {
      scheduler.close();
      throw new UncheckedIOException("cannot configure sentence encoder", failure);
    } catch (RuntimeException | Error failure) {
      scheduler.close();
      throw failure;
    }
  }

  @Override
  public String modelName() {
    return modelName;
  }

  @Override
  public String embeddingModelName() {
    return encoderDirectory == null
        ? "unconfigured"
        : encoderDirectory.getFileName() == null
            ? "sentence-encoder"
            : encoderDirectory.getFileName().toString();
  }

  @Override
  public GenerationConfig defaults() {
    return defaults;
  }

  private GenerationRequest request(List<Map<String, Object>> messages, GenerationConfig config) {
    if (messages == null) {
      throw new IllegalArgumentException("messages must not be null");
    }
    for (Map<String, Object> message : messages) {
      Object role = message == null ? null : message.get("role");
      if (message == null
          || !(message.get("content") instanceof String)
          || !("system".equals(role) || "user".equals(role) || "assistant".equals(role))
          || !message.keySet().equals(java.util.Set.of("role", "content"))) {
        throw new IllegalArgumentException(
            "only system/user/assistant text messages are supported");
      }
    }
    return GenerationRequest.chat(
        tokenizer, messages, ChatTemplateOptions.defaults(true), config, CancellationToken.NONE);
  }

  @Override
  public InferenceCall chat(List<Map<String, Object>> messages, GenerationConfig config) {
    BatchRequestHandle handle = scheduler.submit(request(messages, config), event -> {});
    return new InferenceCall(handle.stage().thenApply(InferenceAwait::completed), handle::cancel);
  }

  @Override
  public InferenceCall chat(
      List<Map<String, Object>> messages, GenerationConfig config, Consumer<String> onDelta) {
    Objects.requireNonNull(onDelta, "onDelta");
    GenerationRequest request = request(messages, config);
    if (!deliveries.tryAcquire()) {
      throw new BatchAdmissionRejectedException("text delivery capacity exhausted");
    }
    Delivery delivery = new Delivery(onDelta, deliveries::release);
    BatchRequestHandle handle;
    try {
      handle = scheduler.submit(request, event -> delivery.offer(event.textDelta()));
    } catch (Throwable failure) {
      deliveries.release();
      throw failure;
    }
    delivery.attach(handle);
    handle.stage().whenComplete(delivery::complete);
    Thread.ofVirtual().name("jmlx-text-delivery").start(delivery::drain);
    return new InferenceCall(
        delivery.result.minimalCompletionStage(),
        delivery::cancel,
        delivery.failureSignal.minimalCompletionStage());
  }

  @Override
  public EmbeddingCall embedCall(List<String> texts) {
    if (encoderDirectory == null) {
      throw new IllegalStateException("no sentence encoder directory was configured");
    }
    if (texts == null || texts.stream().anyMatch(Objects::isNull)) {
      throw new IllegalArgumentException("embedding texts and their elements must not be null");
    }
    List<String> inputs = List.copyOf(texts);
    if (inputs.size() > MAX_EMBEDDING_TEXTS) {
      throw new IllegalArgumentException(
          "embedding batches support at most " + MAX_EMBEDDING_TEXTS + " texts");
    }
    if (encoderLoadFailure != null) {
      throw new IllegalStateException(
          "sentence encoder load previously failed", encoderLoadFailure);
    }
    // Validate the entire batch before doing any native work, including special-token overhead.
    List<TokenizerEncoding> encoded = new ArrayList<>();
    int tokenCount = 0;
    for (String text : inputs) {
      TokenizerEncoding input = encoderTokenizer.encode(text, EncodingOptions.unbounded(true));
      if (input.ids().size() > encoderSpec.limit()) {
        throw new IllegalArgumentException(
            "embedding text exceeds encoder capacity of " + encoderSpec.limit() + " tokens");
      }
      encoderSpec.validate(input, pooling);
      encoded.add(input);
      tokenCount = Math.addExact(tokenCount, input.ids().size());
    }
    AtomicBoolean cancelled = new AtomicBoolean();
    CompletableFuture<EmbeddingBatch> result = new CompletableFuture<>();
    List<BatchGenerationScheduler.WorkerOperation<float[]>> steps = new ArrayList<>();
    for (TokenizerEncoding input : encoded) {
      steps.add(
          root -> {
            loadEncoder(root);
            return pooling == null ? encoder.embed(input) : encoder.embed(input, pooling);
          });
    }
    int tokens = tokenCount;
    var sequence = scheduler.submitSequence(steps, cancelled::get);
    sequence
        .stage()
        .whenComplete(
            (vectors, failure) -> {
              if (failure == null) {
                result.complete(new EmbeddingBatch(vectors, tokens));
              } else {
                result.completeExceptionally(unwrap(failure));
              }
            });
    return new EmbeddingCall(
        result.minimalCompletionStage(),
        () -> {
          cancelled.set(true);
          sequence.cancel();
          result.completeExceptionally(new CancellationException("embedding batch cancelled"));
        });
  }

  private void loadEncoder(MLXScope root) throws IOException {
    if (encoderLoadFailure != null) {
      throw new IllegalStateException(
          "sentence encoder load previously failed", encoderLoadFailure);
    }
    if (encoder == null) {
      MLXScope weights = root.newChild();
      try {
        encoder = TextEncoderModels.load(weights, encoderDirectory);
      } catch (IOException | RuntimeException | Error failure) {
        encoderLoadFailure = failure;
        try {
          weights.close();
        } catch (Throwable cleanup) {
          failure.addSuppressed(cleanup);
        }
        throw failure;
      }
    }
  }

  @Override
  public void close() {
    scheduler.close();
  }

  /** Unwraps a completion failure for framework error callbacks. */
  public static Throwable unwrap(Throwable failure) {
    while ((failure instanceof CompletionException || failure instanceof ExecutionException)
        && failure.getCause() != null) {
      failure = failure.getCause();
    }
    return failure;
  }

  /**
   * Event-driven bounded delivery. The monitor is never held during a user callback. A callback
   * that never returns still holds its delivery slot: callbacks must cooperate with application
   * timeouts. Java cannot safely terminate an arbitrary user callback.
   */
  private static final class Delivery {
    private final ArrayDeque<String> deltas = new ArrayDeque<>();
    private final AtomicReference<BatchRequestHandle> handle = new AtomicReference<>();
    private final CompletableFuture<GenerationResult> result = new CompletableFuture<>();
    private final CompletableFuture<Throwable> failureSignal = new CompletableFuture<>();
    private final Consumer<String> consumer;
    private final Runnable finished;
    private boolean cancelled;
    private boolean sourceDone;
    private GenerationResult sourceValue;
    private Throwable sourceFailure;

    Delivery(Consumer<String> consumer, Runnable finished) {
      this.consumer = consumer;
      this.finished = finished;
    }

    void attach(BatchRequestHandle submitted) {
      handle.set(submitted);
      synchronized (this) {
        if (cancelled) {
          submitted.cancel();
        }
      }
    }

    void offer(String delta) {
      boolean overflow = false;
      synchronized (this) {
        if (delta == null || delta.isEmpty() || cancelled) {
          return;
        }
        if (deltas.size() == 64) {
          sourceFailure = new IllegalStateException("stream consumer exceeded 64 queued deltas");
          overflow = true;
        } else {
          deltas.addLast(delta);
          notifyAll();
        }
      }
      if (overflow) {
        cancel();
      }
    }

    void complete(GenerationResult value, Throwable failure) {
      Throwable aborted = failure == null ? null : unwrap(failure);
      if (aborted == null && value != null && value.finishReason() == FinishReason.CANCELLED) {
        aborted = new CancellationException("generation cancelled");
      }
      synchronized (this) {
        sourceValue = value;
        if (sourceFailure == null) {
          sourceFailure = aborted;
        }
        if (aborted != null) {
          cancelled = true;
        }
        sourceDone = true;
        notifyAll();
      }
      if (aborted != null) {
        failureSignal.complete(aborted);
      }
    }

    void cancel() {
      Throwable failure;
      synchronized (this) {
        cancelled = true;
        if (sourceFailure == null) {
          sourceFailure = new CancellationException("generation cancelled");
        }
        failure = sourceFailure;
        notifyAll();
      }
      BatchRequestHandle current = handle.get();
      if (current != null) {
        current.cancel();
      }
      // Independent of the drain/result: a framework demand wait must wake even while its
      // consumer callback is blocked. Observers may only signal their own monitors here.
      failureSignal.complete(failure);
    }

    void drain() {
      try {
        while (true) {
          String delta;
          GenerationResult value;
          Throwable failure;
          synchronized (this) {
            while (deltas.isEmpty() && !sourceDone && !cancelled) {
              wait();
            }
            if (cancelled || (deltas.isEmpty() && sourceDone)) {
              value = sourceValue;
              failure = sourceFailure;
              delta = null;
            } else {
              delta = deltas.removeFirst();
              value = null;
              failure = null;
            }
          }
          if (delta == null) {
            if (failure == null) {
              result.complete(InferenceAwait.completed(value));
            } else {
              result.completeExceptionally(unwrap(failure));
            }
            return;
          }
          consumer.accept(delta);
        }
      } catch (Throwable failure) {
        cancel();
        result.completeExceptionally(failure);
      } finally {
        finished.run();
      }
    }
  }
}
