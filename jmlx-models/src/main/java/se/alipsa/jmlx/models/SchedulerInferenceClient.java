package se.alipsa.jmlx.models;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.CompletionStage;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.tokenizer.ChatTemplateOptions;
import se.alipsa.jmlx.tokenizer.EncodingOptions;
import se.alipsa.jmlx.tokenizer.HfTokenizer;
import se.alipsa.jmlx.tokenizer.TokenizerEncoding;

/**
 * Owns one scheduler shared by framework views. Encoder weights are loaded lazily on its worker.
 * Close this client once all views are finished. Do not mix it with direct MLX calls or another
 * scheduler. BERT and decoder inference run sequentially; generation still batches compatible rows.
 */
public final class SchedulerInferenceClient implements InferenceClient {
  private final BatchGenerationScheduler scheduler;
  private final HfTokenizer tokenizer;
  private final String modelName;
  private final GenerationConfig defaults;
  private final Path encoderDirectory;
  private final Pooling pooling;
  private final Semaphore deliveries;
  // Accessed only by the scheduler worker.
  private TextEncoderModel encoder;
  private HfTokenizer encoderTokenizer;

  /**
   * Takes ownership of the scheduler. A null encoder directory disables embeddings; a null pooling
   * uses the sentence encoder artifact's defaults. Tokenizers must match their checkpoint.
   */
  public SchedulerInferenceClient(
      BatchGenerationScheduler scheduler,
      HfTokenizer tokenizer,
      String modelName,
      GenerationConfig defaults,
      Path encoderDirectory,
      Pooling pooling) {
    this.scheduler = Objects.requireNonNull(scheduler, "scheduler");
    this.deliveries = new Semaphore(scheduler.admissionCapacity());
    this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer");
    this.modelName = Objects.requireNonNull(modelName, "modelName");
    this.defaults = Objects.requireNonNull(defaults, "defaults");
    this.encoderDirectory = encoderDirectory;
    this.pooling = pooling;
  }

  @Override
  public String modelName() {
    return modelName;
  }

  @Override
  public GenerationConfig defaults() {
    return defaults;
  }

  @Override
  public InferenceCall chat(
      List<Map<String, Object>> messages, GenerationConfig config, Consumer<String> onDelta) {
    Objects.requireNonNull(onDelta, "onDelta");
    for (Map<String, Object> message : messages) {
      if (!(message.get("content") instanceof String)
          || !List.of("system", "user", "assistant").contains(message.get("role"))
          || !message.keySet().equals(java.util.Set.of("role", "content"))) {
        throw new IllegalArgumentException(
            "only system/user/assistant text messages are supported");
      }
    }
    GenerationRequest request =
        GenerationRequest.chat(
            tokenizer,
            messages,
            ChatTemplateOptions.defaults(true),
            config,
            CancellationToken.NONE);
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
    delivery.handle.set(handle);
    handle
        .stage()
        .whenComplete(
            (value, failure) -> {
              delivery.sourceValue = value;
              delivery.sourceFailure = failure;
              delivery.sourceDone = true;
            });
    Thread.ofVirtual().name("jmlx-text-delivery").start(delivery::drain);
    return new InferenceCall(delivery.result.minimalCompletionStage(), delivery::cancel);
  }

  @Override
  public CompletionStage<EmbeddingBatch> embed(List<String> texts) {
    List<String> inputs = List.copyOf(texts);
    if (encoderDirectory == null) {
      throw new IllegalStateException("no sentence encoder directory was configured");
    }
    return scheduler.execute(
        root -> {
          if (encoder == null) {
            MLXScope weights = root.newChild();
            try {
              HfTokenizer loadedTokenizer = HfTokenizer.fromDirectory(encoderDirectory);
              TextEncoderModel loaded = TextEncoderModels.load(weights, encoderDirectory);
              encoderTokenizer = loadedTokenizer;
              encoder = loaded;
            } catch (Throwable failure) {
              weights.close();
              throw failure;
            }
          }
          List<float[]> vectors = new ArrayList<>();
          int tokens = 0;
          for (String text : inputs) {
            TokenizerEncoding input =
                encoderTokenizer.encode(text, EncodingOptions.unbounded(true));
            EncoderResult result =
                pooling == null ? encoder.encode(input) : encoder.encode(input, pooling);
            vectors.add(result.embedding());
            tokens = Math.addExact(tokens, input.ids().size());
          }
          return new EmbeddingBatch(vectors, tokens);
        });
  }

  @Override
  public void close() {
    scheduler.close();
  }

  /** Unwraps a completion failure for framework error callbacks. */
  public static Throwable unwrap(Throwable failure) {
    while (failure instanceof CompletionException && failure.getCause() != null) {
      failure = failure.getCause();
    }
    return failure;
  }

  private static final class Delivery {
    private final LinkedBlockingQueue<String> deltas = new LinkedBlockingQueue<>(64);
    private final AtomicReference<BatchRequestHandle> handle = new AtomicReference<>();
    private final AtomicBoolean cancelled = new AtomicBoolean();
    private final AtomicReference<Throwable> overflow = new AtomicReference<>();
    private final CompletableFuture<GenerationResult> result = new CompletableFuture<>();
    private final Consumer<String> consumer;
    private final Runnable finished;
    private volatile boolean sourceDone;
    private GenerationResult sourceValue;
    private Throwable sourceFailure;

    Delivery(Consumer<String> consumer, Runnable finished) {
      this.consumer = consumer;
      this.finished = finished;
    }

    void offer(String delta) {
      if (delta != null && !delta.isEmpty() && !cancelled.get() && !deltas.offer(delta)) {
        overflow.compareAndSet(
            null, new IllegalStateException("stream consumer exceeded 64 queued deltas"));
        cancel();
      }
    }

    void cancel() {
      cancelled.set(true);
      BatchRequestHandle current = handle.get();
      if (current != null) {
        current.cancel();
      }
    }

    void drain() {
      try {
        while (!sourceDone || !deltas.isEmpty()) {
          String delta = deltas.poll(20, TimeUnit.MILLISECONDS);
          if (cancelled.get()) {
            BatchRequestHandle current = handle.get();
            if (current != null) {
              current.cancel();
            }
            deltas.clear();
          } else if (delta != null) {
            consumer.accept(delta);
          }
        }
        Throwable failure = overflow.get() == null ? sourceFailure : overflow.get();
        if (failure == null) {
          result.complete(sourceValue);
        } else {
          result.completeExceptionally(unwrap(failure));
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
