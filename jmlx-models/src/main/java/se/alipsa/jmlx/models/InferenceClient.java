package se.alipsa.jmlx.models;

import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletionStage;
import java.util.function.Consumer;

/** Framework-neutral text inference; all returned values are detached from native memory. */
public interface InferenceClient extends AutoCloseable {
  /** Configured decoder name, used for response metadata and model-option validation. */
  String modelName();

  /** Configured sentence encoder identity for response metadata. */
  String embeddingModelName();

  /** Explicit generation policy; sampling always carries a seed. */
  GenerationConfig defaults();

  /**
   * Submits ordered role/content text messages. Deltas arrive serially off the MLX worker; the
   * stage completes after the last delta. Slow consumers fail with bounded-buffer overflow.
   */
  InferenceCall chat(
      List<Map<String, Object>> messages, GenerationConfig config, Consumer<String> onDelta);

  /** Submits chat without streaming callbacks; blocking views can avoid a text delivery thread. */
  default InferenceCall chat(List<Map<String, Object>> messages, GenerationConfig config) {
    return chat(messages, config, delta -> {});
  }

  /** Embeds validated texts in order, with explicit cancellation of queued and remaining work. */
  EmbeddingCall embedCall(List<String> texts);

  /**
   * Convenience stage; use {@link #embedCall} when cancellation or interruptible waiting is needed.
   */
  default CompletionStage<EmbeddingBatch> embed(List<String> texts) {
    return embedCall(texts).stage();
  }

  /** Closes the owned runtime; framework adapter views borrow this client and never close it. */
  @Override
  void close();
}
