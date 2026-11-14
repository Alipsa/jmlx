package se.alipsa.jmlx.springai;

import java.util.List;
import java.util.Map;
import java.util.Objects;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.metadata.ChatGenerationMetadata;
import org.springframework.ai.chat.metadata.ChatResponseMetadata;
import org.springframework.ai.chat.metadata.DefaultUsage;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.Generation;
import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.ai.chat.prompt.Prompt;
import reactor.core.publisher.Flux;
import reactor.core.publisher.FluxSink;
import se.alipsa.jmlx.models.GenerationConfig;
import se.alipsa.jmlx.models.GenerationResult;
import se.alipsa.jmlx.models.InferenceCall;
import se.alipsa.jmlx.models.InferenceClient;
import se.alipsa.jmlx.models.InferenceOptions;
import se.alipsa.jmlx.models.SchedulerInferenceClient;

/** Spring AI text chat and cold streaming views over a borrowed shared inference client. */
public final class JmlxChatModel implements ChatModel {
  private final InferenceClient client;

  /** Borrows a client; the application closes it after all adapter views are finished. */
  public JmlxChatModel(InferenceClient client) {
    this.client = Objects.requireNonNull(client, "client");
  }

  @Override
  public ChatResponse call(Prompt prompt) {
    GenerationResult result = client.chat(messages(prompt), config(prompt)).await();
    return response(result.generatedText(), result);
  }

  @Override
  public Flux<ChatResponse> stream(Prompt prompt) {
    return Flux.create(
        sink -> {
          Demand demand = new Demand(sink);
          java.util.concurrent.atomic.AtomicReference<InferenceCall> submitted =
              new java.util.concurrent.atomic.AtomicReference<>();
          sink.onRequest(count -> demand.wake());
          sink.onDispose(
              () -> {
                demand.wake();
                InferenceCall call = submitted.get();
                if (call != null) {
                  call.cancel();
                }
              });
          // Subscription setup also runs off the subscribing thread: even a synchronous custom
          // client may deliver its first delta before chat() returns.
          Thread.ofVirtual()
              .name("jmlx-spring-subscription")
              .start(
                  () -> {
                    if (sink.isCancelled()) {
                      return;
                    }
                    try {
                      InferenceCall call =
                          client.chat(
                              messages(prompt),
                              config(prompt),
                              delta -> {
                                if (demand.await()) {
                                  sink.next(response(delta, null));
                                }
                              });
                      submitted.set(call);
                      if (sink.isCancelled()) {
                        call.cancel();
                      }
                      call.stage()
                          .whenComplete(
                              (result, failure) -> {
                                if (sink.isCancelled()) {
                                  return;
                                }
                                try {
                                  if (failure != null) {
                                    sink.error(SchedulerInferenceClient.unwrap(failure));
                                  } else {
                                    // Build/check before waiting for terminal demand so
                                    // cancellation is an error
                                    // immediately, rather than a success waiting indefinitely for
                                    // another request.
                                    ChatResponse terminal = response("", result);
                                    if (demand.await()) {
                                      sink.next(terminal);
                                      sink.complete();
                                    }
                                  }
                                } catch (RuntimeException error) {
                                  sink.error(error);
                                }
                              });
                    } catch (RuntimeException failure) {
                      sink.error(failure);
                    }
                  });
        },
        FluxSink.OverflowStrategy.ERROR);
  }

  private static final class Demand {
    private final FluxSink<ChatResponse> sink;

    Demand(FluxSink<ChatResponse> sink) {
      this.sink = sink;
    }

    synchronized void wake() {
      notifyAll();
    }

    synchronized boolean await() {
      while (sink.requestedFromDownstream() == 0 && !sink.isCancelled()) {
        try {
          wait();
        } catch (InterruptedException interrupted) {
          Thread.currentThread().interrupt();
          throw new java.util.concurrent.CancellationException("stream delivery interrupted");
        }
      }
      return !sink.isCancelled();
    }
  }

  private GenerationConfig config(Prompt prompt) {
    ChatOptions options = prompt.getOptions();
    if (options == null) {
      return client.defaults();
    }
    if (options instanceof org.springframework.ai.model.tool.ToolCallingChatOptions tools
        && (tools.getToolCallbacks() != null && !tools.getToolCallbacks().isEmpty()
            || tools.getToolContext() != null && !tools.getToolContext().isEmpty())) {
      throw new IllegalArgumentException("tool calling is not supported");
    }
    return InferenceOptions.apply(
        client,
        options.getModel(),
        options.getMaxTokens(),
        options.getTemperature(),
        options.getTopK(),
        options.getTopP(),
        options.getFrequencyPenalty(),
        options.getPresencePenalty(),
        options.getStopSequences());
  }

  private static List<Map<String, Object>> messages(Prompt prompt) {
    return prompt.getInstructions().stream()
        .map(
            message -> {
              String role = message.getMessageType().getValue();
              if (role == null
                  || message.getText() == null
                  || !List.of("system", "user", "assistant").contains(role)
                  || message instanceof UserMessage user && !user.getMedia().isEmpty()
                  || message instanceof AssistantMessage assistant
                      && (!assistant.getToolCalls().isEmpty() || !assistant.getMedia().isEmpty())) {
                throw new IllegalArgumentException(
                    "only system/user/assistant text messages are supported");
              }
              return Map.<String, Object>of("role", role, "content", message.getText());
            })
        .toList();
  }

  private ChatResponse response(String text, GenerationResult result) {
    if (result != null && result.finishReason() == se.alipsa.jmlx.models.FinishReason.CANCELLED) {
      throw new java.util.concurrent.CancellationException("generation cancelled");
    }
    ChatGenerationMetadata generation =
        result == null
            ? ChatGenerationMetadata.NULL
            : ChatGenerationMetadata.builder().finishReason(result.finishReason().name()).build();
    ChatResponseMetadata metadata =
        result == null
            ? ChatResponseMetadata.builder().build()
            : ChatResponseMetadata.builder()
                .model(client.modelName())
                .usage(
                    new DefaultUsage(result.promptPositions(), result.generatedTokenIds().size()))
                .build();
    return new ChatResponse(
        List.of(new Generation(new AssistantMessage(text), generation)), metadata);
  }
}
