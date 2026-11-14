package se.alipsa.jmlx.langchain4j;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.request.ChatRequestParameters;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.chat.response.PartialResponse;
import dev.langchain4j.model.chat.response.PartialResponseContext;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;
import dev.langchain4j.model.output.FinishReason;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;

class JmlxAdapterTest {
  @Test
  void mapsRolesOverridesUsageAndFinishReasonThroughInterface() {
    StubInferenceClient client = new StubInferenceClient();
    ChatModel model = new JmlxChatModel(client);
    final ChatResponse response =
        model.chat(
            ChatRequest.builder()
                .messages(
                    SystemMessage.from("rules"),
                    UserMessage.from("question"),
                    AiMessage.from("answer"),
                    UserMessage.from("followup"))
                .parameters(
                    ChatRequestParameters.builder()
                        .maxOutputTokens(12)
                        .frequencyPenalty(0.5)
                        .build())
                .build());
    assertEquals(
        List.of("system", "user", "assistant", "user"),
        client.messages.stream().map(m -> m.get("role")).toList());
    assertEquals(12, client.policy.maxNewTokens());
    assertEquals(0.5f, client.policy.frequencyPenalty());
    assertEquals("hé", response.aiMessage().text());
    assertEquals(3, response.tokenUsage().inputTokenCount());
    assertEquals(2, response.tokenUsage().outputTokenCount());
    assertEquals(FinishReason.LENGTH, response.finishReason());
  }

  @Test
  void contextCancellationWorksBeforeSubmissionReturnsAndSilencesCompletion() {
    StubInferenceClient client = new StubInferenceClient();
    AtomicInteger partials = new AtomicInteger();
    AtomicInteger terminals = new AtomicInteger();
    new JmlxStreamingChatModel(client)
        .chat(
            "hello",
            new StreamingChatResponseHandler() {
              @Override
              public void onPartialResponse(
                  PartialResponse response, PartialResponseContext context) {
                partials.incrementAndGet();
                assertFalse(context.streamingHandle().isCancelled());
                context.streamingHandle().cancel();
                assertTrue(context.streamingHandle().isCancelled());
              }

              @Override
              public void onCompleteResponse(ChatResponse response) {
                terminals.incrementAndGet();
              }

              @Override
              public void onError(Throwable error) {
                terminals.incrementAndGet();
              }
            });
    assertEquals(1, partials.get());
    assertEquals(0, terminals.get());
    assertTrue(client.cancelled);
  }

  @Test
  void streamsUnicodeAndReportsAsynchronousFailureOnce() {
    StubInferenceClient client = new StubInferenceClient();
    client.pending = new CompletableFuture<>();
    StringBuilder deltas = new StringBuilder();
    AtomicInteger success = new AtomicInteger();
    AtomicInteger errors = new AtomicInteger();
    new JmlxStreamingChatModel(client)
        .chat(
            "hello",
            new StreamingChatResponseHandler() {
              @Override
              public void onPartialResponse(String text) {
                deltas.append(text);
              }

              @Override
              public void onCompleteResponse(ChatResponse response) {
                success.incrementAndGet();
              }

              @Override
              public void onError(Throwable error) {
                errors.incrementAndGet();
                assertEquals("broken", error.getMessage());
              }
            });
    client.listener.accept("hé");
    client.pending.completeExceptionally(new IllegalStateException("broken"));
    assertEquals("hé", deltas.toString());
    assertEquals(1, errors.get());
    assertEquals(0, success.get());
  }

  @Test
  void rejectsUnsupportedOptionsBeforeSubmission() {
    StubInferenceClient client = new StubInferenceClient();
    ChatModel model = new JmlxChatModel(client);
    for (ChatRequestParameters options :
        List.of(
            ChatRequestParameters.builder().temperature(0.5).build(),
            ChatRequestParameters.builder().stopSequences("stop").build(),
            ChatRequestParameters.builder().modelName("other").build())) {
      assertThrows(
          IllegalArgumentException.class,
          () ->
              model.chat(
                  ChatRequest.builder()
                      .messages(UserMessage.from("hello"))
                      .parameters(options)
                      .build()));
    }
    assertEquals(0, client.calls);
  }

  @Test
  void embeddingsRetainInputOrderAndUsageThroughInterface() {
    dev.langchain4j.model.embedding.EmbeddingModel model =
        new JmlxEmbeddingModel(new StubInferenceClient());
    var response = model.embedAll(List.of(TextSegment.from("a"), TextSegment.from("longer")));
    assertArrayEquals(new float[] {1, 1}, response.content().getFirst().vector());
    assertArrayEquals(new float[] {6, 1}, response.content().getLast().vector());
    assertEquals(8, response.tokenUsage().inputTokenCount());
  }
}
