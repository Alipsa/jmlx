package se.alipsa.jmlx.springai;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.concurrent.CompletableFuture;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.embedding.EmbeddingOptions;
import org.springframework.ai.embedding.EmbeddingRequest;

class JmlxAdapterTest {
  @Test
  void mapsRolesOverridesUsageAndFinishReason() {
    StubInferenceClient client = new StubInferenceClient();
    ChatModel model = new JmlxChatModel(client);
    final var response =
        model.call(
            new Prompt(
                List.of(
                    new SystemMessage("rules"),
                    new UserMessage("question"),
                    new AssistantMessage("answer"),
                    new UserMessage("followup")),
                ChatOptions.builder().maxTokens(12).frequencyPenalty(0.5).build()));
    assertEquals(
        List.of("system", "user", "assistant", "user"),
        client.messages.stream().map(m -> m.get("role")).toList());
    assertEquals(12, client.policy.maxNewTokens());
    assertEquals(0.5f, client.policy.frequencyPenalty());
    assertEquals("hé", response.getResult().getOutput().getText());
    assertEquals(3, response.getMetadata().getUsage().getPromptTokens());
    assertEquals(2, response.getMetadata().getUsage().getCompletionTokens());
    assertEquals("MAX_TOKENS", response.getResult().getMetadata().getFinishReason());
  }

  @Test
  void streamingIsColdHasOneTerminalAndDoesNotDuplicateText() {
    StubInferenceClient client = new StubInferenceClient();
    var stream = new JmlxChatModel(client).stream(new Prompt("hello"));
    assertEquals(0, client.calls);
    var chunks = stream.collectList().block();
    assertEquals(1, client.calls);
    assertEquals(
        List.of("h", "é", ""),
        chunks.stream().map(c -> c.getResult().getOutput().getText()).toList());
    assertEquals("MAX_TOKENS", chunks.getLast().getResult().getMetadata().getFinishReason());
    stream.collectList().block();
    assertEquals(2, client.calls);
  }

  @Test
  void cancellationReachesRequestAndFailureDoesNotYieldSuccess() {
    StubInferenceClient client = new StubInferenceClient();
    client.pending = new CompletableFuture<>();
    var subscription = new JmlxChatModel(client).stream(new Prompt("hello")).subscribe();
    assertFalse(client.cancelled);
    subscription.dispose();
    assertTrue(client.cancelled);
    client.failure = new IllegalStateException("failed admission");
    assertThrows(
        IllegalStateException.class,
        () -> new JmlxChatModel(client).stream(new Prompt("hello")).collectList().block());
  }

  @Test
  void rejectsUnseededSamplingStopStringsAndModelSwitchingBeforeSubmission() {
    StubInferenceClient client = new StubInferenceClient();
    ChatModel model = new JmlxChatModel(client);
    for (ChatOptions options :
        List.of(
            ChatOptions.builder().temperature(0.8).build(),
            ChatOptions.builder().stopSequences(List.of("stop")).build(),
            ChatOptions.builder().model("other").build())) {
      assertThrows(IllegalArgumentException.class, () -> model.call(new Prompt("hello", options)));
    }
    assertEquals(0, client.calls);
  }

  @Test
  void embeddingsPreserveOrderAndDocumentTextAndRejectDimensionOverrides() {
    EmbeddingModel model = new JmlxEmbeddingModel(new StubInferenceClient());
    final var response = model.embedForResponse(List.of("a", "longer"));
    assertArrayEquals(new float[] {1, 1}, response.getResults().get(0).getOutput());
    assertArrayEquals(new float[] {6, 1}, response.getResults().get(1).getOutput());
    assertEquals(1, response.getResults().get(1).getIndex());
    assertEquals(8, response.getMetadata().getUsage().getPromptTokens());
    assertArrayEquals(new float[] {3, 1}, model.embed(new Document("abc")));
    assertThrows(
        IllegalArgumentException.class,
        () ->
            model.call(
                new EmbeddingRequest(
                    List.of("a"), EmbeddingOptions.builder().dimensions(1).build())));
  }
}
