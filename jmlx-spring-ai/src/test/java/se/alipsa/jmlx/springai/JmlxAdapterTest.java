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
  void cancellationReachesRequestAndFailureDoesNotYieldSuccess() throws Exception {
    StubInferenceClient client = new StubInferenceClient();
    client.pending = new CompletableFuture<>();
    var subscription = new JmlxChatModel(client).stream(new Prompt("hello")).subscribe();
    assertTrue(client.called.await(5, java.util.concurrent.TimeUnit.SECONDS));
    assertFalse(client.cancelled);
    subscription.dispose();
    assertTrue(client.cancelledSignal.await(5, java.util.concurrent.TimeUnit.SECONDS));
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
            ChatOptions.builder().temperature(1e-50).build(),
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
    assertEquals("tiny-bert", response.getMetadata().getModel());
    assertArrayEquals(new float[] {3, 1}, model.embed(new Document("abc")));
    assertThrows(
        IllegalArgumentException.class,
        () ->
            model.call(
                new EmbeddingRequest(
                    List.of("a"), EmbeddingOptions.builder().dimensions(1).build())));
  }

  private static void assertInterruptCancels(StubInferenceClient client, Runnable blocking)
      throws Exception {
    java.util.concurrent.atomic.AtomicReference<Throwable> failure =
        new java.util.concurrent.atomic.AtomicReference<>();
    java.util.concurrent.atomic.AtomicBoolean interrupted =
        new java.util.concurrent.atomic.AtomicBoolean();
    Thread caller =
        Thread.ofVirtual()
            .start(
                () -> {
                  try {
                    blocking.run();
                  } catch (Throwable error) {
                    failure.set(error);
                    interrupted.set(Thread.currentThread().isInterrupted());
                  }
                });
    try {
      assertTrue(client.called.await(5, java.util.concurrent.TimeUnit.SECONDS));
      caller.interrupt();
      caller.join(5000);
      assertFalse(caller.isAlive());
      assertTrue(failure.get() instanceof java.util.concurrent.CancellationException);
      assertTrue(interrupted.get());
      assertTrue(client.cancelled);
    } finally {
      caller.interrupt();
    }
  }

  @Test
  void interruptingBlockingChatCancelsAndPreservesInterrupt() throws Exception {
    StubInferenceClient client = new StubInferenceClient();
    client.pending = new CompletableFuture<>();
    assertInterruptCancels(client, () -> new JmlxChatModel(client).call(new Prompt("hello")));
  }

  @Test
  void interruptingBlockingEmbeddingsCancelsAndPreservesInterrupt() throws Exception {
    StubInferenceClient client = new StubInferenceClient();
    client.embeddingPending = new CompletableFuture<>();
    assertInterruptCancels(client, () -> new JmlxEmbeddingModel(client).embed("hello"));
  }

  @Test
  void blockingCallsExposeTheUnderlyingFailure() {
    StubInferenceClient client = new StubInferenceClient();
    client.pending =
        CompletableFuture.failedFuture(new IllegalArgumentException("invalid generation"));
    IllegalArgumentException chatFailure =
        assertThrows(
            IllegalArgumentException.class,
            () -> new JmlxChatModel(client).call(new Prompt("hello")));
    assertEquals("invalid generation", chatFailure.getMessage());
    client.embeddingPending =
        CompletableFuture.failedFuture(new IllegalArgumentException("invalid embedding"));
    IllegalArgumentException embedFailure =
        assertThrows(
            IllegalArgumentException.class, () -> new JmlxEmbeddingModel(client).embed("hello"));
    assertEquals("invalid embedding", embedFailure.getMessage());
  }

  @Test
  void schedulerCancellationIsAnErrorForBothBlockingViews() {
    StubInferenceClient client = new StubInferenceClient();
    client.pending =
        CompletableFuture.completedFuture(
            new se.alipsa.jmlx.models.GenerationResult(
                List.of(1),
                List.of(),
                se.alipsa.jmlx.models.FinishReason.CANCELLED,
                List.of(),
                ""));
    assertThrows(
        java.util.concurrent.CancellationException.class,
        () -> new JmlxChatModel(client).call(new Prompt("hello")));
  }

  @Test
  void boundedDemandWaitsForEachDeltaAndTheTerminalResponse() throws Exception {
    StubInferenceClient client = new StubInferenceClient();
    client.pending = new CompletableFuture<>();
    java.util.concurrent.LinkedBlockingQueue<org.springframework.ai.chat.model.ChatResponse>
        chunks = new java.util.concurrent.LinkedBlockingQueue<>();
    CompletableFuture<Void> completed = new CompletableFuture<>();
    reactor.core.publisher.BaseSubscriber<org.springframework.ai.chat.model.ChatResponse>
        subscriber =
            new reactor.core.publisher.BaseSubscriber<>() {
              @Override
              protected void hookOnSubscribe(org.reactivestreams.Subscription subscription) {}

              @Override
              protected void hookOnNext(org.springframework.ai.chat.model.ChatResponse value) {
                chunks.add(value);
              }

              @Override
              protected void hookOnComplete() {
                completed.complete(null);
              }

              @Override
              protected void hookOnError(Throwable error) {
                completed.completeExceptionally(error);
              }
            };
    new JmlxChatModel(client).stream(new Prompt("hello")).subscribe(subscriber);
    Thread producer = null;
    try {
      assertTrue(client.called.await(5, java.util.concurrent.TimeUnit.SECONDS));
      // The double exposes its listener before returning the pending call.
      producer =
          Thread.ofVirtual()
              .start(
                  () -> {
                    client.listener.accept("h");
                    client.listener.accept("é");
                    client.pending.complete(StubInferenceClient.result());
                  });
      assertEquals(null, chunks.poll(50, java.util.concurrent.TimeUnit.MILLISECONDS));
      assertFalse(completed.isDone());
      subscriber.request(1);
      assertEquals(
          "h",
          chunks.poll(5, java.util.concurrent.TimeUnit.SECONDS).getResult().getOutput().getText());
      assertEquals(null, chunks.poll(50, java.util.concurrent.TimeUnit.MILLISECONDS));
      subscriber.request(1);
      assertEquals(
          "é",
          chunks.poll(5, java.util.concurrent.TimeUnit.SECONDS).getResult().getOutput().getText());
      subscriber.request(1);
      assertEquals(
          "",
          chunks.poll(5, java.util.concurrent.TimeUnit.SECONDS).getResult().getOutput().getText());
      completed.get(5, java.util.concurrent.TimeUnit.SECONDS);
    } finally {
      subscriber.dispose();
      if (producer != null) {
        producer.join(5000);
      }
    }
  }

  @Test
  void missingAssistantTextIsAValidationError() {
    ChatModel model = new JmlxChatModel(new StubInferenceClient());
    assertThrows(
        IllegalArgumentException.class,
        () -> model.call(new Prompt(List.of(new AssistantMessage((String) null)))));
  }

  @Test
  void schedulerCancellationIsAStreamErrorWithoutSuccessTerminal() {
    StubInferenceClient client = new StubInferenceClient();
    client.pending =
        CompletableFuture.completedFuture(
            new se.alipsa.jmlx.models.GenerationResult(
                List.of(1),
                List.of(),
                se.alipsa.jmlx.models.FinishReason.CANCELLED,
                List.of(),
                ""));
    assertThrows(
        java.util.concurrent.CancellationException.class,
        () -> new JmlxChatModel(client).stream(new Prompt("hello")).collectList().block());
  }
}
