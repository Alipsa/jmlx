package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;

class InferenceCallTest {
  @Test
  void alreadyCompletedNullResultSignalsAContractFailure() throws Exception {
    InferenceCall call = new InferenceCall(CompletableFuture.completedFuture(null), () -> {});
    Throwable failure = call.failureSignal().toCompletableFuture().get(5, TimeUnit.SECONDS);
    assertInstanceOf(IllegalStateException.class, failure);
    assertEquals("inference client returned a null generation result", failure.getMessage());
    assertThrows(IllegalStateException.class, call::await);
  }

  @Test
  void laterNullResultCompletesTheFailureSignal() throws Exception {
    CompletableFuture<GenerationResult> source = new CompletableFuture<>();
    InferenceCall call = new InferenceCall(source, () -> {});
    assertFalse(call.failureSignal().toCompletableFuture().isDone());
    source.complete(null);
    assertInstanceOf(
        IllegalStateException.class,
        call.failureSignal().toCompletableFuture().get(5, TimeUnit.SECONDS));
    assertThrows(IllegalStateException.class, call::await);
  }
}
