package se.alipsa.jmlx.langchain4j;

import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.PartialResponse;
import dev.langchain4j.model.chat.response.PartialResponseContext;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;
import dev.langchain4j.model.chat.response.StreamingHandle;
import java.util.Objects;
import se.alipsa.jmlx.models.InferenceCall;
import se.alipsa.jmlx.models.InferenceClient;
import se.alipsa.jmlx.models.SchedulerInferenceClient;

/** LangChain4j incremental text with context cancellation, delivered off the MLX worker. */
public final class JmlxStreamingChatModel implements StreamingChatModel {
  private final InferenceClient client;

  /** Borrows a client; the application owns its lifecycle. */
  public JmlxStreamingChatModel(InferenceClient client) {
    this.client = Objects.requireNonNull(client, "client");
  }

  @Override
  public void doChat(ChatRequest request, StreamingChatResponseHandler handler) {
    Objects.requireNonNull(handler, "handler");
    Handle handle = new Handle();
    try {
      InferenceCall call =
          client.chat(
              JmlxChatMapping.messages(request),
              JmlxChatMapping.config(client, request),
              delta -> {
                synchronized (handle) {
                  if (!handle.cancelled) {
                    handler.onPartialResponse(
                        new PartialResponse(delta), new PartialResponseContext(handle));
                  }
                }
              });
      handle.attach(call);
      call.stage()
          .whenComplete(
              (result, failure) -> {
                synchronized (handle) {
                  if (handle.cancelled) {
                    return;
                  }
                  if (failure != null) {
                    handler.onError(SchedulerInferenceClient.unwrap(failure));
                  } else {
                    dev.langchain4j.model.chat.response.ChatResponse response;
                    try {
                      response = JmlxChatMapping.response(client, result);
                    } catch (RuntimeException error) {
                      handler.onError(error);
                      return;
                    }
                    handler.onCompleteResponse(response);
                  }
                }
              });
    } catch (RuntimeException failure) {
      handler.onError(failure);
    }
  }

  private static final class Handle implements StreamingHandle {
    private boolean cancelled;
    private InferenceCall call;

    synchronized void attach(InferenceCall submitted) {
      call = submitted;
      if (cancelled) {
        call.cancel();
      }
    }

    @Override
    public synchronized void cancel() {
      cancelled = true;
      if (call != null) {
        call.cancel();
      }
    }

    @Override
    public synchronized boolean isCancelled() {
      return cancelled;
    }
  }
}
