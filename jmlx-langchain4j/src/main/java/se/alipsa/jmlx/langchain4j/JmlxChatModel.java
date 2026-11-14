package se.alipsa.jmlx.langchain4j;

import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import java.util.Objects;
import se.alipsa.jmlx.models.InferenceClient;

/** LangChain4j synchronous text chat over a borrowed shared inference client. */
public final class JmlxChatModel implements ChatModel {
  private final InferenceClient client;

  /** Borrows a client; the application owns its lifecycle. */
  public JmlxChatModel(InferenceClient client) {
    this.client = Objects.requireNonNull(client, "client");
  }

  @Override
  public ChatResponse doChat(ChatRequest request) {
    return JmlxChatMapping.response(
        client,
        client
            .chat(JmlxChatMapping.messages(request), JmlxChatMapping.config(client, request))
            .await());
  }
}
