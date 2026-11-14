package se.alipsa.jmlx.langchain4j;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.request.ChatRequestParameters;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.output.FinishReason;
import dev.langchain4j.model.output.TokenUsage;
import java.util.List;
import java.util.Map;
import se.alipsa.jmlx.models.GenerationConfig;
import se.alipsa.jmlx.models.GenerationResult;
import se.alipsa.jmlx.models.InferenceClient;
import se.alipsa.jmlx.models.InferenceOptions;

/** Common text-only mappings for the synchronous and streaming LangChain4j views. */
final class JmlxChatMapping {
  private JmlxChatMapping() {}

  static List<Map<String, Object>> messages(ChatRequest request) {
    return request.messages().stream()
        .map(
            message -> {
              if (message instanceof SystemMessage system) {
                return Map.<String, Object>of("role", "system", "content", system.text());
              }
              if (message instanceof UserMessage user
                  && user.hasSingleText()
                  && user.name() == null) {
                return Map.<String, Object>of("role", "user", "content", user.singleText());
              }
              if (message instanceof AiMessage assistant
                  && !assistant.hasToolExecutionRequests()
                  && assistant.thinking() == null) {
                return Map.<String, Object>of("role", "assistant", "content", assistant.text());
              }
              throw new IllegalArgumentException(
                  "only unnamed system/user/assistant text messages are supported");
            })
        .toList();
  }

  static GenerationConfig config(InferenceClient client, ChatRequest request) {
    ChatRequestParameters options = request.parameters();
    if (options == null) {
      return client.defaults();
    }
    if (options.toolSpecifications() != null && !options.toolSpecifications().isEmpty()
        || options.toolChoice() != null
        || options.responseFormat() != null) {
      throw new IllegalArgumentException("tools and response formats are not supported");
    }
    return InferenceOptions.apply(
        client,
        options.modelName(),
        options.maxOutputTokens(),
        options.temperature(),
        options.topK(),
        options.topP(),
        options.frequencyPenalty(),
        options.presencePenalty(),
        options.stopSequences());
  }

  static ChatResponse response(InferenceClient client, GenerationResult result) {
    FinishReason reason =
        switch (result.finishReason()) {
          case MAX_TOKENS -> FinishReason.LENGTH;
          case EOS, STOP_TOKEN -> FinishReason.STOP;
          case CANCELLED ->
              throw new java.util.concurrent.CancellationException("generation cancelled");
        };
    return ChatResponse.builder()
        .aiMessage(AiMessage.from(result.generatedText()))
        .modelName(client.modelName())
        .finishReason(reason)
        .tokenUsage(new TokenUsage(result.promptPositions(), result.generatedTokenIds().size()))
        .build();
  }
}
