package se.alipsa.jmlx.langchain4j;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.model.output.TokenUsage;
import java.util.List;
import java.util.Objects;
import se.alipsa.jmlx.models.EmbeddingBatch;
import se.alipsa.jmlx.models.InferenceClient;

/** LangChain4j sentence embeddings over a borrowed client with a configured BERT encoder. */
public final class JmlxEmbeddingModel implements EmbeddingModel {
  private final InferenceClient client;

  /** Borrows a client; the application owns its lifecycle. */
  public JmlxEmbeddingModel(InferenceClient client) {
    this.client = Objects.requireNonNull(client, "client");
  }

  @Override
  public Response<List<Embedding>> embedAll(List<TextSegment> segments) {
    EmbeddingBatch batch =
        client.embedCall(segments.stream().map(TextSegment::text).toList()).await();
    return Response.from(
        batch.vectors().stream().map(Embedding::from).toList(),
        new TokenUsage(batch.inputTokens(), 0));
  }
}
