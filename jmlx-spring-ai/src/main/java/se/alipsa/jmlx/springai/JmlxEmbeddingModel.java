package se.alipsa.jmlx.springai;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import org.springframework.ai.chat.metadata.DefaultUsage;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.Embedding;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.embedding.EmbeddingOptions;
import org.springframework.ai.embedding.EmbeddingRequest;
import org.springframework.ai.embedding.EmbeddingResponse;
import org.springframework.ai.embedding.EmbeddingResponseMetadata;
import se.alipsa.jmlx.models.EmbeddingBatch;
import se.alipsa.jmlx.models.InferenceClient;

/**
 * Spring AI sentence embeddings over a borrowed scheduler client with a configured BERT encoder.
 */
public final class JmlxEmbeddingModel implements EmbeddingModel {
  private final InferenceClient client;

  /** Borrows a client; model selection and dimensions belong to its configured encoder. */
  public JmlxEmbeddingModel(InferenceClient client) {
    this.client = Objects.requireNonNull(client, "client");
  }

  @Override
  public EmbeddingResponse call(EmbeddingRequest request) {
    EmbeddingOptions options = request.getOptions();
    if (options != null && (options.getModel() != null || options.getDimensions() != null)) {
      throw new IllegalArgumentException("embedding model and dimensions are fixed by the encoder");
    }
    EmbeddingBatch batch = client.embed(request.getInstructions()).toCompletableFuture().join();
    List<Embedding> results = new ArrayList<>();
    List<float[]> vectors = batch.vectors();
    for (int i = 0; i < vectors.size(); i++) {
      results.add(new Embedding(vectors.get(i), i));
    }
    return new EmbeddingResponse(
        results, new EmbeddingResponseMetadata("", new DefaultUsage(batch.inputTokens(), 0)));
  }

  @Override
  public float[] embed(Document document) {
    return embed(Objects.requireNonNull(document.getText(), "document text"));
  }
}
