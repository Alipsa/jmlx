package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.tokenizer.EncodingOptions;
import se.alipsa.jmlx.tokenizer.HfTokenizer;
import tools.jackson.databind.JsonNode;

class TierBEmbeddingTest {
  @Test
  void matchesSentenceTransformersReference() throws Exception {
    TierBReferences reference = new TierBReferences("embedding");
    HfTokenizer tokenizer = HfTokenizer.fromDirectory(reference.directory);
    try (MLXScope scope = new MLXScope()) {
      TextEncoderModel model = TextEncoderModels.load(scope, reference.directory);
      reference.model(model.metadata());
      for (JsonNode input : reference.reference.required("cases")) {
        var encoding =
            tokenizer.encode(input.required("text").asString(), EncodingOptions.unbounded(true));
        assertEquals(TierBReferences.ids(input.required("source_ids")), encoding.ids());
        reference.compare(input.required("embedding"), model.encode(encoding).embedding());
      }
      reference.report();
    }
  }
}
