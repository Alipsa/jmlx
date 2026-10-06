package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.tokenizer.EncodingOptions;
import se.alipsa.jmlx.tokenizer.HfTokenizer;
import se.alipsa.jmlx.tokenizer.PairEncodingOptions;
import se.alipsa.jmlx.tokenizer.PairTruncationStrategy;
import se.alipsa.jmlx.tokenizer.TokenizerEncoding;
import tools.jackson.databind.JsonNode;

class TierBClassificationTest {
  @Test
  void comparesLogitsAndStableClassIds() throws Exception {
    TierBReferences reference = new TierBReferences("classification");
    HfTokenizer tokenizer = HfTokenizer.fromDirectory(reference.directory);
    try (MLXScope scope = new MLXScope()) {
      SequenceClassifier model = SequenceClassifiers.load(scope, reference.directory);
      reference.model(model.metadata());
      for (JsonNode input : reference.reference.required("cases")) {
        String text = input.required("text").asString();
        TokenizerEncoding encoding =
            input.has("text_pair")
                ? tokenizer.encode(
                    text,
                    input.required("text_pair").asString(),
                    new PairEncodingOptions(
                        EncodingOptions.unbounded(true), PairTruncationStrategy.LONGEST_FIRST))
                : tokenizer.encode(text, EncodingOptions.unbounded(true));
        assertEquals(TierBReferences.ids(input.required("source_ids")), encoding.ids());
        float[] logits = model.classify(encoding).logits();
        reference.compare(input.required("logits"), logits);
        reference.stable(input.required("gap").asDouble());
        int best = 0;
        for (int i = 1; i < logits.length; i++) {
          if (logits[i] > logits[best]) {
            best = i;
          }
        }
        assertEquals(input.required("class_id").intValue(), best);
      }
      reference.report();
    }
  }
}
