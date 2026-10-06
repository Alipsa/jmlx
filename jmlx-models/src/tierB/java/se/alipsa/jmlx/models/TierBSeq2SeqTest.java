package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.Arrays;
import java.util.Set;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.tokenizer.HfTokenizer;
import tools.jackson.databind.JsonNode;

class TierBSeq2SeqTest {
  @Test
  void comparesFullGreedySequenceAndTeacherForcedLogits() throws Exception {
    TierBReferences reference = new TierBReferences("seq2seq");
    HfTokenizer tokenizer = HfTokenizer.fromDirectory(reference.directory);
    try (MLXScope scope = new MLXScope()) {
      T5Model model = (T5Model) TextGenerationModels.load(scope, reference.directory);
      reference.model(model.metadata());
      for (JsonNode input : reference.reference.required("cases")) {
        GenerationRequest request =
            GenerationRequest.text(
                tokenizer,
                input.required("text").asString(),
                PromptSpecialTokens.ADD,
                GenerationConfig.greedyDefaults(
                    input.required("max_new_tokens").intValue(), Set.of(1)),
                CancellationToken.NONE);
        assertEquals(
            TierBReferences.ids(input.required("source_ids")),
            Arrays.stream(request.promptTokenIds()).boxed().toList());
        int[] source = request.promptTokenIds();
        int[] sourceMask = new int[source.length];
        Arrays.fill(sourceMask, 1);
        try (MLXScope state = scope.newChild()) {
          MLXArray encoded;
          try (MLXScope stage = state.newChild()) {
            encoded = MLX.hoist(model.encode(stage, source, sourceMask), state);
            MLX.eval(encoded);
          }
          var cross = model.project(state, encoded);
          int index = 0;
          for (JsonNode history : input.required("histories")) {
            try (MLXScope step = state.newChild()) {
              int[] ids =
                  TierBReferences.ids(history).stream().mapToInt(Integer::intValue).toArray();
              float[] logits =
                  model.forward(step, ids, source.length, sourceMask, cross, null).toFloatArray();
              int width = model.metadata().vocabSize();
              reference.compare(
                  input.required("logits").get(index),
                  Arrays.copyOfRange(logits, logits.length - width, logits.length));
              reference.stable(input.required("gaps").get(index).asDouble());
              index++;
            }
          }
        }
        GenerationResult result = model.generate(request, ignored -> {});
        assertEquals(TierBReferences.ids(input.required("target_ids")), result.generatedTokenIds());
        assertEquals(
            FinishReason.valueOf(input.required("finish_reason").asString()),
            result.finishReason());
      }
      reference.report();
    }
  }
}
