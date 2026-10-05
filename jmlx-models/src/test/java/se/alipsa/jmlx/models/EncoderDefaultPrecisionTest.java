package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;
import tools.jackson.databind.ObjectMapper;

/** Measures the ordinary inference mode independently of the strict float32 reference suite. */
@EnabledIfNativeAvailable
class EncoderDefaultPrecisionTest {
  @Test
  void defaultModePreservesEncoderAndSeq2SeqReferenceBounds() throws Exception {
    Path goldens = BertGoldenTest.root().resolve("tools/hf-reference/goldens");
    ObjectMapper json = new ObjectMapper();
    double maximum = 0;
    try (MLXScope scope = new MLXScope()) {
      var data = json.readTree(goldens.resolve("bert.json").toFile());
      var bert = TextEncoderModels.load(scope, goldens.resolve("checkpoints/bert"));
      float[][] actual = bert.encode(BertGoldenTest.input(data)).hiddenStates();
      for (int i = 0; i < actual.length; i++) {
        maximum =
            Math.max(
                maximum,
                error(BertGoldenTest.floats(data.path("hidden_states").get(i)), actual[i]));
      }
      for (String family : List.of("bert-sequence", "bert-token")) {
        var head = json.readTree(goldens.resolve(family + ".json").toFile());
        if (family.equals("bert-sequence")) {
          var classifier =
              SequenceClassifiers.load(scope, goldens.resolve("checkpoints/" + family));
          maximum =
              Math.max(
                  maximum,
                  error(
                      BertGoldenTest.floats(head.path("logits")),
                      classifier.classify(BertGoldenTest.input(head)).logits()));
        } else {
          var classifier = TokenClassifiers.load(scope, goldens.resolve("checkpoints/" + family));
          var logits = classifier.classify(BertGoldenTest.input(head)).logits();
          for (int i = 0; i < logits.length; i++) {
            maximum =
                Math.max(
                    maximum, error(BertGoldenTest.floats(head.path("logits").get(i)), logits[i]));
          }
        }
      }
      System.out.println("BERT default-mode maximum absolute error=" + maximum);
      for (String family : List.of("t5-relu", "t5-gated")) {
        data = json.readTree(goldens.resolve(family + ".json").toFile());
        T5Model model = T5Model.load(scope, goldens.resolve("checkpoints/" + family));
        int[] source =
            BertGoldenTest.integers(data.path("source_ids")).stream()
                .mapToInt(Integer::intValue)
                .toArray();
        int[] mask =
            BertGoldenTest.integers(data.path("source_mask")).stream()
                .mapToInt(Integer::intValue)
                .toArray();
        int[] history =
            BertGoldenTest.integers(data.path("history")).stream()
                .mapToInt(Integer::intValue)
                .toArray();
        try (MLXScope state = scope.newChild()) {
          double familyMaximum = 0;
          var encoded = model.encode(state, source, mask);
          MLX.eval(encoded);
          familyMaximum =
              Math.max(
                  familyMaximum,
                  error(BertGoldenTest.floats(data.path("encoder")), encoded.toFloatArray()));
          var cross = model.project(state, encoded);
          familyMaximum =
              Math.max(
                  familyMaximum,
                  error(
                      BertGoldenTest.floats(data.path("logits")),
                      model
                          .forward(state, history, source.length, mask, cross, null)
                          .toFloatArray()));
          var caches = new java.util.ArrayList<se.alipsa.jmlx.nn.KVCache>();
          for (int i = 0; i < model.metadata().numHiddenLayers(); i++) {
            caches.add(new se.alipsa.jmlx.nn.KVCache(state));
          }
          for (int i = 0; i < history.length; i++) {
            try (MLXScope step = state.newChild()) {
              var logits =
                  model.forward(step, new int[] {history[i]}, source.length, mask, cross, caches);
              familyMaximum =
                  Math.max(
                      familyMaximum,
                      error(
                          BertGoldenTest.floats(data.path("cached_logits").get(i)),
                          logits.toFloatArray()));
              for (var cache : caches) {
                MLX.eval(cache.keys(), cache.values());
              }
            }
          }
          maximum = Math.max(maximum, familyMaximum);
          assertTrue(familyMaximum <= 0.03, family + " default precision: " + familyMaximum);
          System.out.println(family + " default-mode maximum absolute error=" + familyMaximum);
        }
      }
    }
    // Independent default-mode bound: measured 0.020421 on M5 Max, rounded up with headroom.
    // Strict CPU float32 comparisons retain 1e-4 in the separate golden JVM.
    assertTrue(
        maximum <= 0.03, "default-mode error exceeds independently retained bound: " + maximum);
    System.out.println("Phase 7.2 default-mode maximum absolute error=" + maximum);
  }

  private static double error(float[] expected, float[] actual) {
    org.junit.jupiter.api.Assertions.assertEquals(expected.length, actual.length);
    double maximum = 0;
    for (int i = 0; i < actual.length; i++) {
      double error = Math.abs((double) expected[i] - actual[i]);
      assertTrue(Double.isFinite(error));
      maximum = Math.max(maximum, error);
    }
    return maximum;
  }
}
