package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXIO;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.tokenizer.TokenOffset;
import se.alipsa.jmlx.tokenizer.TokenizerEncoding;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;

@EnabledIfNativeAvailable
@Tag("full-float32")
class BertGoldenTest {
  private static final ObjectMapper JSON = new ObjectMapper();

  @TempDir Path temporaryDirectory;

  static Path root() {
    return Path.of(System.getProperty("jmlx.repository.root"));
  }

  static TokenizerEncoding input(JsonNode data) {
    List<Integer> ids = integers(data.path("ids"));
    return new TokenizerEncoding(
        ids,
        integers(data.path("type_ids")),
        integers(data.path("attention_mask")),
        java.util.Collections.nCopies(ids.size(), 0),
        java.util.Collections.nCopies(ids.size(), TokenOffset.NONE),
        java.util.Collections.nCopies(ids.size(), "fixture"));
  }

  static List<Integer> integers(JsonNode node) {
    List<Integer> result = new ArrayList<>();
    node.forEach(value -> result.add(value.intValue()));
    return result;
  }

  static float[] floats(JsonNode node) {
    List<Float> output = new ArrayList<>();
    flatten(node, output);
    float[] result = new float[output.size()];
    for (int i = 0; i < result.length; i++) {
      result[i] = output.get(i);
    }
    return result;
  }

  private static void flatten(JsonNode node, List<Float> output) {
    if (node.isArray()) {
      node.forEach(child -> flatten(child, output));
    } else {
      output.add((float) node.asDouble());
    }
  }

  @Test
  void embeddingsAndPaddedHiddenRowsMatchHf() throws Exception {
    Path goldens = root().resolve("tools/hf-reference/goldens");
    JsonNode data = JSON.readTree(goldens.resolve("bert.json").toFile());
    try (MLXScope scope = new MLXScope()) {
      TextEncoderModel model = TextEncoderModels.load(scope, goldens.resolve("checkpoints/bert"));
      TokenizerEncoding input = input(data);
      for (Pooling.Mode mode : Pooling.Mode.values()) {
        EncoderResult result = model.encode(input, new Pooling(mode, false));
        assertArrayEquals(
            floats(data.path(mode.name().toLowerCase(java.util.Locale.ROOT))),
            result.embedding(),
            1e-4f);
        float[][] hidden = result.hiddenStates();
        for (int i = 0; i < hidden.length; i++) {
          assertArrayEquals(floats(data.path("hidden_states").get(i)), hidden[i], 1e-4f);
        }
      }
      assertArrayEquals(
          floats(data.path("normalized_mean")),
          model.encode(input, new Pooling(Pooling.Mode.MEAN, true)).embedding(),
          1e-4f);
      assertEquals(2, model.metadata().numEncoderLayers());
      List<Integer> ids = new ArrayList<>(input.ids());
      ids.set(0, 32);
      TokenizerEncoding invalid =
          new TokenizerEncoding(
              ids,
              input.typeIds(),
              input.attentionMask(),
              input.specialTokensMask(),
              input.offsets(),
              input.tokens());
      TokenizerEncoding invalidIds = invalid;
      assertThrows(IllegalArgumentException.class, () -> model.encode(invalidIds));
      List<Integer> types = new ArrayList<>(input.typeIds());
      types.set(0, 2);
      invalid =
          new TokenizerEncoding(
              input.ids(),
              types,
              input.attentionMask(),
              input.specialTokensMask(),
              input.offsets(),
              input.tokens());
      TokenizerEncoding invalidTypes = invalid;
      assertThrows(IllegalArgumentException.class, () -> model.encode(invalidTypes));
    }
  }

  @Test
  void bothTaskHeadsMatchHf() throws Exception {
    Path goldens = root().resolve("tools/hf-reference/goldens");
    JsonNode sequence = JSON.readTree(goldens.resolve("bert-sequence.json").toFile());
    JsonNode token = JSON.readTree(goldens.resolve("bert-token.json").toFile());
    try (MLXScope scope = new MLXScope()) {
      SequenceClassifier sequenceModel =
          SequenceClassifiers.load(scope, goldens.resolve("checkpoints/bert-sequence"));
      assertArrayEquals(
          floats(sequence.path("logits")), sequenceModel.classify(input(sequence)).logits(), 1e-4f);
      TokenClassifier tokenModel =
          TokenClassifiers.load(scope, goldens.resolve("checkpoints/bert-token"));
      float[][] logits = tokenModel.classify(input(token)).logits();
      for (int i = 0; i < logits.length; i++) {
        assertArrayEquals(floats(token.path("logits").get(i)), logits[i], 1e-4f);
      }
      assertThrows(
          IllegalArgumentException.class,
          () -> TextEncoderModels.load(scope, goldens.resolve("checkpoints/bert-sequence")));
    }
  }

  @Test
  void tokenClassificationIgnoresPoolerWeightsFromBaseCheckpoint() throws Exception {
    Path goldens = root().resolve("tools/hf-reference/goldens");
    Path base = goldens.resolve("checkpoints/bert-token");
    JsonNode token = JSON.readTree(goldens.resolve("bert-token.json").toFile());
    int hidden = JSON.readTree(base.resolve("config.json").toFile()).get("hidden_size").intValue();
    Path withPooler = Files.createDirectory(temporaryDirectory.resolve("with-pooler"));
    try (MLXScope scope = new MLXScope()) {
      MLXIO.SafetensorsResult tensors =
          MLXIO.loadSafetensors(scope, base.resolve("model.safetensors").toString());
      Map<String, MLXArray> withPoolerTensors = new LinkedHashMap<>(tensors.tensors());
      withPoolerTensors.put(
          "bert.pooler.dense.weight",
          MLX.array(scope, new float[hidden * hidden], new int[] {hidden, hidden}));
      withPoolerTensors.put(
          "bert.pooler.dense.bias", MLX.array(scope, new float[hidden], new int[] {hidden}));
      Files.copy(base.resolve("config.json"), withPooler.resolve("config.json"));
      MLXIO.saveSafetensors(
          withPooler.resolve("model.safetensors").toString(), withPoolerTensors, Map.of());
      TokenClassifier baseModel = TokenClassifiers.load(scope, base);
      TokenClassifier poolerModel = TokenClassifiers.load(scope, withPooler);
      float[][] expected = baseModel.classify(input(token)).logits();
      float[][] actual = poolerModel.classify(input(token)).logits();
      for (int i = 0; i < expected.length; i++) {
        assertArrayEquals(expected[i], actual[i]);
      }
    }
  }
}
