package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tools.jackson.databind.ObjectMapper;
import tools.jackson.databind.node.ObjectNode;

/** These schema/path/value tests run without native staging. */
class EncoderContractsTest {
  private static final ObjectMapper JSON = new ObjectMapper();

  @Test
  void pipelineModesDimensionsAndLimitsAreValidatedWithoutNative(@TempDir Path directory)
      throws Exception {
    assertEquals(
        new Pooling(Pooling.Mode.CLS, false), BertModels.pipeline(directory, 64, 16).pooling());
    Files.createDirectory(directory.resolve("1_Pooling"));
    Files.writeString(
        directory.resolve("modules.json"),
        """
        [{"type":"sentence_transformers.models.Transformer","path":""},
         {"type":"sentence_transformers.models.Pooling","path":"1_Pooling"},
         {"type":"sentence_transformers.models.Normalize","path":"2_Normalize"}]
        """);
    Files.writeString(directory.resolve("sentence_bert_config.json"), "{\"max_seq_length\":32}");
    Path pooling = directory.resolve("1_Pooling/config.json");
    Files.writeString(
        pooling, "{\"word_embedding_dimension\":16,\"pooling_mode_mean_tokens\":true}");
    var pipeline = BertModels.pipeline(directory, 64, 16);
    assertEquals(32, pipeline.limit());
    assertEquals(new Pooling(Pooling.Mode.MEAN, true), pipeline.pooling());
    assertThrows(IllegalArgumentException.class, () -> BertModels.pipeline(directory, 64, 8));
    Files.writeString(
        pooling,
        """
        {"word_embedding_dimension":16,"pooling_mode_mean_tokens":true,
         "pooling_mode_cls_token":true}
        """);
    assertThrows(IllegalArgumentException.class, () -> BertModels.pipeline(directory, 64, 16));
    Files.writeString(
        directory.resolve("modules.json"),
        Files.readString(directory.resolve("modules.json")).replace(".Pooling", ".Dense"));
    assertThrows(IllegalArgumentException.class, () -> BertModels.pipeline(directory, 64, 16));
  }

  @Test
  void probabilityRulesUseSoftmaxOrExplicitSigmoid() {
    assertArrayEquals(new float[] {0.5f, 0.5f}, BertModels.scores(new float[] {2, 2}, false));
    assertArrayEquals(
        new float[] {0.5f, 0.8807971f}, BertModels.scores(new float[] {0, 2}, true), 1e-6f);
  }

  @Test
  void missingStartFailsBeforeNativeLoadingAndBucketsHandleLongDistances(@TempDir Path directory)
      throws Exception {
    ObjectNode config =
        (ObjectNode)
            JSON.readTree(Seq2SeqGenerationTest.checkpoint().resolve("config.json").toFile());
    config.remove("decoder_start_token_id");
    Files.writeString(directory.resolve("config.json"), JSON.writeValueAsString(config));
    assertThrows(IllegalArgumentException.class, () -> T5Model.load(null, directory));
    assertEquals(7, T5Model.relativeBucket(Integer.MIN_VALUE, false, 8, 16));
    assertEquals(7, T5Model.relativeBucket(Integer.MAX_VALUE, true, 8, 16));
    assertEquals(3, T5Model.relativeBucket(Integer.MIN_VALUE, true, 8, 16));
    assertEquals(0, T5Model.relativeBucket(1, false, 8, 16));
  }

  @Test
  void denseActFnAndGatedFlagsConflictingWithProjectionAreRejected(@TempDir Path directory)
      throws Exception {
    // HF lets explicit dense_act_fn/is_gated_act override what feed_forward_proj derives; this
    // loader only implements the derived pair, so configs carrying disagreeing keys must fail
    // before any weight loading. Consistent keys (as in both committed T5 checkpoints) pass.
    ObjectNode gated =
        (ObjectNode)
            JSON.readTree(Seq2SeqGenerationTest.checkpoint().resolve("config.json").toFile());
    gated.put("dense_act_fn", "gelu");
    Files.writeString(directory.resolve("config.json"), JSON.writeValueAsString(gated));
    assertThrows(IllegalArgumentException.class, () -> T5Model.load(null, directory));
    gated.put("dense_act_fn", "gelu_new");
    gated.put("is_gated_act", false);
    Files.writeString(directory.resolve("config.json"), JSON.writeValueAsString(gated));
    assertThrows(IllegalArgumentException.class, () -> T5Model.load(null, directory));
    ObjectNode relu =
        (ObjectNode)
            JSON.readTree(
                BertGoldenTest.root()
                    .resolve("tools/hf-reference/goldens/checkpoints/t5-relu/config.json")
                    .toFile());
    ObjectNode reluGatedAct = (ObjectNode) relu.deepCopy();
    reluGatedAct.put("is_gated_act", true);
    Files.writeString(directory.resolve("config.json"), JSON.writeValueAsString(reluGatedAct));
    assertThrows(IllegalArgumentException.class, () -> T5Model.load(null, directory));
    relu.put("dense_act_fn", "gelu_new");
    Files.writeString(directory.resolve("config.json"), JSON.writeValueAsString(relu));
    assertThrows(IllegalArgumentException.class, () -> T5Model.load(null, directory));
  }

  @Test
  void inputIdsTypesMasksAndCapacityAreCheckedWithoutNativeWork() {
    for (int[] values :
        List.of(
            new int[] {-1, 0, 1},
            new int[] {32, 0, 1},
            new int[] {1, -1, 1},
            new int[] {1, 2, 1},
            new int[] {1, 0, 2},
            new int[] {1, 0, 0})) {
      var input =
          new se.alipsa.jmlx.tokenizer.TokenizerEncoding(
              List.of(values[0]),
              List.of(values[1]),
              List.of(values[2]),
              List.of(0),
              List.of(se.alipsa.jmlx.tokenizer.TokenOffset.NONE));
      assertThrows(
          IllegalArgumentException.class,
          () -> BertModels.validateInput(input, true, 32, 2, 64, 64));
    }
    var empty =
        new se.alipsa.jmlx.tokenizer.TokenizerEncoding(
            List.of(), List.of(), List.of(), List.of(), List.of());
    assertThrows(
        IllegalArgumentException.class,
        () -> BertModels.validateInput(empty, false, 32, 2, 64, 64));
    var valid =
        new se.alipsa.jmlx.tokenizer.TokenizerEncoding(
            List.of(1, 2),
            List.of(0, 1),
            List.of(1, 1),
            List.of(0, 0),
            java.util.Collections.nCopies(2, se.alipsa.jmlx.tokenizer.TokenOffset.NONE));
    BertModels.validateInput(valid, true, 32, 2, 64, 64);
    assertThrows(
        IllegalArgumentException.class, () -> BertModels.validateInput(valid, false, 32, 2, 1, 64));
    assertThrows(
        IllegalArgumentException.class, () -> BertModels.validateInput(valid, false, 32, 2, 64, 1));
    assertThrows(
        IllegalArgumentException.class,
        () ->
            new se.alipsa.jmlx.tokenizer.TokenizerEncoding(
                List.of(1),
                List.of(),
                List.of(1),
                List.of(0),
                List.of(se.alipsa.jmlx.tokenizer.TokenOffset.NONE)));
  }

  @Test
  void resultsCopyNestedRowsAndLabels() {
    float[][] logits = {{1, 2}, {3, 4}};
    float[][] scores = {{0.2f, 0.8f}, {0.4f, 0.6f}};
    TokenClassificationResult result =
        new TokenClassificationResult(null, logits, scores, List.of("a", "b"));
    logits[0][0] = 100;
    scores[0][0] = 100;
    result.logits()[0][0] = 200;
    assertArrayEquals(new float[] {1, 2}, result.logits()[0]);
    assertArrayEquals(new float[] {0.2f, 0.8f}, result.scores()[0]);
    assertThrows(UnsupportedOperationException.class, () -> result.labels().add("c"));
    assertThrows(IllegalArgumentException.class, () -> new T5LoadOptions(0));
    assertThrows(NullPointerException.class, () -> new Pooling(null, false));
  }

  @Test
  void poolingIncludesAttendedSpecialsAndZeroNormalizationStaysZero() {
    float[][] rows = {{2, 0}, {0, 4}, {100, 100}};
    assertArrayEquals(
        new float[] {1, 2},
        BertModels.pool(rows, List.of(1, 1, 0), new Pooling(Pooling.Mode.MEAN, false)));
    assertArrayEquals(
        new float[] {2, 4},
        BertModels.pool(rows, List.of(1, 1, 0), new Pooling(Pooling.Mode.MAX, false)));
    assertArrayEquals(
        new float[] {0, 0},
        BertModels.pool(new float[][] {{0, 0}}, List.of(1), new Pooling(Pooling.Mode.CLS, true)));
  }

  @Test
  void regressionAndConflictingLabelsAreRejected() throws Exception {
    ObjectNode config =
        (ObjectNode)
            JSON.readTree(
                BertGoldenTest.root()
                    .resolve("tools/hf-reference/goldens/checkpoints/bert-sequence/config.json")
                    .toFile());
    config.put("num_labels", 1);
    config.remove("problem_type");
    assertThrows(
        IllegalArgumentException.class,
        () -> BertModels.validateConfig(config, BertModels.Task.SEQUENCE));
    ObjectNode labels = JSON.createObjectNode();
    labels.set("id2label", JSON.readTree("{\"0\":\"a\",\"1\":\"b\"}"));
    labels.set("label2id", JSON.readTree("{\"a\":1,\"b\":0}"));
    assertThrows(IllegalArgumentException.class, () -> BertModels.labels(labels, 2));
    labels.remove("label2id");
    assertEquals(List.of("a", "b"), BertModels.labels(labels, 2));
    assertEquals(List.of("LABEL_0", "LABEL_1"), BertModels.labels(JSON.createObjectNode(), 2));
  }

  @Test
  void modulePathsRejectTraversalAndSymlinkEscapes(@TempDir Path root) throws Exception {
    Path model = Files.createDirectory(root.resolve("model"));
    Path outside = Files.createDirectory(root.resolve("outside"));
    Files.createSymbolicLink(model.resolve("escape"), outside);
    for (String path :
        List.of("../outside", "/outside", "a//b", "a/./b", "a\\b", "escape/config.json")) {
      assertThrows(IllegalArgumentException.class, () -> BertModels.modulePath(model, path, false));
    }
    assertEquals(
        model.toRealPath().resolve("1_Pooling/config.json"),
        BertModels.modulePath(model, "1_Pooling/config.json", false));
  }
}
