package se.alipsa.jmlx.models;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import se.alipsa.jmlx.core.DType;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.nn.Activation;
import se.alipsa.jmlx.nn.AttentionMask;
import se.alipsa.jmlx.nn.BidirectionalAttention;
import se.alipsa.jmlx.nn.Embedding;
import se.alipsa.jmlx.nn.EncoderBlock;
import se.alipsa.jmlx.nn.LayerNorm;
import se.alipsa.jmlx.nn.Linear;
import se.alipsa.jmlx.nn.Module;
import se.alipsa.jmlx.nn.Sequential;
import se.alipsa.jmlx.tokenizer.TokenizerEncoding;
import tools.jackson.databind.JsonNode;

/** Task-selected BERT backbone with explicit key-only masking and float32 execution. */
final class BertModels extends Module {
  enum Task {
    ENCODER,
    SEQUENCE,
    TOKEN
  }

  private final Task task;
  private final int vocab;
  private final int hidden;
  private final int types;
  private final int positions;
  private final int limit;
  private final ModelMetadata metadata;
  private final List<String> labels;
  private final boolean multiLabel;
  private final Pooling pooling;
  private final Embedding words;
  private final Embedding positionEmbedding;
  private final Embedding typeEmbedding;
  private final LayerNorm embeddingNorm;
  private final List<EncoderBlock> blocks = new ArrayList<>();
  private final Linear pooler;
  private final Linear classifier;

  static BertModels load(MLXScope scope, Path directory, Task task) throws IOException {
    JsonNode config = JsonFiles.read(directory.resolve("config.json"));
    validateConfig(config, task);
    int hidden = positive(config, "hidden_size");
    final int layers = positive(config, "num_hidden_layers");
    final int intermediate = positive(config, "intermediate_size");
    int vocab = positive(config, "vocab_size");
    int positions = positive(config, "max_position_embeddings");
    final int types = positive(config, "type_vocab_size");
    int labels = labelCount(config);
    if (task == Task.ENCODER) {
      pipeline(directory, positions, hidden);
    } else {
      labels(config, labels);
    }
    if (task != Task.ENCODER && labels <= 0) {
      throw new IllegalArgumentException("BERT classifier requires num_labels or id2label");
    }
    String prefix = task == Task.ENCODER ? "" : "bert.";
    Map<String, int[]> shapes = new LinkedHashMap<>();
    shapes.put(prefix + "embeddings.word_embeddings.weight", new int[] {vocab, hidden});
    shapes.put(prefix + "embeddings.position_embeddings.weight", new int[] {positions, hidden});
    shapes.put(prefix + "embeddings.token_type_embeddings.weight", new int[] {types, hidden});
    normShapes(shapes, prefix + "embeddings.LayerNorm", hidden);
    for (int i = 0; i < layers; i++) {
      String block = prefix + "encoder.layer." + i + ".";
      for (String projection : List.of("query", "key", "value")) {
        linearShapes(shapes, block + "attention.self." + projection, hidden, hidden);
      }
      linearShapes(shapes, block + "attention.output.dense", hidden, hidden);
      normShapes(shapes, block + "attention.output.LayerNorm", hidden);
      linearShapes(shapes, block + "intermediate.dense", intermediate, hidden);
      linearShapes(shapes, block + "output.dense", hidden, intermediate);
      normShapes(shapes, block + "output.LayerNorm", hidden);
    }
    Set<String> optional = new java.util.HashSet<>();
    optional.add(prefix + "embeddings.position_ids");
    optional.add(prefix + "embeddings.token_type_ids");
    if (task == Task.ENCODER) {
      optional.add("pooler.dense.weight");
      optional.add("pooler.dense.bias");
    }
    if (task == Task.SEQUENCE) {
      linearShapes(shapes, "bert.pooler.dense", hidden, hidden);
    }
    if (task != Task.ENCODER) {
      linearShapes(shapes, "classifier", labels, hidden);
    }
    TensorPlan plan = new TensorPlan(shapes.keySet(), optional, Set.of(), Set.of());
    CheckpointLoader.Preflight checked = CheckpointLoader.preflight(directory, plan);
    SafetensorsHeaders.validateBertBuffers(checked.shards(), prefix, positions);
    try (MLXScope staging = scope.newChild()) {
      Map<String, MLXArray> tensors = CheckpointLoader.load(staging, directory, plan);
      for (var entry : shapes.entrySet()) {
        if (!Arrays.equals(tensors.get(entry.getKey()).shape(), entry.getValue())) {
          throw new IllegalArgumentException("BERT tensor shape: " + entry.getKey());
        }
      }
      for (String key : optional) {
        if (tensors.containsKey(key) && key.startsWith("pooler.")) {
          int[] expected = key.endsWith("weight") ? new int[] {hidden, hidden} : new int[] {hidden};
          if (!Arrays.equals(tensors.get(key).shape(), expected)) {
            throw new IllegalArgumentException("BERT pooler shape: " + key);
          }
        }
      }
      if (tensors.containsKey(prefix + "pooler.dense.weight")
          != tensors.containsKey(prefix + "pooler.dense.bias")) {
        throw new IllegalArgumentException("BERT pooler requires both weight and bias");
      }
      Map<String, MLXArray> promoted = new LinkedHashMap<>();
      for (var entry : tensors.entrySet()) {
        if (entry.getKey().endsWith("position_ids") || entry.getKey().endsWith("token_type_ids")) {
          continue;
        }
        DType dtype = entry.getValue().dtype();
        if (dtype != DType.FLOAT32 && dtype != DType.FLOAT16 && dtype != DType.BFLOAT16) {
          throw new IllegalArgumentException("BERT requires floating, unquantized weights");
        }
        promoted.put(entry.getKey(), MLX.hoist(MLX.astype(entry.getValue(), DType.FLOAT32), scope));
      }
      return new BertModels(scope, config, directory, task, prefix, promoted);
    }
  }

  private BertModels(
      MLXScope scope,
      JsonNode config,
      Path directory,
      Task task,
      String prefix,
      Map<String, MLXArray> tensors)
      throws IOException {
    super(scope);
    this.task = task;
    vocab = positive(config, "vocab_size");
    hidden = positive(config, "hidden_size");
    types = positive(config, "type_vocab_size");
    positions = positive(config, "max_position_embeddings");
    final int layers = positive(config, "num_hidden_layers");
    final int heads = positive(config, "num_attention_heads");
    final float epsilon = (float) config.path("layer_norm_eps").asDouble(1e-12);
    Pipeline pipeline =
        task == Task.ENCODER
            ? pipeline(directory, positions, hidden)
            : new Pipeline(new Pooling(Pooling.Mode.CLS, false), positions);
    limit = pipeline.limit();
    pooling = pipeline.pooling();
    metadata = new EncoderMetadata("bert", vocab, layers);
    labels =
        task == Task.ENCODER
            ? List.of()
            : labels(config, tensors.get("classifier.bias").shape()[0]);
    multiLabel = "multi_label_classification".equals(config.path("problem_type").asString());
    words =
        child(
            "words",
            new Embedding(scope, tensors.get(prefix + "embeddings.word_embeddings.weight")));
    positionEmbedding =
        child(
            "positions",
            new Embedding(scope, tensors.get(prefix + "embeddings.position_embeddings.weight")));
    typeEmbedding =
        child(
            "types",
            new Embedding(scope, tensors.get(prefix + "embeddings.token_type_embeddings.weight")));
    embeddingNorm =
        child("embeddingNorm", norm(scope, tensors, prefix + "embeddings.LayerNorm", epsilon));
    for (int i = 0; i < layers; i++) {
      String name = prefix + "encoder.layer." + i + ".";
      BidirectionalAttention attention =
          new BidirectionalAttention(
              scope,
              heads,
              hidden / heads,
              (float) (1.0 / Math.sqrt(hidden / heads)),
              linear(scope, tensors, name + "attention.self.query"),
              linear(scope, tensors, name + "attention.self.key"),
              linear(scope, tensors, name + "attention.self.value"),
              linear(scope, tensors, name + "attention.output.dense"));
      blocks.add(
          child(
              "block" + i,
              new EncoderBlock(
                  scope,
                  attention,
                  norm(scope, tensors, name + "attention.output.LayerNorm", epsilon),
                  new Sequential(
                      scope,
                      linear(scope, tensors, name + "intermediate.dense"),
                      Activation.GELU.layer(scope),
                      linear(scope, tensors, name + "output.dense")),
                  norm(scope, tensors, name + "output.LayerNorm", epsilon),
                  false)));
    }
    String poolerName = prefix + "pooler.dense";
    pooler =
        tensors.containsKey(poolerName + ".weight")
            ? child("pooler", linear(scope, tensors, poolerName))
            : null;
    classifier =
        task != Task.ENCODER ? child("classifier", linear(scope, tensors, "classifier")) : null;
    train(false);
  }

  static void validateConfig(JsonNode config, Task task) {
    if (!"bert".equals(config.path("model_type").asString())) {
      throw new IllegalArgumentException("encoder task requires model_type=bert");
    }
    String expected =
        switch (task) {
          case ENCODER -> "BertModel";
          case SEQUENCE -> "BertForSequenceClassification";
          case TOKEN -> "BertForTokenClassification";
        };
    if (!config.path("architectures").isArray()
        || config.path("architectures").size() != 1
        || !expected.equals(config.path("architectures").get(0).asString())) {
      throw new IllegalArgumentException("BERT artifact architectures must select " + expected);
    }
    int hidden = positive(config, "hidden_size");
    if (hidden % positive(config, "num_attention_heads") != 0
        || config.path("is_decoder").asBoolean(false)
        || config.path("add_cross_attention").asBoolean(false)
        || !"absolute".equals(config.path("position_embedding_type").asString("absolute"))
        || !"gelu".equals(config.path("hidden_act").asString("gelu"))
        || !config.path("pruned_heads").isEmpty()
        || config.has("quantization_config")) {
      throw new IllegalArgumentException(
          "unsupported BERT attention, activation or quantization config");
    }
    double epsilon = config.path("layer_norm_eps").asDouble(1e-12);
    if (!Double.isFinite(epsilon) || epsilon <= 0) {
      throw new IllegalArgumentException("BERT layer_norm_eps must be finite and positive");
    }
    if (task != Task.ENCODER) {
      String problem = config.path("problem_type").asString("");
      int labels = labelCount(config);
      if ("regression".equals(problem)
          || (problem.isEmpty() && labels == 1)
          || !List.of("", "single_label_classification", "multi_label_classification")
              .contains(problem)) {
        throw new IllegalArgumentException(
            "unsupported BERT classification problem_type: " + problem);
      }
    }
  }

  private void validate(TokenizerEncoding input, boolean cls) {
    scope().checkAccess();
    validateInput(input, cls, vocab, types, positions, limit);
  }

  static void validateInput(
      TokenizerEncoding input, boolean cls, int vocab, int types, int positions, int limit) {
    int length = input.ids().size();
    if (length == 0
        || length > positions
        || length > limit
        || input.typeIds().size() != length
        || input.attentionMask().size() != length
        || input.specialTokensMask().size() != length
        || input.offsets().size() != length) {
      throw new IllegalArgumentException(
          "BERT input is empty, inconsistent or exceeds sequence capacity");
    }
    int valid = 0;
    for (int i = 0; i < length; i++) {
      int id = input.ids().get(i);
      int type = input.typeIds().get(i);
      int mask = input.attentionMask().get(i);
      int special = input.specialTokensMask().get(i);
      if (id < 0
          || id >= vocab
          || type < 0
          || type >= types
          || (mask != 0 && mask != 1)
          || (special != 0 && special != 1)) {
        throw new IllegalArgumentException("invalid BERT token ID, type ID or mask at " + i);
      }
      valid += mask;
    }
    if (valid == 0 || (cls && input.attentionMask().getFirst() != 1)) {
      throw new IllegalArgumentException("BERT requires attended keys and an attended CLS index 0");
    }
  }

  private MLXArray forward(MLXScope step, TokenizerEncoding input) {
    int length = input.ids().size();
    MLXArray ids = MLX.array(step, ints(input.ids()), new int[] {1, length});
    MLXArray typeIds = MLX.array(step, ints(input.typeIds()), new int[] {1, length});
    MLXArray positionIds =
        MLX.array(
            step, java.util.stream.IntStream.range(0, length).toArray(), new int[] {1, length});
    MLXArray x =
        embeddingNorm.forward(
            MLXOps.add(
                MLXOps.add(words.forward(ids), typeEmbedding.forward(typeIds)),
                positionEmbedding.forward(positionIds)));
    MLXArray mask =
        AttentionMask.bidirectional(step, new int[][] {ints(input.attentionMask())}, length);
    for (EncoderBlock block : blocks) {
      x = block.forward(x, mask, null);
    }
    return x;
  }

  EncoderResult encode(TokenizerEncoding input, Pooling policy) {
    validate(input, policy.mode() == Pooling.Mode.CLS);
    try (MLXScope step = scope().newChild()) {
      float[][] rows = rows(forward(step, input).toFloatArray(), input.ids().size(), hidden);
      return new EncoderResult(input, rows, pool(rows, input.attentionMask(), policy));
    }
  }

  SequenceClassificationResult sequence(TokenizerEncoding input) {
    validate(input, true);
    try (MLXScope step = scope().newChild()) {
      MLXArray x = forward(step, input);
      MLXArray first =
          se.alipsa.jmlx.core.MLXShape.takeAxis(
              x, MLX.array(step, new int[] {0}, new int[] {1}), 1);
      float[] logits = classifier.forward(MLXOps.tanh(pooler.forward(first))).toFloatArray();
      return new SequenceClassificationResult(logits, scores(logits, multiLabel), labels);
    }
  }

  TokenClassificationResult token(TokenizerEncoding input) {
    validate(input, false);
    try (MLXScope step = scope().newChild()) {
      float[][] logits =
          rows(
              classifier.forward(forward(step, input)).toFloatArray(),
              input.ids().size(),
              labels.size());
      float[][] scores = new float[logits.length][];
      for (int i = 0; i < logits.length; i++) {
        scores[i] = scores(logits[i], multiLabel);
      }
      return new TokenClassificationResult(input, logits, scores, labels);
    }
  }

  static float[] pool(float[][] rows, List<Integer> mask, Pooling policy) {
    float[] result = new float[rows[0].length];
    if (policy.mode() == Pooling.Mode.CLS) {
      result = rows[0].clone();
    } else {
      if (policy.mode() == Pooling.Mode.MAX) {
        Arrays.fill(result, -Float.MAX_VALUE);
      }
      int count = 0;
      for (int i = 0; i < rows.length; i++) {
        if (mask.get(i) == 0) {
          continue;
        }
        count++;
        for (int j = 0; j < result.length; j++) {
          result[j] =
              policy.mode() == Pooling.Mode.MEAN
                  ? result[j] + rows[i][j]
                  : Math.max(result[j], rows[i][j]);
        }
      }
      if (count == 0) {
        throw new IllegalArgumentException("pooling requires an attended token");
      }
      if (policy.mode() == Pooling.Mode.MEAN) {
        for (int j = 0; j < result.length; j++) {
          result[j] /= count;
        }
      }
    }
    if (policy.l2Normalize()) {
      double squared = 0;
      for (float value : result) {
        squared += (double) value * value;
      }
      if (squared > 0) {
        double norm = Math.sqrt(squared);
        for (int j = 0; j < result.length; j++) {
          result[j] /= norm;
        }
      }
    }
    return result;
  }

  static float[] scores(float[] logits, boolean multi) {
    float[] result = new float[logits.length];
    double maximum = -Double.MAX_VALUE;
    for (float logit : logits) {
      maximum = Math.max(maximum, logit);
    }
    double sum = 0;
    for (int i = 0; i < logits.length; i++) {
      result[i] = (float) (multi ? 1 / (1 + Math.exp(-logits[i])) : Math.exp(logits[i] - maximum));
      sum += result[i];
    }
    if (!multi) {
      for (int i = 0; i < result.length; i++) {
        result[i] /= sum;
      }
    }
    return result;
  }

  record Pipeline(Pooling pooling, int limit) {}

  static Pipeline pipeline(Path directory, int positions, int hidden) throws IOException {
    if (!Files.exists(directory.resolve("modules.json"))) {
      return new Pipeline(new Pooling(Pooling.Mode.CLS, false), positions);
    }
    JsonNode modules = JsonFiles.read(directory.resolve("modules.json"));
    if (!modules.isArray() || modules.size() < 2 || modules.size() > 3) {
      throw new IllegalArgumentException("unsupported sentence-transformers pipeline");
    }
    String[] expected = {"Transformer", "Pooling", "Normalize"};
    Path poolingPath = null;
    for (int i = 0; i < modules.size(); i++) {
      JsonNode module = modules.get(i);
      if (!module.path("type").asString().equals("sentence_transformers.models." + expected[i])) {
        throw new IllegalArgumentException("unsupported sentence-transformers module order/type");
      }
      String path = module.path("path").asString();
      if (i == 0 && !path.isEmpty()) {
        throw new IllegalArgumentException("only a root Transformer module is supported");
      }
      Path resolved = modulePath(directory, path, i == 0);
      if (i == 1) {
        poolingPath = resolved.resolve("config.json");
      }
    }
    JsonNode config =
        JsonFiles.read(
            modulePath(
                directory,
                directory
                    .toRealPath()
                    .relativize(poolingPath.toAbsolutePath().normalize())
                    .toString(),
                false));
    if (positive(config, "word_embedding_dimension") != hidden
        || !config.path("include_prompt").asBoolean(true)) {
      throw new IllegalArgumentException("pooling dimension or prompt exclusion is unsupported");
    }
    List<Pooling.Mode> enabled = new ArrayList<>();
    for (Pooling.Mode mode : Pooling.Mode.values()) {
      if (config
          .path(
              "pooling_mode_"
                  + mode.name().toLowerCase(java.util.Locale.ROOT)
                  + (mode == Pooling.Mode.CLS ? "_token" : "_tokens"))
          .asBoolean(false)) {
        enabled.add(mode);
      }
    }
    for (String key :
        List.of(
            "pooling_mode_mean_sqrt_len_tokens",
            "pooling_mode_weightedmean_tokens",
            "pooling_mode_lasttoken")) {
      if (config.path(key).asBoolean(false)) {
        throw new IllegalArgumentException("unsupported pooling mode");
      }
    }
    if (enabled.size() != 1) {
      throw new IllegalArgumentException("pooling concatenation is unsupported");
    }
    JsonNode sentence = JsonFiles.read(directory.resolve("sentence_bert_config.json"));
    return new Pipeline(
        new Pooling(enabled.getFirst(), modules.size() == 3),
        Math.min(positions, positive(sentence, "max_seq_length")));
  }

  static Path modulePath(Path directory, String relative, boolean allowRoot) throws IOException {
    Path root = directory.toRealPath();
    if (relative.isEmpty() && allowRoot) {
      return root;
    }
    if (relative.isEmpty() || relative.contains("\\") || Path.of(relative).isAbsolute()) {
      throw new IllegalArgumentException("module path must be relative POSIX");
    }
    Path current = root;
    for (String part : relative.split("/", -1)) {
      if (part.isEmpty() || part.equals(".") || part.equals("..")) {
        throw new IllegalArgumentException("module path contains dot or empty components");
      }
      current = current.resolve(part);
      if ((Files.exists(current) || Files.isSymbolicLink(current))
          && !current.toRealPath().startsWith(root)) {
        throw new IllegalArgumentException("module path symlink escapes model directory");
      }
    }
    return current;
  }

  static List<String> labels(JsonNode config, int count) {
    JsonNode mapping = config.path("id2label");
    if (!mapping.isMissingNode() && !mapping.isNull() && !mapping.isObject()) {
      throw new IllegalArgumentException("id2label must be an object");
    }
    List<String> labels = new ArrayList<>();
    if (!mapping.isMissingNode() && !mapping.isNull() && mapping.size() != count) {
      throw new IllegalArgumentException("id2label width differs from classifier");
    }
    for (int i = 0; i < count; i++) {
      JsonNode label = mapping.get(Integer.toString(i));
      if (!mapping.isMissingNode() && !mapping.isNull() && (label == null || !label.isString())) {
        throw new IllegalArgumentException("id2label requires every numeric index");
      }
      labels.add(label == null ? "LABEL_" + i : label.asString());
    }
    if (labels.stream().distinct().count() != labels.size()) {
      throw new IllegalArgumentException("id2label labels must be unique");
    }
    JsonNode reverse = config.path("label2id");
    if (!reverse.isMissingNode() && !reverse.isNull()) {
      if (!reverse.isObject() || reverse.size() != count) {
        throw new IllegalArgumentException("label2id must cover every label");
      }
      for (int i = 0; i < labels.size(); i++) {
        JsonNode id = reverse.get(labels.get(i));
        if (id == null || !id.isIntegralNumber() || id.asLong() != i) {
          throw new IllegalArgumentException("label2id conflicts with id2label");
        }
      }
    }
    return List.copyOf(labels);
  }

  private static int labelCount(JsonNode config) {
    if (config.has("num_labels")) {
      return positive(config, "num_labels");
    }
    return config.has("id2label") ? config.path("id2label").size() : 2;
  }

  private static int positive(JsonNode config, String key) {
    JsonNode value = config.path(key);
    if (!value.isIntegralNumber() || !value.canConvertToInt() || value.intValue() <= 0) {
      throw new IllegalArgumentException("BERT " + key + " must be a positive integer");
    }
    return value.intValue();
  }

  private static void normShapes(Map<String, int[]> shapes, String name, int width) {
    shapes.put(name + ".weight", new int[] {width});
    shapes.put(name + ".bias", new int[] {width});
  }

  private static void linearShapes(Map<String, int[]> shapes, String name, int out, int in) {
    shapes.put(name + ".weight", new int[] {out, in});
    shapes.put(name + ".bias", new int[] {out});
  }

  private static Linear linear(MLXScope scope, Map<String, MLXArray> tensors, String name) {
    return new Linear(scope, tensors.get(name + ".weight"), tensors.get(name + ".bias"));
  }

  private static LayerNorm norm(
      MLXScope scope, Map<String, MLXArray> tensors, String name, float eps) {
    return new LayerNorm(scope, tensors.get(name + ".weight"), tensors.get(name + ".bias"), eps);
  }

  private static int[] ints(List<Integer> input) {
    return input.stream().mapToInt(Integer::intValue).toArray();
  }

  static float[][] rows(float[] input, int rows, int columns) {
    float[][] result = new float[rows][columns];
    for (int i = 0; i < rows; i++) {
      System.arraycopy(input, i * columns, result[i], 0, columns);
    }
    return result;
  }

  static final class Encoder implements TextEncoderModel {
    private final BertModels model;

    Encoder(BertModels model) {
      this.model = model;
    }

    public ModelMetadata metadata() {
      return model.metadata;
    }

    public EncoderResult encode(TokenizerEncoding input) {
      return model.encode(input, model.pooling);
    }

    public EncoderResult encode(TokenizerEncoding input, Pooling pooling) {
      return model.encode(input, pooling);
    }
  }

  static final class Sequence implements SequenceClassifier {
    private final BertModels model;

    Sequence(BertModels model) {
      this.model = model;
    }

    public ModelMetadata metadata() {
      return model.metadata;
    }

    public SequenceClassificationResult classify(TokenizerEncoding input) {
      return model.sequence(input);
    }
  }

  static final class Token implements TokenClassifier {
    private final BertModels model;

    Token(BertModels model) {
      this.model = model;
    }

    public ModelMetadata metadata() {
      return model.metadata;
    }

    public TokenClassificationResult classify(TokenizerEncoding input) {
      return model.token(input);
    }
  }
}
