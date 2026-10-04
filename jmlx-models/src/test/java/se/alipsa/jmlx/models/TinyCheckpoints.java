package se.alipsa.jmlx.models;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import se.alipsa.jmlx.core.DType;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXIO;
import se.alipsa.jmlx.core.MLXRandom;
import se.alipsa.jmlx.memory.MLXScope;

/** Deterministic nonzero checkpoints for decoder mapping tests. */
final class TinyCheckpoints {
  private static final int HIDDEN = 64;
  private static final int HEADS = 4;
  private static final int HEAD_DIM = HIDDEN / HEADS;
  private static final int QWEN3_HEAD_DIM = 32;
  private static final int INTERMEDIATE = 128;
  private static final int VOCAB = 128;
  private static final int LAYERS = 2;

  private TinyCheckpoints() {}

  static void randomLlama(
      Path directory, long seed, int kvHeads, boolean attentionBias, boolean tiedHead)
      throws IOException {
    write(directory, "llama", seed, kvHeads, attentionBias, tiedHead);
  }

  static void randomQwen2(Path directory, long seed) throws IOException {
    write(directory, "qwen2", seed, 2, true, false);
  }

  /**
   * A tiny qwen3 checkpoint: explicit {@code head_dim} (32, deliberately different from
   * {@code hidden_size / num_attention_heads == 16}), no projection biases unless requested, and
   * the float-only {@code q_norm}/{@code k_norm} weights.
   */
  static void randomQwen3(
      Path directory, long seed, int kvHeads, boolean attentionBias, boolean tiedHead)
      throws IOException {
    if (kvHeads < 1 || kvHeads >= HEADS || HEADS % kvHeads != 0) {
      throw new IllegalArgumentException("kvHeads must divide and be smaller than heads");
    }
    Files.createDirectories(directory);
    Files.writeString(
        directory.resolve("config.json"),
        "{\"model_type\":\"qwen3\",\"vocab_size\":"
            + VOCAB
            + ",\"hidden_size\":"
            + HIDDEN
            + ",\"intermediate_size\":"
            + INTERMEDIATE
            + ",\"num_hidden_layers\":"
            + LAYERS
            + ",\"num_attention_heads\":"
            + HEADS
            + ",\"num_key_value_heads\":"
            + kvHeads
            + ",\"rms_norm_eps\":0.000001,\"rope_theta\":10000,\"head_dim\":"
            + QWEN3_HEAD_DIM
            + ",\"attention_bias\":"
            + attentionBias
            + ",\"tie_word_embeddings\":"
            + tiedHead
            + "}");
    try (MLXScope scope = new MLXScope()) {
      MLXRandom.seed(seed);
      Map<String, MLXArray> tensors = new LinkedHashMap<>();
      put(tensors, scope, "model.embed_tokens.weight", VOCAB, HIDDEN);
      put(tensors, scope, "model.norm.weight", HIDDEN);
      if (!tiedHead) {
        put(tensors, scope, "lm_head.weight", VOCAB, HIDDEN);
      }
      for (int i = 0; i < LAYERS; i++) {
        String prefix = "model.layers." + i + ".";
        put(tensors, scope, prefix + "input_layernorm.weight", HIDDEN);
        put(tensors, scope, prefix + "post_attention_layernorm.weight", HIDDEN);
        String attn = prefix + "self_attn.";
        put(tensors, scope, attn + "q_proj.weight", HEADS * QWEN3_HEAD_DIM, HIDDEN);
        put(tensors, scope, attn + "k_proj.weight", kvHeads * QWEN3_HEAD_DIM, HIDDEN);
        put(tensors, scope, attn + "v_proj.weight", kvHeads * QWEN3_HEAD_DIM, HIDDEN);
        put(tensors, scope, attn + "o_proj.weight", HIDDEN, HEADS * QWEN3_HEAD_DIM);
        put(tensors, scope, attn + "q_norm.weight", QWEN3_HEAD_DIM);
        put(tensors, scope, attn + "k_norm.weight", QWEN3_HEAD_DIM);
        if (attentionBias) {
          put(tensors, scope, attn + "q_proj.bias", HEADS * QWEN3_HEAD_DIM);
          put(tensors, scope, attn + "k_proj.bias", kvHeads * QWEN3_HEAD_DIM);
          put(tensors, scope, attn + "v_proj.bias", kvHeads * QWEN3_HEAD_DIM);
          put(tensors, scope, attn + "o_proj.bias", HIDDEN);
        }
        String mlp = prefix + "mlp.";
        put(tensors, scope, mlp + "gate_proj.weight", INTERMEDIATE, HIDDEN);
        put(tensors, scope, mlp + "up_proj.weight", INTERMEDIATE, HIDDEN);
        put(tensors, scope, mlp + "down_proj.weight", HIDDEN, INTERMEDIATE);
      }
      MLXIO.saveSafetensors(directory.resolve("model.safetensors").toString(), tensors, Map.of());
    }
  }

  private static void write(
      Path directory, String family, long seed, int kvHeads, boolean bias, boolean tied)
      throws IOException {
    if (kvHeads < 1 || kvHeads >= HEADS || HEADS % kvHeads != 0) {
      throw new IllegalArgumentException("kvHeads must divide and be smaller than heads");
    }
    Files.createDirectories(directory);
    Files.writeString(
        directory.resolve("config.json"),
        "{\"model_type\":\""
            + family
            + "\",\"vocab_size\":"
            + VOCAB
            + ",\"hidden_size\":"
            + HIDDEN
            + ",\"intermediate_size\":"
            + INTERMEDIATE
            + ",\"num_hidden_layers\":"
            + LAYERS
            + ",\"num_attention_heads\":"
            + HEADS
            + ",\"num_key_value_heads\":"
            + kvHeads
            + ",\"rms_norm_eps\":0.000001,\"rope_theta\":10000"
            + ",\"attention_bias\":"
            + (family.equals("llama") && bias)
            + ",\"tie_word_embeddings\":"
            + tied
            + "}");
    try (MLXScope scope = new MLXScope()) {
      MLXRandom.seed(seed);
      Map<String, MLXArray> tensors = new LinkedHashMap<>();
      put(tensors, scope, "model.embed_tokens.weight", VOCAB, HIDDEN);
      put(tensors, scope, "model.norm.weight", HIDDEN);
      if (!tied) {
        put(tensors, scope, "lm_head.weight", VOCAB, HIDDEN);
      }
      for (int i = 0; i < LAYERS; i++) {
        String prefix = "model.layers." + i + ".";
        put(tensors, scope, prefix + "input_layernorm.weight", HIDDEN);
        put(tensors, scope, prefix + "post_attention_layernorm.weight", HIDDEN);
        String attn = prefix + "self_attn.";
        put(tensors, scope, attn + "q_proj.weight", HIDDEN, HIDDEN);
        put(tensors, scope, attn + "k_proj.weight", kvHeads * HEAD_DIM, HIDDEN);
        put(tensors, scope, attn + "v_proj.weight", kvHeads * HEAD_DIM, HIDDEN);
        put(tensors, scope, attn + "o_proj.weight", HIDDEN, HIDDEN);
        if (bias) {
          put(tensors, scope, attn + "q_proj.bias", HIDDEN);
          put(tensors, scope, attn + "k_proj.bias", kvHeads * HEAD_DIM);
          put(tensors, scope, attn + "v_proj.bias", kvHeads * HEAD_DIM);
          if (family.equals("llama")) {
            put(tensors, scope, attn + "o_proj.bias", HIDDEN);
          }
        }
        String mlp = prefix + "mlp.";
        put(tensors, scope, mlp + "gate_proj.weight", INTERMEDIATE, HIDDEN);
        put(tensors, scope, mlp + "up_proj.weight", INTERMEDIATE, HIDDEN);
        put(tensors, scope, mlp + "down_proj.weight", HIDDEN, INTERMEDIATE);
      }
      MLXIO.saveSafetensors(directory.resolve("model.safetensors").toString(), tensors, Map.of());
    }
  }

  private static void put(Map<String, MLXArray> tensors, MLXScope scope, String key, int... shape) {
    tensors.put(key, MLXRandom.normal(scope, shape, DType.FLOAT32, 0f, 0.02f));
  }
}
