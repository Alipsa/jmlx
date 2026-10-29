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
   * A tiny qwen3 checkpoint: explicit {@code head_dim} (32, deliberately different from {@code
   * hidden_size / num_attention_heads == 16}), no projection biases unless requested, and the
   * float-only {@code q_norm}/{@code k_norm} weights.
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

  // The tiny Qwen3-VL geometry: the 21-token fixture vocabulary (sentinels 10-13, matching
  // qwen3vl-tokens/tokenizer.json), a two-layer text side with explicit head_dim 32
  // (2 * sum(mrope_section) = 2 * 16 = 32) and a two-block vision side with one DeepStack level.
  static final int QWEN3VL_VOCAB = 21;
  static final int QWEN3VL_TEXT_HIDDEN = 64;
  static final int QWEN3VL_TEXT_LAYERS = 2;
  static final int QWEN3VL_TEXT_HEADS = 2;
  static final int QWEN3VL_TEXT_KV_HEADS = 1;
  static final int QWEN3VL_HEAD_DIM = 32;
  static final int QWEN3VL_TEXT_INTERMEDIATE = 128;
  static final int QWEN3VL_DEPTH = 2;
  static final int QWEN3VL_VISION_HIDDEN = 64;
  static final int QWEN3VL_VISION_HEADS = 4;
  static final int QWEN3VL_VISION_INTERMEDIATE = 128;
  static final int QWEN3VL_PATCH = 16;
  static final int QWEN3VL_TEMPORAL_PATCH = 2;
  static final int QWEN3VL_MERGE = 2;
  static final int QWEN3VL_POSITION_EMBEDDINGS = 16 * 16;

  /**
   * A tiny HF float Qwen3-VL checkpoint: the strict ten-key config, a matching {@code
   * preprocessor_config.json}, the committed 21-token qwen3vl fixture tokenizer, and every weight
   * stored BF16 under the {@code model.language_model.*} / {@code model.visual.*} layout (no shard
   * metadata, no quantization declaration, no lm_head: the head is tied).
   */
  static void randomQwen3Vl(Path directory, long seed) throws IOException {
    Files.createDirectories(directory);
    Files.writeString(directory.resolve("config.json"), qwen3VlConfigJson());
    Files.writeString(directory.resolve("preprocessor_config.json"), qwen3VlPreprocessorJson());
    Files.copy(
        Path.of(
            System.getProperty("jmlx.repository.root"),
            "jmlx-models/src/test/resources/qwen3vl-tokens/tokenizer.json"),
        directory.resolve("tokenizer.json"));
    int h = QWEN3VL_TEXT_HIDDEN;
    int inter = QWEN3VL_TEXT_INTERMEDIATE;
    int v = QWEN3VL_VISION_HIDDEN;
    int visionInter = QWEN3VL_VISION_INTERMEDIATE;
    int m = v * QWEN3VL_MERGE * QWEN3VL_MERGE;
    int qkv = QWEN3VL_TEXT_HEADS * QWEN3VL_HEAD_DIM;
    int kv = QWEN3VL_TEXT_KV_HEADS * QWEN3VL_HEAD_DIM;
    try (MLXScope scope = new MLXScope()) {
      MLXRandom.seed(seed);
      Map<String, MLXArray> tensors = new LinkedHashMap<>();
      String text = "model.language_model.";
      putBf16(tensors, scope, text + "embed_tokens.weight", QWEN3VL_VOCAB, h);
      putBf16(tensors, scope, text + "norm.weight", h);
      for (int i = 0; i < QWEN3VL_TEXT_LAYERS; i++) {
        String layer = text + "layers." + i + ".";
        putBf16(tensors, scope, layer + "input_layernorm.weight", h);
        putBf16(tensors, scope, layer + "post_attention_layernorm.weight", h);
        String attn = layer + "self_attn.";
        putBf16(tensors, scope, attn + "q_proj.weight", qkv, h);
        putBf16(tensors, scope, attn + "k_proj.weight", kv, h);
        putBf16(tensors, scope, attn + "v_proj.weight", kv, h);
        putBf16(tensors, scope, attn + "o_proj.weight", h, qkv);
        putBf16(tensors, scope, attn + "q_norm.weight", QWEN3VL_HEAD_DIM);
        putBf16(tensors, scope, attn + "k_norm.weight", QWEN3VL_HEAD_DIM);
        String mlp = layer + "mlp.";
        putBf16(tensors, scope, mlp + "gate_proj.weight", inter, h);
        putBf16(tensors, scope, mlp + "up_proj.weight", inter, h);
        putBf16(tensors, scope, mlp + "down_proj.weight", h, inter);
      }
      String visual = "model.visual.";
      // PyTorch OIHW channels-first conv weight; the tower transposes it to channels-last.
      putBf16(
          tensors,
          scope,
          visual + "patch_embed.proj.weight",
          v,
          3,
          QWEN3VL_TEMPORAL_PATCH,
          QWEN3VL_PATCH,
          QWEN3VL_PATCH);
      putBf16(tensors, scope, visual + "patch_embed.proj.bias", v);
      putBf16(tensors, scope, visual + "pos_embed.weight", QWEN3VL_POSITION_EMBEDDINGS, v);
      for (int i = 0; i < QWEN3VL_DEPTH; i++) {
        String block = visual + "blocks." + i + ".";
        for (String norm : new String[] {"norm1", "norm2"}) {
          putBf16(tensors, scope, block + norm + ".weight", v);
          putBf16(tensors, scope, block + norm + ".bias", v);
        }
        putBf16(tensors, scope, block + "attn.qkv.weight", 3 * v, v);
        putBf16(tensors, scope, block + "attn.qkv.bias", 3 * v);
        putBf16(tensors, scope, block + "attn.proj.weight", v, v);
        putBf16(tensors, scope, block + "attn.proj.bias", v);
        putBf16(tensors, scope, block + "mlp.linear_fc1.weight", visionInter, v);
        putBf16(tensors, scope, block + "mlp.linear_fc1.bias", visionInter);
        putBf16(tensors, scope, block + "mlp.linear_fc2.weight", v, visionInter);
        putBf16(tensors, scope, block + "mlp.linear_fc2.bias", v);
      }
      for (String merger : new String[] {"merger", "deepstack_merger_list.0"}) {
        // The main merger norms over hidden; the DeepStack merger norms over m = hidden * merge^2.
        int normWidth = merger.equals("merger") ? v : m;
        putBf16(tensors, scope, visual + merger + ".norm.weight", normWidth);
        putBf16(tensors, scope, visual + merger + ".norm.bias", normWidth);
        putBf16(tensors, scope, visual + merger + ".linear_fc1.weight", m, m);
        putBf16(tensors, scope, visual + merger + ".linear_fc1.bias", m);
        putBf16(tensors, scope, visual + merger + ".linear_fc2.weight", v, m);
        putBf16(tensors, scope, visual + merger + ".linear_fc2.bias", v);
      }
      MLXIO.saveSafetensors(directory.resolve("model.safetensors").toString(), tensors, Map.of());
    }
  }

  static String qwen3VlConfigJson() {
    String config =
        """
        {
          "model_type": "qwen3_vl",
          "architectures": ["Qwen3VLForConditionalGeneration"],
          "tie_word_embeddings": true,
          "text_config": {
            "model_type": "qwen3_vl_text",
            "hidden_size": %d,
            "num_hidden_layers": %d,
            "num_attention_heads": %d,
            "num_key_value_heads": %d,
            "head_dim": %d,
            "intermediate_size": %d,
            "vocab_size": %d,
            "max_position_embeddings": 256,
            "rms_norm_eps": 0.000001,
            "rope_theta": 10000,
            "rope_scaling": {"rope_type": "default", "mrope_interleaved": true, "mrope_section": [8, 4, 4]},
            "attention_bias": false,
            "hidden_act": "silu",
            "attention_dropout": 0.0,
            "dtype": "bfloat16",
            "initializer_range": 0.02,
            "use_cache": true
          },
          "vision_config": {
            "model_type": "qwen3_vl",
            "depth": %d,
            "hidden_size": %d,
            "num_heads": %d,
            "intermediate_size": %d,
            "in_channels": 3,
            "patch_size": %d,
            "temporal_patch_size": %d,
            "spatial_merge_size": %d,
            "num_position_embeddings": %d,
            "out_hidden_size": %d,
            "deepstack_visual_indexes": [1],
            "hidden_act": "gelu_pytorch_tanh",
            "initializer_range": 0.02
          },
          "image_token_id": 10,
          "video_token_id": 11,
          "vision_start_token_id": 12,
          "vision_end_token_id": 13,
          "transformers_version": "4.57.0.dev0"
        }
        """;
    return config.formatted(
        QWEN3VL_TEXT_HIDDEN,
        QWEN3VL_TEXT_LAYERS,
        QWEN3VL_TEXT_HEADS,
        QWEN3VL_TEXT_KV_HEADS,
        QWEN3VL_HEAD_DIM,
        QWEN3VL_TEXT_INTERMEDIATE,
        QWEN3VL_VOCAB,
        QWEN3VL_DEPTH,
        QWEN3VL_VISION_HIDDEN,
        QWEN3VL_VISION_HEADS,
        QWEN3VL_VISION_INTERMEDIATE,
        QWEN3VL_PATCH,
        QWEN3VL_TEMPORAL_PATCH,
        QWEN3VL_MERGE,
        QWEN3VL_POSITION_EMBEDDINGS,
        QWEN3VL_TEXT_HIDDEN);
  }

  static String qwen3VlPreprocessorJson() {
    String preprocessor =
        """
        {
          "do_convert_rgb": true,
          "do_normalize": true,
          "do_rescale": true,
          "do_resize": true,
          "image_mean": [0.5, 0.5, 0.5],
          "image_processor_type": "Qwen2VLImageProcessor",
          "image_std": [0.5, 0.5, 0.5],
          "merge_size": %d,
          "patch_size": %d,
          "processor_class": "Qwen3VLProcessor",
          "resample": 3,
          "rescale_factor": 0.00392156862745098,
          "size": {"shortest_edge": 32, "longest_edge": 4096},
          "temporal_patch_size": %d
        }
        """;
    return preprocessor.formatted(QWEN3VL_MERGE, QWEN3VL_PATCH, QWEN3VL_TEMPORAL_PATCH);
  }

  private static void putBf16(
      Map<String, MLXArray> tensors, MLXScope scope, String key, int... shape) {
    tensors.put(key, MLXRandom.normal(scope, shape, DType.BFLOAT16, 0f, 0.02f));
  }

  // The tiny Phi-3.5-Vision geometry: a 21-token vocabulary (the hand-written
  // phi3v-tiny-tokens fixture), a two-layer GQA text side (head_dim 64 derived from
  // hidden/heads) with NTK-Su scaling (original max 470, max positions 2048, 32 short/long
  // factors), and the CLIP tower at its fixed width 1024 with a small MLP (128) and three
  // encoder layers (two executed, the third plan-optional).
  static final int PHI3V_VOCAB = 21;
  static final int PHI3V_TEXT_HIDDEN = 128;
  static final int PHI3V_TEXT_LAYERS = 2;
  static final int PHI3V_TEXT_HEADS = 2;
  static final int PHI3V_TEXT_KV_HEADS = 1;
  static final int PHI3V_TEXT_INTERMEDIATE = 256;
  static final int PHI3V_CLIP_LAYERS = 3;
  static final int PHI3V_CLIP_INTERMEDIATE = 128;
  static final int PHI3V_CLIP_HIDDEN = 1024;
  static final int PHI3V_ORIGINAL_MAX_POSITIONS = 470;
  static final int PHI3V_MAX_POSITIONS = 2048;

  /**
   * A tiny HF float {@code phi3_v} checkpoint: the strict config (fused qkv and gate_up
   * projections, NTK-Su rope), a matching {@code preprocessor_config.json}, the committed 21-token
   * phi3v-tiny fixture tokenizer, and every weight stored BF16 (the 4.2B artifact's storage dtype).
   * The optional last CLIP layer and {@code post_layernorm} are present, exercising the load path's
   * accepted-and-discarded handling of them.
   */
  static void randomPhi3V(Path directory, long seed) throws IOException {
    Files.createDirectories(directory);
    Files.writeString(directory.resolve("config.json"), phi3VisionConfigJson());
    Files.writeString(directory.resolve("preprocessor_config.json"), phi3VisionPreprocessorJson());
    Files.copy(
        Path.of(
            System.getProperty("jmlx.repository.root"),
            "jmlx-models/src/test/resources/phi3v-tiny-tokens/tokenizer.json"),
        directory.resolve("tokenizer.json"));
    int h = PHI3V_TEXT_HIDDEN;
    int inter = PHI3V_TEXT_INTERMEDIATE;
    int v = PHI3V_CLIP_HIDDEN;
    int visionInter = PHI3V_CLIP_INTERMEDIATE;
    int headDim = h / PHI3V_TEXT_HEADS;
    int qkv = PHI3V_TEXT_HEADS * headDim + 2 * PHI3V_TEXT_KV_HEADS * headDim;
    String clip = "model.vision_embed_tokens.img_processor.vision_model.";
    try (MLXScope scope = new MLXScope()) {
      MLXRandom.seed(seed);
      Map<String, MLXArray> tensors = new LinkedHashMap<>();
      putBf16(tensors, scope, "model.embed_tokens.weight", PHI3V_VOCAB, h);
      putBf16(tensors, scope, "model.norm.weight", h);
      putBf16(tensors, scope, "lm_head.weight", PHI3V_VOCAB, h);
      for (int i = 0; i < PHI3V_TEXT_LAYERS; i++) {
        String layer = "model.layers." + i + ".";
        putBf16(tensors, scope, layer + "input_layernorm.weight", h);
        putBf16(tensors, scope, layer + "self_attn.qkv_proj.weight", qkv, h);
        putBf16(tensors, scope, layer + "self_attn.o_proj.weight", h, h);
        putBf16(tensors, scope, layer + "post_attention_layernorm.weight", h);
        putBf16(tensors, scope, layer + "mlp.gate_up_proj.weight", 2 * inter, h);
        putBf16(tensors, scope, layer + "mlp.down_proj.weight", h, inter);
      }
      putBf16(tensors, scope, "model.vision_embed_tokens.glb_GN", 1, 1, 4 * v);
      putBf16(tensors, scope, "model.vision_embed_tokens.sub_GN", 1, 1, 1, 4 * v);
      putBf16(tensors, scope, clip + "embeddings.class_embedding", v);
      // PyTorch OIHW channels-first conv weight; the model transposes it to channels-last.
      putBf16(tensors, scope, clip + "embeddings.patch_embedding.weight", v, 3, 14, 14);
      putBf16(tensors, scope, clip + "embeddings.position_embedding.weight", 577, v);
      // The pinned checkpoint's real (typo'd) key name.
      putBf16(tensors, scope, clip + "pre_layrnorm.weight", v);
      putBf16(tensors, scope, clip + "pre_layrnorm.bias", v);
      for (int i = 0; i < PHI3V_CLIP_LAYERS; i++) {
        String block = clip + "encoder.layers." + i + ".";
        for (String projection : new String[] {"q_proj", "k_proj", "v_proj", "out_proj"}) {
          putBf16(tensors, scope, block + "self_attn." + projection + ".weight", v, v);
          putBf16(tensors, scope, block + "self_attn." + projection + ".bias", v);
        }
        for (String norm : new String[] {"layer_norm1", "layer_norm2"}) {
          putBf16(tensors, scope, block + norm + ".weight", v);
          putBf16(tensors, scope, block + norm + ".bias", v);
        }
        putBf16(tensors, scope, block + "mlp.fc1.weight", visionInter, v);
        putBf16(tensors, scope, block + "mlp.fc1.bias", visionInter);
        putBf16(tensors, scope, block + "mlp.fc2.weight", v, visionInter);
        putBf16(tensors, scope, block + "mlp.fc2.bias", v);
      }
      putBf16(tensors, scope, clip + "post_layernorm.weight", v);
      putBf16(tensors, scope, clip + "post_layernorm.bias", v);
      putBf16(tensors, scope, "model.vision_embed_tokens.img_projection.0.weight", h, 4 * v);
      putBf16(tensors, scope, "model.vision_embed_tokens.img_projection.0.bias", h);
      putBf16(tensors, scope, "model.vision_embed_tokens.img_projection.2.weight", h, h);
      putBf16(tensors, scope, "model.vision_embed_tokens.img_projection.2.bias", h);
      MLXIO.saveSafetensors(directory.resolve("model.safetensors").toString(), tensors, Map.of());
    }
  }

  static String phi3VisionConfigJson() {
    String shortFactors = factors(32, "1.0");
    String longFactors = factors(32, "2.0");
    String config =
        """
        {
          "model_type": "phi3_v",
          "architectures": ["Phi3VForCausalLM"],
          "vocab_size": %d,
          "hidden_size": %d,
          "intermediate_size": %d,
          "num_hidden_layers": %d,
          "num_attention_heads": %d,
          "num_key_value_heads": %d,
          "rms_norm_eps": 0.000001,
          "rope_theta": 10000,
          "rope_scaling": {"type": "su", "short_factor": [%s], "long_factor": [%s]},
          "original_max_position_embeddings": %d,
          "max_position_embeddings": %d,
          "hidden_act": "silu",
          "attention_dropout": 0.0,
          "tie_word_embeddings": false,
          "bos_token_id": 1,
          "eos_token_id": 2,
          "img_processor": {
            "name": "clip_vision_model",
            "model_name": "openai/clip-vit-large-patch14-336",
            "image_dim_out": %d,
            "num_img_tokens": 144
          },
          "embd_layer": {
            "projection_cls": "mlp",
            "use_hd_transform": true,
            "hd_transform_order": "sub_glb",
            "with_learnable_separator": true,
            "embedding_cls": "image"
          }
        }
        """;
    return config.formatted(
        PHI3V_VOCAB,
        PHI3V_TEXT_HIDDEN,
        PHI3V_TEXT_INTERMEDIATE,
        PHI3V_TEXT_LAYERS,
        PHI3V_TEXT_HEADS,
        PHI3V_TEXT_KV_HEADS,
        shortFactors,
        longFactors,
        PHI3V_ORIGINAL_MAX_POSITIONS,
        PHI3V_MAX_POSITIONS,
        PHI3V_CLIP_HIDDEN);
  }

  static String phi3VisionPreprocessorJson() {
    String preprocessor =
        """
        {
          "do_convert_rgb": true,
          "image_mean": [0.48145466, 0.4578275, 0.40821073],
          "image_processor_type": "Phi3VImageProcessor",
          "image_std": [0.26862954, 0.26130258, 0.27577711],
          "num_crops": 4,
          "num_img_tokens": 144,
          "processor_class": "Phi3VProcessor"
        }
        """;
    return preprocessor;
  }

  private static String factors(int count, String value) {
    StringBuilder builder = new StringBuilder();
    for (int i = 0; i < count; i++) {
      if (i > 0) {
        builder.append(',');
      }
      builder.append(value);
    }
    return builder.toString();
  }
}
