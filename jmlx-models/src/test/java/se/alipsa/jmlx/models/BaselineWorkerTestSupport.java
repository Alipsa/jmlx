package se.alipsa.jmlx.models;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.Map;
import se.alipsa.jmlx.core.DType;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXIO;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * Tiny llama artifact plus a test-built mini workload manifest for the Phase 8.2 baseline worker
 * and adapter gate tests. The artifact is the same shape as the {@code LlamaModelTest} tiny
 * checkpoint (vocab 4, hidden 4, one layer, all-zero weights — every greedy selection is token 0,
 * first-maximum tie-break), extended with an {@code eos_token_id} the stock worker's stop-set
 * assertion can resolve. The mini manifest pins the artifact's digests and carries small run policy
 * values (timeouts and emitted token counts) so the protocol tests stay fast; the frozen production
 * manifest is never touched.
 */
final class BaselineWorkerTestSupport {
  private BaselineWorkerTestSupport() {}

  static final String CASE_NAME = "tiny-llama";

  /** A self-consistent mini repository: manifest under the canonical path, artifact beside it. */
  record Workspace(
      Path repoDir,
      Path manifestPath,
      Path artifactDir,
      String manifestSha256,
      int workerTimeoutS,
      int referencePairTimeoutS,
      int emittedTokens,
      int measuredRequests) {}

  /** Builds the mini repository under {@code tempDir} and returns its coordinates. */
  static Workspace build(
      Path tempDir,
      int workerTimeoutS,
      int referencePairTimeoutS,
      int emittedTokens,
      int measuredRequests)
      throws Exception {
    Path artifact = tempDir.resolve("artifact");
    writeTinyArtifact(artifact);
    Map<String, String> digests = new LinkedHashMap<>();
    for (String name :
        new String[] {
          "config.json", "generation_config.json", "model.safetensors", "tokenizer.json"
        }) {
      digests.put(name, BaselineWorker.sha256File(artifact.resolve(name)));
    }
    int[] ids32 = promptIds(32);
    int[] ids512 = promptIds(512);
    String manifest =
        """
        {
          "schema": "jmlx-decode-baseline-workloads-1",
          "created": "2026-11-07",
          "run_policy": {
            "worker_timeout_s": %d,
            "reference_pair_worker_timeout_s": %d,
            "cell_time_budget_s": 3600,
            "session_time_budget_s": 7200,
            "first_requests": 1,
            "warmups": 0,
            "measured_requests": %d,
            "emitted_tokens": %d,
            "cache_regime": "primed-per-request",
            "io_affected_cells": []
          },
          "controlled_generation": {
            "prompt_lengths": [32, 512],
            "max_new_tokens": %d,
            "wired_limit": "none",
            "stop_termination": false
          },
          "stock_generation": {"wired_limit": "none"},
          "text_cases": [{
            "name": "%s",
            "role": "fixture",
            "stop_token_ids": [3],
            "files": {
              "config.json": {"sha256": "%s"},
              "generation_config.json": {"sha256": "%s"},
              "model.safetensors": {"sha256": "%s"},
              "tokenizer.json": {"sha256": "%s"}
            },
            "prompts": {"per_length": {
              "32": {"ids": %s, "sha256": "%s"},
              "512": {"ids": %s, "sha256": "%s"}
            }}
          }],
          "vlm_cases": []
        }
        """
            .formatted(
                workerTimeoutS,
                referencePairTimeoutS,
                measuredRequests,
                emittedTokens,
                emittedTokens,
                CASE_NAME,
                digests.get("config.json"),
                digests.get("generation_config.json"),
                digests.get("model.safetensors"),
                digests.get("tokenizer.json"),
                idsJson(ids32),
                canonicalSha256(ids32),
                idsJson(ids512),
                canonicalSha256(ids512));
    Path manifestPath =
        tempDir
            .resolve("repo")
            .resolve("tools")
            .resolve("decode-baseline")
            .resolve("workloads.json");
    Files.createDirectories(manifestPath.getParent());
    Files.writeString(manifestPath, manifest, StandardCharsets.UTF_8);
    Path repoDir = tempDir.resolve("repo").toAbsolutePath().normalize();
    return new Workspace(
        repoDir,
        manifestPath,
        artifact,
        BaselineWorker.sha256File(manifestPath),
        workerTimeoutS,
        referencePairTimeoutS,
        emittedTokens,
        measuredRequests);
  }

  /** The tiny llama checkpoint (all-zero weights; every greedy selection is token 0). */
  static void writeTinyArtifact(Path artifactDir) throws Exception {
    Files.createDirectories(artifactDir);
    Files.writeString(
        artifactDir.resolve("config.json"),
        """
        {"model_type":"llama","vocab_size":4,"hidden_size":4,"intermediate_size":8,
         "num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":2,
         "rms_norm_eps":0.000001,"rope_theta":10000,"tie_word_embeddings":true,
         "eos_token_id":3}
        """,
        StandardCharsets.UTF_8);
    Files.writeString(
        artifactDir.resolve("generation_config.json"),
        """
        {"eos_token_id":[3]}
        """,
        StandardCharsets.UTF_8);
    Files.writeString(
        artifactDir.resolve("tokenizer.json"),
        """
        {
          "version":"1.0","truncation":null,"padding":null,"added_tokens":[],
          "normalizer":null,
          "pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,
                            "trim_offsets":true,"use_regex":false},
          "post_processor":null,
          "decoder":{"type":"ByteLevel"},
          "model":{"type":"BPE","dropout":null,"unk_token":null,
                   "continuing_subword_prefix":"","end_of_word_suffix":"",
                   "fuse_unk":false,"byte_fallback":false,"ignore_merges":false,
                   "vocab":{"a":0,"p":1,"x":2,"y":3},"merges":[]}
        }
        """,
        StandardCharsets.UTF_8);
    try (MLXScope saveScope = new MLXScope()) {
      Map<String, MLXArray> tensors = new LinkedHashMap<>();
      tensors.put("model.embed_tokens.weight", zeros(saveScope, 4, 4));
      tensors.put("model.norm.weight", zeros(saveScope, 4));
      String p = "model.layers.0.";
      tensors.put(p + "input_layernorm.weight", zeros(saveScope, 4));
      tensors.put(p + "post_attention_layernorm.weight", zeros(saveScope, 4));
      tensors.put(p + "self_attn.q_proj.weight", zeros(saveScope, 4, 4));
      tensors.put(p + "self_attn.k_proj.weight", zeros(saveScope, 4, 4));
      tensors.put(p + "self_attn.v_proj.weight", zeros(saveScope, 4, 4));
      tensors.put(p + "self_attn.o_proj.weight", zeros(saveScope, 4, 4));
      tensors.put(p + "mlp.gate_proj.weight", zeros(saveScope, 8, 4));
      tensors.put(p + "mlp.up_proj.weight", zeros(saveScope, 8, 4));
      tensors.put(p + "mlp.down_proj.weight", zeros(saveScope, 4, 8));
      MLXIO.saveSafetensors(artifactDir.resolve("model.safetensors").toString(), tensors, Map.of());
    }
  }

  /** Deterministic frozen-prompt ids (a repeat of the four-token vocabulary). */
  static int[] promptIds(int length) {
    int[] ids = new int[length];
    for (int i = 0; i < length; i++) {
      ids[i] = i % 4;
    }
    return ids;
  }

  static String idsJson(int[] ids) {
    StringBuilder out = new StringBuilder("[");
    for (int i = 0; i < ids.length; i++) {
      if (i > 0) {
        out.append(',');
      }
      out.append(ids[i]);
    }
    return out.append(']').toString();
  }

  /** The manifest prompt_generator's canonical form: compact JSON array of the ids. */
  static String canonicalSha256(int[] ids) {
    try {
      return HexFormat.of()
          .formatHex(
              MessageDigest.getInstance("SHA-256")
                  .digest(idsJson(ids).getBytes(StandardCharsets.UTF_8)));
    } catch (NoSuchAlgorithmException e) {
      throw new IllegalStateException(e);
    }
  }

  private static MLXArray zeros(MLXScope scope, int... shape) {
    return MLX.zeros(scope, shape, DType.FLOAT32);
  }
}
