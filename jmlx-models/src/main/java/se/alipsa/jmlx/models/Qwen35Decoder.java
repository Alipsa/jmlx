package se.alipsa.jmlx.models;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import se.alipsa.jmlx.core.DType;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.nn.Activation;
import se.alipsa.jmlx.nn.Embedding;
import se.alipsa.jmlx.nn.EmbeddingLayer;
import se.alipsa.jmlx.nn.GatedDecoderAttention;
import se.alipsa.jmlx.nn.GatedDeltaNet;
import se.alipsa.jmlx.nn.GatedMlp;
import se.alipsa.jmlx.nn.HybridDecoderBlock;
import se.alipsa.jmlx.nn.KVCache;
import se.alipsa.jmlx.nn.KVCachePolicy;
import se.alipsa.jmlx.nn.Linear;
import se.alipsa.jmlx.nn.Module;
import se.alipsa.jmlx.nn.MultiAxisRotation;
import se.alipsa.jmlx.nn.QuantizedEmbedding;
import se.alipsa.jmlx.nn.QuantizedLinear;
import se.alipsa.jmlx.nn.RMSNorm;
import se.alipsa.jmlx.nn.RecurrentCache;
import se.alipsa.jmlx.nn.UnaryLayer;

/** Hybrid stack with request-local state and an explicit all-roots evaluation/commit boundary. */
final class Qwen35Decoder extends Module {
  private final Qwen35Descriptor.Text text;
  private final EmbeddingLayer embedding;
  private final List<HybridDecoderBlock> layers = new ArrayList<>();
  private final GatedDeltaNet[] recurrentLayers;
  private final RMSNorm finalNorm;
  private final UnaryLayer lmHead;

  Qwen35Decoder(MLXScope scope, Qwen35Descriptor descriptor, Map<String, MLXArray> weights) {
    super(scope);
    text = descriptor.text();
    embedding =
        child(
            "embedding",
            weights.containsKey("text.embed_tokens.scales")
                ? new QuantizedEmbedding(
                    scope,
                    weights.get("text.embed_tokens.weight"),
                    weights.get("text.embed_tokens.scales"),
                    weights.get("text.embed_tokens.biases"),
                    64,
                    4)
                : new Embedding(scope, weights.get("text.embed_tokens.weight")));
    recurrentLayers = new GatedDeltaNet[text.layers()];
    for (int i = 0; i < text.layers(); i++) {
      String p = "text.layers." + i + ".";
      RMSNorm input = norm(scope, weights, p + "input_layernorm.weight");
      RMSNorm post = norm(scope, weights, p + "post_attention_layernorm.weight");
      GatedMlp mlp =
          new GatedMlp(
              scope,
              projection(scope, weights, p + "mlp.gate_proj"),
              projection(scope, weights, p + "mlp.up_proj"),
              projection(scope, weights, p + "mlp.down_proj"),
              Activation.SILU);
      HybridDecoderBlock block;
      if (text.linearLayers().get(i)) {
        String a = p + "linear_attn.";
        GatedDeltaNet delta =
            new GatedDeltaNet(
                scope,
                text.keyHeads(),
                text.valueHeads(),
                text.keyDim(),
                text.valueDim(),
                projection(scope, weights, a + "in_proj_qkv"),
                projection(scope, weights, a + "in_proj_z"),
                projection(scope, weights, a + "in_proj_b"),
                projection(scope, weights, a + "in_proj_a"),
                weights.get(a + "conv1d.weight"),
                weights.get(a + "A_log"),
                weights.get(a + "dt_bias"),
                weights.get(a + "norm.weight"),
                text.epsilon(),
                projection(scope, weights, a + "out_proj"));
        recurrentLayers[i] = delta;
        block = new HybridDecoderBlock(scope, input, delta, post, mlp);
      } else {
        String a = p + "self_attn.";
        GatedDecoderAttention attention =
            new GatedDecoderAttention(
                scope,
                text.heads(),
                text.kvHeads(),
                text.headDim(),
                text.rotaryDim(),
                projection(scope, weights, a + "q_proj"),
                projection(scope, weights, a + "k_proj"),
                projection(scope, weights, a + "v_proj"),
                projection(scope, weights, a + "o_proj"),
                weights.get(a + "q_norm.weight"),
                weights.get(a + "k_norm.weight"),
                text.epsilon());
        block = new HybridDecoderBlock(scope, input, attention, post, mlp);
      }
      layers.add(child("layer" + i, block));
    }
    finalNorm = child("finalNorm", norm(scope, weights, "text.norm.weight"));
    lmHead = text.tied() ? null : child("lmHead", projection(scope, weights, "lm_head"));
  }

  Session newSession(MLXScope request, KVCachePolicy policy) {
    if (policy.evicts()) {
      throw new IllegalArgumentException("Qwen3.5 supports FULL retention only");
    }
    return new Session(request, policy);
  }

  /** A provisional stack update cannot be reused until all roots evaluate and commit succeeds. */
  final class Session implements AutoCloseable {
    private final RecurrentCache[] recurrent = new RecurrentCache[text.layers()];
    private final KVCache[] full = new KVCache[text.layers()];
    private final KVCachePolicy policy;
    private int position;
    private int pending;
    private boolean poisoned;
    private boolean closed;

    private Session(MLXScope request, KVCachePolicy policy) {
      this.policy = policy;
      try {
        for (int i = 0; i < text.layers(); i++) {
          if (recurrentLayers[i] != null) {
            recurrent[i] = recurrentLayers[i].newCache(request, 1);
          } else {
            full[i] = new KVCache(request, policy);
          }
        }
      } catch (RuntimeException | Error error) {
        close();
        throw error;
      }
    }

    int position() {
      return position;
    }

    boolean isPoisoned() {
      return poisoned;
    }

    RecurrentCache recurrent(int layer) {
      return recurrent[layer];
    }

    KVCache full(int layer) {
      return full[layer];
    }

    MLXArray build(MLXArray ids, MultiAxisRotation rotation, boolean lastOnly) {
      int[] shape = ids.shape();
      if (ids.dtype() != DType.INT32 || shape.length != 2 || shape[0] != 1 || shape[1] < 1) {
        throw new IllegalArgumentException("Qwen3.5 requires INT32 token IDs [1,T]");
      }
      preflight(shape[1]);
      if (rotation != null) {
        rotation.validateAgainst(1, shape[1], text.headDim(), text.rotaryDim());
      }
      try {
        return buildEmbeddings(embedding.forward(ids), rotation, lastOnly);
      } catch (RuntimeException | Error error) {
        poison();
        throw error;
      }
    }

    MLXArray buildEmbeddings(MLXArray input, MultiAxisRotation rotation, boolean lastOnly) {
      int[] shape = input.shape();
      if (input.dtype() != DType.FLOAT32
          || shape.length != 3
          || shape[0] != 1
          || shape[1] < 1
          || shape[2] != text.hidden()) {
        throw new IllegalArgumentException("Qwen3.5 requires FLOAT32 embeddings [1,T,hidden]");
      }
      preflight(shape[1]);
      if (rotation == null) {
        int[][] positions = new int[3][shape[1]];
        for (int axis = 0; axis < 3; axis++) {
          for (int j = 0; j < shape[1]; j++) {
            positions[axis][j] = position + j;
          }
        }
        rotation = text.rope().prepare(input.scope(), 1, shape[1], positions);
      }
      rotation.validateAgainst(1, shape[1], text.headDim(), text.rotaryDim());
      try {
        MLXArray hidden = input;
        for (int i = 0; i < layers.size(); i++) {
          try (MLXScope layer = input.scope().newChild()) {
            MLXArray local =
                MLXShape.reshape(MLXShape.expandDims(hidden, layer, 0), hidden.shape());
            MLXArray output =
                recurrent[i] != null
                    ? layers
                        .get(i)
                        .forward(local, recurrent[i], GatedDeltaNet.PrefillMethod.CHUNKED)
                    : layers.get(i).forward(local, full[i], null, rotation);
            MLXArray next = MLX.hoist(output, input.scope());
            if (hidden != input) {
              hidden.close();
            }
            hidden = next;
          }
        }
        MLXArray normalized = finalNorm.forward(hidden);
        if (lastOnly) {
          normalized = MLXShape.slice(normalized, new int[] {0, shape[1] - 1, 0}, shape);
        }
        MLXArray logits = text.tied() ? embedding.project(normalized) : lmHead.forward(normalized);
        pending = shape[1];
        return logits;
      } catch (RuntimeException | Error error) {
        poison();
        throw error;
      }
    }

    MLXArray[] retainedArrays() {
      List<MLXArray> arrays = new ArrayList<>();
      for (int i = 0; i < text.layers(); i++) {
        if (recurrent[i] != null) {
          arrays.add(recurrent[i].convolution());
          arrays.add(recurrent[i].state());
        } else if (full[i].length() > 0) {
          arrays.add(full[i].keys());
          arrays.add(full[i].values());
        }
      }
      return arrays.toArray(MLXArray[]::new);
    }

    void commit() {
      if (closed || poisoned || pending == 0) {
        throw new IllegalStateException("Qwen3.5 session has no committable step");
      }
      int next = Math.addExact(position, pending);
      for (int i = 0; i < text.layers(); i++) {
        long actual = recurrent[i] == null ? full[i].nextPosition() : recurrent[i].position();
        if (actual != next) {
          poison();
          throw new IllegalStateException("Qwen3.5 layer positions disagree");
        }
      }
      position = next;
      pending = 0;
    }

    void evaluateAndCommit(MLXArray logits, StepBoundaryEvaluator evaluator) {
      List<MLXArray> roots = new ArrayList<>(List.of(retainedArrays()));
      roots.add(logits);
      try {
        evaluator.evaluate(roots.toArray(MLXArray[]::new));
        commit();
      } catch (RuntimeException | Error error) {
        poison();
        throw error;
      }
    }

    void poison() {
      poisoned = true;
      for (int i = 0; i < text.layers(); i++) {
        if (recurrent[i] != null) {
          recurrent[i].poison();
        }
        if (full[i] != null) {
          full[i].poison();
        }
      }
    }

    private void preflight(int length) {
      if (closed || poisoned || pending != 0) {
        throw new IllegalStateException("Qwen3.5 session is closed, poisoned or awaiting commit");
      }
      policy.requireCapacity((long) position + length, "Qwen3.5 positions");
      if ((long) position + length > text.context()) {
        throw new IllegalArgumentException("Qwen3.5 positions exceed max_position_embeddings");
      }
      for (int i = 0; i < text.layers(); i++) {
        long actual = recurrent[i] == null ? full[i].nextPosition() : recurrent[i].position();
        if (actual != position) {
          throw new IllegalStateException("Qwen3.5 layer positions disagree before execution");
        }
      }
    }

    @Override
    public void close() {
      if (closed) {
        return;
      }
      closed = true;
      for (int i = 0; i < text.layers(); i++) {
        if (recurrent[i] != null) {
          recurrent[i].close();
        }
        if (full[i] != null) {
          full[i].reset();
        }
      }
    }
  }

  private RMSNorm norm(MLXScope scope, Map<String, MLXArray> weights, String name) {
    return new RMSNorm(scope, weights.get(name), text.epsilon());
  }

  private static UnaryLayer projection(MLXScope scope, Map<String, MLXArray> weights, String name) {
    return weights.containsKey(name + ".scales")
        ? new QuantizedLinear(
            scope,
            weights.get(name + ".weight"),
            weights.get(name + ".scales"),
            weights.get(name + ".biases"),
            null,
            64,
            4)
        : new Linear(scope, weights.get(name + ".weight"), null);
  }
}
