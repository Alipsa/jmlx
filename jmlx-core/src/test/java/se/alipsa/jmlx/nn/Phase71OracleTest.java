package se.alipsa.jmlx.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.stream.IntStream;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXConv;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.core.OracleFixtureReader;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;
import tools.jackson.databind.JsonNode;

/** Differential evidence against CPU/full-float32 MLX 0.31.2. */
@EnabledIfNativeAvailable
@Tag("full-float32")
class Phase71OracleTest {
  @Test
  void everyCommittedCaseMatches() {
    JsonNode inputs = OracleFixtureReader.read("phase7-1-core.input.json").get("cases");
    JsonNode expected = OracleFixtureReader.read("phase7-1-core.expected.json").get("cases");
    assertEquals(inputs.size(), expected.size());
    for (int i = 0; i < inputs.size(); i++) {
      JsonNode c = inputs.get(i);
      JsonNode reference = expected.get(i);
      String name = c.get("name").asString();
      try (MLXScope scope = new MLXScope()) {
        MLXArray result = apply(scope, c);
        assertArrayEquals(ints(reference.get("shape")), result.shape(), name);
        float[] actual = result.toFloatArray();
        float[] wanted = OracleFixtureReader.floats(reference.get("values"));
        assertEquals(wanted.length, actual.length, name);
        float maximumError = 0;
        for (int j = 0; j < wanted.length; j++) {
          if (Float.isFinite(wanted[j])) {
            maximumError = Math.max(maximumError, Math.abs(actual[j] - wanted[j]));
          }
          if (!Float.isFinite(wanted[j])) {
            assertEquals(wanted[j], actual[j], name + "[" + j + "]");
          } else {
            assertEquals(
                wanted[j], actual[j], 1e-4f + 1e-5f * Math.abs(wanted[j]), name + "[" + j + "]");
          }
        }
        System.out.println(name + " maxAbsoluteError=" + maximumError);
      } catch (RuntimeException error) {
        throw new AssertionError(name, error);
      }
    }
  }

  private static int[] ints(JsonNode node) {
    if (node.isNumber()) {
      return new int[] {node.intValue()};
    }
    return IntStream.range(0, node.size()).map(i -> node.get(i).intValue()).toArray();
  }

  private static MLXArray array(MLXScope scope, JsonNode c, String key) {
    JsonNode value = c.get(key);
    if (value == null || value.isNull()) {
      return null;
    }
    int[] shape = ints(value.get("shape"));
    if ("int32".equals(value.path("dtype").asString())) {
      return MLX.array(scope, ints(value.get("values")), shape);
    }
    return MLX.array(scope, OracleFixtureReader.floats(value.get("values")), shape);
  }

  private static int[] spatial(JsonNode p, String key, int rank, int fallback) {
    JsonNode value = p.get(key);
    if (value == null) {
      return IntStream.range(0, rank).map(i -> fallback).toArray();
    }
    if (value.isNumber()) {
      return IntStream.range(0, rank).map(i -> value.intValue()).toArray();
    }
    return ints(value);
  }

  private static MLXArray apply(MLXScope s, JsonNode c) {
    String op = c.get("op").asString();
    JsonNode p = c.get("options");
    MLXArray x = array(s, c, "x");
    MLXArray y = array(s, c, "y");
    MLXArray w = array(s, c, "w");
    MLXArray b = array(s, c, "b");
    if (op.startsWith("conv") && !op.equals("conv_general")) {
      int rank = x.ndim() - 2;
      int[] stride = spatial(p, "stride", rank, 1);
      int[] padding = spatial(p, "padding", rank, 0);
      int[] dilation = spatial(p, "dilation", rank, 1);
      int[] output = spatial(p, "output_padding", rank, 0);
      int groups = p.path("groups").asInt(1);
      UnaryLayer layer =
          switch (op) {
            case "conv1d" -> new Conv1d(s, w, b, stride[0], padding[0], dilation[0], groups);
            case "conv2d" -> new Conv2d(s, w, b, stride, padding, dilation, groups);
            case "conv3d" -> new Conv3d(s, w, b, stride, padding, dilation, groups);
            case "conv_transpose1d" ->
                new ConvTranspose1d(s, w, b, stride[0], padding[0], dilation[0], output[0], groups);
            case "conv_transpose2d" ->
                new ConvTranspose2d(s, w, b, stride, padding, dilation, output, groups);
            case "conv_transpose3d" ->
                new ConvTranspose3d(s, w, b, stride, padding, dilation, output, groups);
            default -> throw new AssertionError(op);
          };
      return layer.forward(x);
    }
    Losses.Reduction reduction =
        Losses.Reduction.valueOf(
            p.path("reduction").asString("none").toUpperCase(java.util.Locale.ROOT));
    int axis = p.path("axis").asInt(-1);
    return switch (op) {
      case "dropout" -> new Dropout(s, (float) p.path("p").asDouble(0.5)).forward(x);
      case "relu" -> new ReLU(s).forward(x);
      case "leaky_relu" ->
          new LeakyReLU(s, (float) p.path("negative_slope").asDouble(0.01)).forward(x);
      case "elu" -> new ELU(s, (float) p.path("alpha").asDouble(1)).forward(x);
      case "selu" -> new SELU(s).forward(x);
      case "tanh" -> new Tanh(s).forward(x);
      case "sigmoid" -> new Sigmoid(s).forward(x);
      case "softplus" -> new Softplus(s).forward(x);
      case "mish" -> new Mish(s).forward(x);
      case "hardswish" -> new HardSwish(s).forward(x);
      case "quick_gelu" -> new QuickGELU(s).forward(x);
      case "group_norm" ->
          new GroupNorm(
                  s,
                  p.get("num_groups").asInt(),
                  p.get("dims").asInt(),
                  (float) p.path("eps").asDouble(1e-5),
                  p.path("pytorch_compatible").asBoolean(false),
                  w,
                  b)
              .forward(x);
      case "instance_norm" ->
          new InstanceNorm(s, p.get("dims").asInt(), (float) p.path("eps").asDouble(1e-5), w, b)
              .forward(x);
      case "batch_norm" ->
          new BatchNorm(
                  s,
                  p.get("num_features").asInt(),
                  (float) p.path("eps").asDouble(1e-5),
                  p.path("track_running_stats").asBoolean(true),
                  array(s, c, "mean"),
                  array(s, c, "variance"),
                  w,
                  b)
              .forward(x);
      case "max_pool1d" ->
          new MaxPool1d(
                  s,
                  p.get("kernel_size").asInt(),
                  p.get("stride").asInt(),
                  p.get("padding").asInt())
              .forward(x);
      case "avg_pool1d" ->
          new AvgPool1d(
                  s,
                  p.get("kernel_size").asInt(),
                  p.get("stride").asInt(),
                  p.get("padding").asInt())
              .forward(x);
      case "max_pool2d" ->
          new MaxPool2d(
                  s,
                  spatial(p, "kernel_size", 2, 1),
                  spatial(p, "stride", 2, 1),
                  spatial(p, "padding", 2, 0))
              .forward(x);
      case "avg_pool2d" ->
          new AvgPool2d(
                  s,
                  spatial(p, "kernel_size", 2, 1),
                  spatial(p, "stride", 2, 1),
                  spatial(p, "padding", 2, 0))
              .forward(x);
      case "upsample" ->
          new Upsample(
                  s,
                  new double[] {p.get("scale_factor").asDouble()},
                  Upsample.Mode.valueOf(
                      p.get("mode").asString().toUpperCase(java.util.Locale.ROOT)),
                  p.path("align_corners").asBoolean(false))
              .forward(x);
      case "sinusoidal" ->
          new SinusoidalPositionalEncoding(
                  s,
                  p.get("dims").asInt(),
                  (float) p.path("min_freq").asDouble(0.0001),
                  (float) p.path("max_freq").asDouble(1),
                  (float) p.path("scale").asDouble(Math.sqrt(2.0 / p.get("dims").asInt())),
                  p.path("cos_first").asBoolean(false),
                  p.path("full_turns").asBoolean(false))
              .forward(x);
      case "alibi" -> new ALiBi(s).forward(x, p.get("offset").asInt());
      case "cross_entropy" ->
          Losses.crossEntropy(
              x, y, w, axis, (float) p.path("label_smoothing").asDouble(0), reduction);
      case "binary_cross_entropy" ->
          Losses.binaryCrossEntropy(x, y, w, p.path("with_logits").asBoolean(true), reduction);
      case "nll_loss" -> Losses.nll(x, y, axis, reduction);
      case "mse_loss" -> Losses.mse(x, y, reduction);
      case "l1_loss" -> Losses.l1(x, y, reduction);
      case "smooth_l1_loss" -> Losses.smoothL1(x, y, (float) p.path("beta").asDouble(1), reduction);
      case "kl_div_loss" -> Losses.klDiv(x, y, axis, reduction);
      case "cosine_similarity_loss" ->
          Losses.cosineSimilarity(
              x, y, p.path("axis").asInt(1), (float) p.path("eps").asDouble(1e-8), reduction);
      case "minimum" -> MLXOps.minimum(x, y);
      case "abs" -> MLXOps.abs(x);
      case "floor" -> MLXOps.floor(x);
      case "log1p" -> MLXOps.log1p(x);
      case "log_softmax" -> MLXOps.logSoftmax(x, axis);
      case "max" -> MLXOps.maxAxes(x, ints(p.get("axis")), p.path("keepdims").asBoolean(false));
      case "var" ->
          MLXOps.varAxes(
              x, ints(p.get("axis")), p.path("keepdims").asBoolean(false), p.path("ddof").asInt(0));
      case "softmax" ->
          MLXOps.softmaxAxes(x, ints(p.get("axis")), p.path("precise").asBoolean(false));
      case "tile" -> MLXShape.tile(x, ints(p.get("repetitions")));
      case "repeat_axis" -> MLXShape.repeatAxis(x, p.get("repeats").asInt(), axis);
      case "clip" -> MLXOps.clip(x, array(s, c, "lower"), array(s, c, "upper"));
      case "pad_symmetric" ->
          new PadSymmetric(
                  s,
                  p.get("width").asInt(),
                  (float) p.get("value").asDouble(),
                  MLXShape.PadMode.CONSTANT)
              .forward(x);
      case "pad" ->
          new Pad(
                  s,
                  new int[] {0, 1},
                  new int[] {
                    p.get("pad_width").get(0).get(0).asInt(),
                    p.get("pad_width").get(1).get(0).asInt()
                  },
                  new int[] {
                    p.get("pad_width").get(0).get(1).asInt(),
                    p.get("pad_width").get(1).get(1).asInt()
                  },
                  (float) p.path("constant_values").asDouble(0),
                  MLXShape.PadMode.valueOf(
                      p.get("mode").asString().toUpperCase(java.util.Locale.ROOT)))
              .forward(x);
      case "as_strided" ->
          MLXShape.asStrided(
              x,
              ints(p.get("shape")),
              IntStream.of(ints(p.get("strides"))).mapToLong(i -> i).toArray(),
              p.get("offset").asLong());
      case "conv_general" ->
          MLXConv.convGeneral(
              x,
              w,
              ints(p.get("stride")),
              new int[] {1, 0},
              new int[] {0, 1},
              ints(p.get("kernel_dilation")),
              ints(p.get("input_dilation")),
              1,
              true);
      default -> throw new AssertionError(op);
    };
  }
}
