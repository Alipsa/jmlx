package se.alipsa.jmlx.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.DType;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXConv;
import se.alipsa.jmlx.core.MLXMemory;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;

/** Parameter, safety, lifecycle and inference contracts independently of the Python dispatcher. */
@EnabledIfNativeAvailable
class Phase71ContractsTest {
  @Test
  void modeContainersAndTrainingRejection() {
    try (MLXScope s = new MLXScope()) {
      Dropout dropout = new Dropout(s, 0);
      Sequential sequence = new Sequential(s, dropout, new ReLU(s));
      assertFalse(sequence.isTraining());
      MLXArray x = MLX.array(s, new float[] {-1, 2}, new int[] {2});
      assertArrayEquals(new float[] {0, 2}, sequence.forward(x).toFloatArray());
      assertArrayEquals(
          new float[] {0},
          new Sequential(s, new SiLU(s), new GELU(s))
              .forward(MLX.zeros(s, new int[] {1}, DType.FLOAT32))
              .toFloatArray());
      assertSame(x, new Sequential(s).forward(x));
      sequence.freeze();
      sequence.train(true);
      assertTrue(dropout.isTraining());
      assertThrows(UnsupportedOperationException.class, () -> sequence.forward(x));
      sequence.train(false);
      assertSame(x, dropout.forward(x));
      assertThrows(IllegalArgumentException.class, () -> new Dropout(s, 1));
      assertThrows(IllegalArgumentException.class, () -> new Dropout(s, Float.NaN));
      ModuleList modules = new ModuleList(s, sequence);
      assertSame(sequence, modules.get(0));
      assertEquals(1, modules.size());
      assertThrows(NullPointerException.class, () -> new Sequential(s, (UnaryLayer) null));
      class Parent extends Module {
        Parent() {
          super(s);
          train(true);
        }

        void register(Module module) {
          child("new", module);
        }
      }

      Parent parent = new Parent();
      parent.register(dropout);
      assertTrue(dropout.isTraining());
    }
  }

  @Test
  void convolutionReadsUpdatesAndRebindsIntoChildScope() {
    try (MLXScope model = new MLXScope()) {
      MLXArray w = MLX.ones(model, new int[] {1, 2, 1}, DType.FLOAT32);
      Conv1d layer = new Conv1d(model, w, null);
      Sequential seq = new Sequential(model, layer);
      seq.freeze();
      assertEquals(List.of("0.weight"), List.copyOf(seq.parameters().keySet()));
      try (MLXScope step = model.newChild()) {
        MLXArray x = MLX.array(step, new float[] {1, 2, 3}, new int[] {1, 3, 1});
        assertSame(step, seq.forward(x).scope());
        assertArrayEquals(new float[] {3, 5}, seq.forward(x).toFloatArray());
        MLXArray doubled = MLX.full(model, new int[] {1, 2, 1}, 2, DType.FLOAT32);
        seq.update(Map.of("0.weight", doubled));
        assertArrayEquals(new float[] {6, 10}, seq.forward(x).toFloatArray());
        LinkedHashMap<String, MLXArray> restored = new LinkedHashMap<>();
        restored.put("0.weight", w);
        seq.rebind(restored);
        assertArrayEquals(new float[] {3, 5}, seq.forward(x).toFloatArray());
      }
      try (MLXScope unrelated = new MLXScope()) {
        MLXArray x = MLX.ones(unrelated, new int[] {1, 3, 1}, DType.FLOAT32);
        assertThrows(IllegalArgumentException.class, () -> layer.forward(x));
      }
    }
  }

  @Test
  void labelsPoolingAndRanksRejectBeforeUnsafeWork() {
    try (MLXScope s = new MLXScope()) {
      MLXArray logits = MLX.ones(s, new int[] {2, 3}, DType.FLOAT32);
      MLXArray invalid = MLX.array(s, new int[] {-1, 3}, new int[] {2});
      assertThrows(IllegalArgumentException.class, () -> Losses.crossEntropy(logits, invalid));
      assertThrows(IllegalArgumentException.class, () -> Losses.nll(logits, invalid));
      assertThrows(
          IllegalArgumentException.class,
          () -> Losses.crossEntropy(logits, MLX.ones(s, new int[] {2}, DType.FLOAT32)));
      MLXArray tiny = MLX.ones(s, new int[] {1, 1, 1}, DType.FLOAT32);
      IllegalArgumentException error =
          assertThrows(
              IllegalArgumentException.class, () -> new MaxPool1d(s, 4, 1, 0).forward(tiny));
      assertTrue(error.getMessage().contains("window 4"));
      assertThrows(IllegalArgumentException.class, () -> new SinusoidalPositionalEncoding(s, 2));
      BatchNorm bn = new BatchNorm(s, 3, 1e-5f, false, null, null, null, null);
      assertThrows(
          IllegalArgumentException.class,
          () -> bn.forward(MLX.ones(s, new int[] {3}, DType.FLOAT32)));
      bn.train(true);
      assertThrows(UnsupportedOperationException.class, () -> bn.forward(logits));
      assertThrows(IllegalArgumentException.class, () -> MLXShape.tile(logits, new int[] {-1}));
      assertThrows(IllegalArgumentException.class, () -> MLXShape.repeatAxis(logits, 1, 2));
    }
  }

  @Test
  void runningStatisticsRemainDifferentiatedAfterFreeze() {
    try (MLXScope s = new MLXScope()) {
      BatchNorm layer =
          new BatchNorm(
              s,
              1,
              MLX.zeros(s, new int[] {1}, DType.FLOAT32),
              MLX.ones(s, new int[] {1}, DType.FLOAT32),
              null,
              null);
      layer.freeze();
      try (ModuleGrad grad =
              ModuleGrad.of(
                  layer,
                  (params, inputs) -> new MLXArray[] {MLXOps.sum(layer.forward(inputs[0]))});
          MLXScope step = s.newChild()) {
        MLXArray x = MLX.array(step, new float[] {2, 4}, new int[] {2, 1});
        ModuleGrad.Result result = grad.apply(step, new MLXArray[] {x});
        assertEquals(List.of("running_mean", "running_var"), List.copyOf(result.grads().keySet()));
        assertArrayEquals(
            new float[] {-2}, result.grads().get("running_mean").toFloatArray(), 1e-4f);
        assertArrayEquals(
            new float[] {-3}, result.grads().get("running_var").toFloatArray(), 1e-4f);
      }
    }
  }

  @Test
  void repeatedForwardDoesNotRetainActivationBuffers() {
    try (MLXScope model = new MLXScope()) {
      Conv2d conv = new Conv2d(model, MLX.ones(model, new int[] {4, 3, 3, 2}, DType.FLOAT32), null);
      BatchNorm norm =
          new BatchNorm(
              model,
              4,
              MLX.zeros(model, new int[] {4}, DType.FLOAT32),
              MLX.ones(model, new int[] {4}, DType.FLOAT32),
              null,
              null);
      Sequential layer =
          new Sequential(
              model, conv, norm, new Mish(model), new AvgPool2d(model, new int[] {2, 2}));
      long baseline = 0;
      for (int i = 0; i < 80; i++) {
        try (MLXScope step = model.newChild()) {
          layer.forward(MLX.ones(step, new int[] {1, 16, 16, 2}, DType.FLOAT32)).toFloatArray();
        }
        if (i == 9) {
          baseline = MLXMemory.activeBytes();
        }
      }
      assertTrue(
          MLXMemory.activeBytes() <= baseline + 65536,
          "active bytes after warmup must stay within 64 KiB");
    }
  }

  @Test
  void closedAndWrongThreadArraysFail() throws Exception {
    MLXArray closed;
    try (MLXScope s = new MLXScope()) {
      closed = MLX.ones(s, new int[] {2}, DType.FLOAT32);
    }
    assertThrows(IllegalStateException.class, () -> MLXOps.abs(closed));
    try (MLXScope s = new MLXScope()) {
      MLXArray x = MLX.ones(s, new int[] {2}, DType.FLOAT32);
      AtomicReference<Throwable> failure = new AtomicReference<>();
      Thread t =
          Thread.ofPlatform()
              .start(
                  () -> {
                    try {
                      MLXOps.abs(x);
                    } catch (Throwable error) {
                      failure.set(error);
                    }
                  });
      t.join();
      assertInstanceOf(IllegalStateException.class, failure.get());
    }
  }

  @Test
  void poolingNoncontiguousInputAndRelatedClipOperands() {
    try (MLXScope model = new MLXScope();
        MLXScope step = model.newChild()) {
      MLXArray x =
          MLX.array(
              step, new float[] {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, new int[] {1, 3, 4, 1});
      MLXArray transposed = MLXShape.transpose(x, new int[] {0, 2, 1, 3});
      assertArrayEquals(
          new float[] {5, 9, 6, 10, 7, 11},
          new MaxPool2d(model, new int[] {2, 2}, new int[] {1, 1}, new int[] {0, 0})
              .forward(transposed)
              .toFloatArray());
      MLXArray lower = MLX.full(model, new int[0], 2, DType.FLOAT32);
      MLXArray upper = MLX.full(step, new int[0], 5, DType.FLOAT32);
      MLXArray clipped = MLXOps.clip(x, lower, upper);
      assertSame(step, clipped.scope());
      assertArrayEquals(new float[] {2, 2, 2, 3, 4, 5, 5, 5, 5, 5, 5, 5}, clipped.toFloatArray());
      MLXArray weight = MLX.ones(model, new int[] {2, 2, 2, 1}, DType.FLOAT32);
      MLXArray input = MLX.ones(step, new int[] {1, 3, 3, 2}, DType.FLOAT32);
      assertThrows(
          UnsupportedOperationException.class,
          () ->
              MLXConv.convTranspose2d(
                  input,
                  weight,
                  new int[] {2, 2},
                  new int[] {0, 0},
                  new int[] {1, 1},
                  new int[] {0, 0},
                  2));
      assertThrows(
          IllegalArgumentException.class,
          () ->
              new Conv2d(
                  model, weight, null, new int[] {1}, new int[] {0, 0}, new int[] {1, 1}, 1));
    }
  }

  @Test
  void simpleLayerAndLossCasesHaveIndependentKnownAnswers() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray zero = MLX.zeros(scope, new int[] {1}, DType.FLOAT32);
      UnaryLayer[] activations = {
        new ReLU(scope),
        new LeakyReLU(scope),
        new ELU(scope),
        new SELU(scope),
        new Tanh(scope),
        new Sigmoid(scope),
        new Softplus(scope),
        new Mish(scope),
        new HardSwish(scope),
        new QuickGELU(scope)
      };
      float[] answers = {0, 0, 0, 0, 0, 0.5f, (float) Math.log(2), 0, 0, 0};
      for (int i = 0; i < activations.length; i++) {
        assertArrayEquals(
            new float[] {answers[i]}, activations[i].forward(zero).toFloatArray(), 1e-6f);
      }
      MLXArray spatial = MLX.array(scope, new float[] {1, 3}, new int[] {1, 2, 1});
      assertArrayEquals(
          new float[] {-1, 1},
          new InstanceNorm(scope, 1, 0, null, null).forward(spatial).toFloatArray(),
          1e-6f);
      assertArrayEquals(
          new float[] {-1, 1},
          new BatchNorm(scope, 1, 0, false, null, null, null, null).forward(spatial).toFloatArray(),
          1e-6f);
      MLXArray grouped = MLX.array(scope, new float[] {1, 3, 2, 4}, new int[] {1, 1, 4});
      assertArrayEquals(
          new float[] {-1, -1, 1, 1},
          new GroupNorm(scope, 2, 4, 0, false, null, null).forward(grouped).toFloatArray(),
          1e-6f);
      assertArrayEquals(
          new float[] {-1, 1, -1, 1},
          new GroupNorm(scope, 2, 4, 0, true, null, null).forward(grouped).toFloatArray(),
          1e-6f);
      assertArrayEquals(
          new float[] {1, 1, 3, 3}, new Upsample(scope, 2).forward(spatial).toFloatArray());
      assertArrayEquals(
          new float[] {1, 1.5f, 2.5f, 3},
          new Upsample(scope, new double[] {2}, Upsample.Mode.LINEAR, false)
              .forward(spatial)
              .toFloatArray(),
          1e-6f);
      assertThrows(
          IllegalArgumentException.class,
          () ->
              new Upsample(scope, new double[] {0.5}, Upsample.Mode.LINEAR, true).forward(spatial));
      MLXArray square = MLX.array(scope, new float[] {1, 2, 3, 4}, new int[] {1, 2, 2, 1});
      assertArrayEquals(
          new float[] {4}, new MaxPool2d(scope, new int[] {2, 2}).forward(square).toFloatArray());
      assertArrayEquals(
          new float[] {2.5f},
          new AvgPool2d(scope, new int[] {2, 2}).forward(square).toFloatArray());
      assertArrayEquals(new float[] {3}, new MaxPool1d(scope, 2).forward(spatial).toFloatArray());
      assertArrayEquals(new float[] {2}, new AvgPool1d(scope, 2).forward(spatial).toFloatArray());
      for (int rank = 1; rank <= 3; rank++) {
        int[] inputShape = new int[rank + 2];
        java.util.Arrays.fill(inputShape, 1);
        int[] weightShape = inputShape.clone();
        for (int i = 1; i <= rank; i++) {
          weightShape[i] = 2;
        }
        MLXArray x = MLX.full(scope, inputShape, 2, DType.FLOAT32);
        MLXArray w = MLX.ones(scope, weightShape, DType.FLOAT32);
        UnaryLayer transpose =
            switch (rank) {
              case 1 -> new ConvTranspose1d(scope, w, null);
              case 2 -> new ConvTranspose2d(scope, w, null);
              default -> new ConvTranspose3d(scope, w, null);
            };
        float[] expected = new float[1 << rank];
        java.util.Arrays.fill(expected, 2);
        assertArrayEquals(expected, transpose.forward(x).toFloatArray());
        UnaryLayer convolution =
            switch (rank) {
              case 1 -> new Conv1d(scope, w, null);
              case 2 -> new Conv2d(scope, w, null);
              default -> new Conv3d(scope, w, null);
            };
        assertArrayEquals(
            new float[] {2 * (1 << rank)},
            convolution.forward(MLX.full(scope, weightShape, 2, DType.FLOAT32)).toFloatArray());
      }
      MLXArray scores = MLX.zeros(scope, new int[] {1, 1, 2, 2}, DType.FLOAT32);
      assertArrayEquals(
          new float[] {0, -1f / 256, -1f / 256, 0},
          new ALiBi(scope).forward(scores).toFloatArray());
      assertArrayEquals(
          new float[] {0, 0, (float) Math.sqrt(0.5), (float) Math.sqrt(0.5)},
          new SinusoidalPositionalEncoding(scope, 4).forward(zero).toFloatArray(),
          1e-6f);
      MLXArray logits = MLX.zeros(scope, new int[] {1, 2}, DType.FLOAT32);
      MLXArray label = MLX.array(scope, new int[] {0}, new int[] {1});
      float logTwo = (float) Math.log(2);
      assertArrayEquals(
          new float[] {logTwo}, Losses.crossEntropy(logits, label).toFloatArray(), 1e-6f);
      assertArrayEquals(
          new float[] {logTwo},
          Losses.nll(MLXOps.logSoftmax(logits, -1), label).toFloatArray(),
          1e-6f);
      MLXArray one = MLX.ones(scope, new int[] {1}, DType.FLOAT32);
      assertArrayEquals(
          new float[] {logTwo}, Losses.binaryCrossEntropy(zero, one).toFloatArray(), 1e-6f);
      assertArrayEquals(new float[] {1}, Losses.mse(one, zero).toFloatArray());
      assertArrayEquals(new float[] {1}, Losses.l1(one, zero).toFloatArray());
      assertArrayEquals(new float[] {0.5f}, Losses.smoothL1(one, zero).toFloatArray());
      assertArrayEquals(new float[] {0}, Losses.klDiv(logits, logits).toFloatArray());
      assertArrayEquals(
          new float[] {1},
          Losses.cosineSimilarity(
                  MLX.ones(scope, new int[] {1, 2}, DType.FLOAT32),
                  MLX.ones(scope, new int[] {1, 2}, DType.FLOAT32))
              .toFloatArray(),
          1e-6f);
    }
  }
}
