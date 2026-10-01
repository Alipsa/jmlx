package se.alipsa.jmlx.nn;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static se.alipsa.jmlx.nn.ArrayBytes.bytes;
import static se.alipsa.jmlx.nn.ArrayBytes.elementBytes;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.function.Function;
import java.util.function.Supplier;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.DType;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXMemory;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.core.MLXRandom;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * Phase 6.4.1 step 1: measures candidate A (packed {@code quantized_matmul} attention) against
 * float SDPA for criteria P1-P5. Every line starts with {@code PACKED_ATTN_}; decisions are made
 * from the printed numbers, so the test asserts only that the probe ran.
 */
@EnabledIfNativeAvailable
class PackedAttentionProbeTest {
  private static final int KV_HEADS = 2;
  private static final int PER_KV = 4;
  private static final int HEADS = KV_HEADS * PER_KV;
  private static final int RUNS = 5;
  private static final int WARMUP_CALLS = 30;
  private static final int TIMED_CALLS = 60;

  @BeforeEach
  void seed() {
    MLXRandom.seed(20261001L);
  }

  private record Cfg(DType dtype, int headDim, int bits, int group) {
    String id() {
      return dtype + "-D" + headDim + "-b" + bits + "-g" + group;
    }
  }

  private static final List<Cfg> CONFIGS =
      List.of(
          new Cfg(DType.FLOAT32, 16, 4, 32),
          new Cfg(DType.FLOAT32, 16, 8, 32),
          new Cfg(DType.FLOAT32, 64, 4, 64),
          new Cfg(DType.FLOAT32, 64, 8, 64),
          new Cfg(DType.BFLOAT16, 64, 4, 64),
          new Cfg(DType.BFLOAT16, 64, 8, 64));

  private static void printf(String format, Object... args) {
    System.out.print(String.format(Locale.ROOT, format, args));
  }

  private static MLXArray random(MLXScope scope, int[] shape, DType dtype) {
    MLXArray x = MLXRandom.normal(scope, shape, DType.FLOAT32, 0f, 1f);
    return dtype == DType.FLOAT32 ? x : MLX.astype(x, dtype);
  }

  private static double median(double[] v) {
    double[] c = v.clone();
    Arrays.sort(c);
    return c[c.length / 2];
  }

  private static double[] hostFloats(MLXArray a) {
    float[] f = MLX.astype(a, DType.FLOAT32).toFloatArray();
    double[] d = new double[f.length];
    for (int i = 0; i < f.length; i++) {
      d[i] = f[i];
    }
    return d;
  }

  private static double maxAbs(double[] v) {
    double m = 0;
    for (double x : v) {
      m = Math.max(m, Math.abs(x));
    }
    return m;
  }

  /**
   * Packs {@code x} and hoists the packed arrays into {@code target}. The pack intermediates land
   * in {@code x}'s own scope (an op allocates into its operand's scope), so the caller controls
   * when they are released by choosing where {@code x} lives; {@code target} must be {@code x}'s
   * scope or an ancestor of it.
   */
  private static PackedAttentionPrototype.Packed packInto(
      MLXScope target, MLXArray x, int group, int bits) {
    PackedAttentionPrototype.Packed p = PackedAttentionPrototype.pack(x, group, bits);
    MLX.eval(p.arrays());
    return new PackedAttentionPrototype.Packed(
        MLX.hoist(p.w(), target), MLX.hoist(p.scales(), target), MLX.hoist(p.biases(), target));
  }

  /** Peak bytes of {@code step}; the caller builds and evaluates its context first. */
  private static long peakOf(Runnable step) {
    MLXMemory.resetPeak();
    step.run();
    return MLXMemory.peakBytes();
  }

  private record Peaks(double median, double min, double max) {}

  private static Peaks peaks(int runs, Supplier<Long> one) {
    double[] v = new double[runs];
    for (int i = 0; i < runs; i++) {
      v[i] = one.get();
    }
    double[] sorted = v.clone();
    Arrays.sort(sorted);
    return new Peaks(median(v), sorted[0], sorted[sorted.length - 1]);
  }

  // ---- memory steps -------------------------------------------------------------------------

  private long floatStep(Cfg c, int context, boolean attend) {
    try (MLXScope root = new MLXScope()) {
      int[] kv = {1, KV_HEADS, context, c.headDim()};
      MLXArray k = random(root, kv, c.dtype());
      MLXArray v = random(root, kv, c.dtype());
      MLXArray nk = random(root, new int[] {1, KV_HEADS, 1, c.headDim()}, c.dtype());
      MLXArray nv = random(root, new int[] {1, KV_HEADS, 1, c.headDim()}, c.dtype());
      MLXArray q = random(root, new int[] {1, HEADS, 1, c.headDim()}, c.dtype());
      MLX.eval(k, v, nk, nv, q);
      // Every operand lives in root, so every intermediate of the step does too; root is closed
      // right after the single measured step, which is what releases them.
      return peakOf(
          () -> {
            MLXArray k2 = MLXShape.concatenate(new MLXArray[] {k, nk}, 2);
            MLXArray v2 = MLXShape.concatenate(new MLXArray[] {v, nv}, 2);
            if (attend) {
              MLXArray out = PackedAttentionPrototype.floatAttend(q, k2, v2, null, false, KV_HEADS);
              MLX.eval(out, k2, v2);
            } else {
              MLX.eval(k2, v2);
            }
          });
    }
  }

  private long packedStep(Cfg c, int context, boolean attend) {
    try (MLXScope root = new MLXScope()) {
      int[] kv = {1, KV_HEADS, context, c.headDim()};
      PackedAttentionPrototype.Packed pk;
      PackedAttentionPrototype.Packed pv;
      try (MLXScope sources = root.newChild()) {
        // The float sources and their pack intermediates live in sources (packInto allocates into
        // the source's scope), so closing it leaves only the packed arrays hoisted into root.
        pk = packInto(root, random(sources, kv, c.dtype()), c.group(), c.bits());
        pv = packInto(root, random(sources, kv, c.dtype()), c.group(), c.bits());
      }
      MLXArray nk = random(root, new int[] {1, KV_HEADS, 1, c.headDim()}, c.dtype());
      MLXArray nv = random(root, new int[] {1, KV_HEADS, 1, c.headDim()}, c.dtype());
      MLXArray q = random(root, new int[] {1, HEADS, 1, c.headDim()}, c.dtype());
      MLX.eval(nk, nv, q);
      // As in floatStep: every operand lives in root, which is closed after the one measured step.
      return peakOf(
          () -> {
            PackedAttentionPrototype.Packed k2 =
                pk.concat(PackedAttentionPrototype.pack(nk, c.group(), c.bits()));
            PackedAttentionPrototype.Packed v2 =
                pv.concat(PackedAttentionPrototype.pack(nv, c.group(), c.bits()));
            List<MLXArray> all = new ArrayList<>(Arrays.asList(k2.arrays()));
            all.addAll(Arrays.asList(v2.arrays()));
            if (attend) {
              all.add(
                  PackedAttentionPrototype.attend(
                      q, k2, v2, null, KV_HEADS, c.headDim(), c.group(), c.bits()));
            }
            MLX.eval(all.toArray(new MLXArray[0]));
          });
    }
  }

  /** Bytes per token per kv head for K and V together, from real array sizes. */
  private static double packedBytesPerKvHeadToken(Cfg c) {
    try (MLXScope scope = new MLXScope()) {
      int tokens = 256;
      MLXArray x = random(scope, new int[] {1, KV_HEADS, tokens, c.headDim()}, c.dtype());
      PackedAttentionPrototype.Packed p = PackedAttentionPrototype.pack(x, c.group(), c.bits());
      MLX.eval(p.arrays());
      long b = bytes(p.w()) + bytes(p.scales()) + bytes(p.biases());
      return 2.0 * b / (KV_HEADS * tokens);
    }
  }

  @Test
  void memoryAndScaling() {
    final int n1 = 2048;
    final int n2 = 4096;
    for (Cfg c : CONFIGS) {
      double packed = packedBytesPerKvHeadToken(c);
      double floatPerToken = 2.0 * c.headDim() * elementBytes(c.dtype());
      // Step 1, first act: the append multiplier a on packed storage, before any attention.
      Peaks a1 = peaks(RUNS, () -> packedStep(c, n1, false));
      Peaks a2 = peaks(RUNS, () -> packedStep(c, n2, false));
      double appendMultiplier = (a2.median() - a1.median()) / (n2 - n1) / KV_HEADS / packed;
      double appendSpread =
          ((a2.max() - a1.min()) - (a2.min() - a1.max())) / (n2 - n1) / KV_HEADS / packed;
      Peaks floatAppend1 = peaks(RUNS, () -> floatStep(c, n1, false));
      Peaks floatAppend2 = peaks(RUNS, () -> floatStep(c, n2, false));
      double floatAppendMult =
          (floatAppend2.median() - floatAppend1.median()) / (n2 - n1) / KV_HEADS / floatPerToken;
      printf(
          "PACKED_ATTN_APPEND cfg=%s packedPerKvHeadToken=%.2f floatPerKvHeadToken=%.1f"
              + " aMedian=%.3f appendSpread=%.3f floatAppendMultiplier=%.3f%n",
          c.id(), packed, floatPerToken, appendMultiplier, appendSpread, floatAppendMult);

      Peaks f1 = peaks(RUNS, () -> floatStep(c, n1, true));
      Peaks f2 = peaks(RUNS, () -> floatStep(c, n2, true));
      Peaks q1 = peaks(RUNS, () -> packedStep(c, n1, true));
      Peaks q2 = peaks(RUNS, () -> packedStep(c, n2, true));
      double quantGrowth = (q2.median() - q1.median()) / (n2 - n1) / KV_HEADS;
      double quantGrowthSpread =
          ((q2.max() - q1.min()) - (q2.min() - q1.max())) / (n2 - n1) / KV_HEADS;
      double floatGrowth = (f2.median() - f1.median()) / (n2 - n1) / KV_HEADS;
      // Counted from the prototype graph at L=1 with no mask: raw scores and softmax weights.
      int scoreIntermediates = 2;
      double scoreKept = scoreIntermediates * elementBytes(c.dtype()) * PER_KV * 1.0;
      double floatKept = floatPerToken;
      double bound = appendMultiplier * packed + 0.5 * floatKept + scoreKept;
      double uncertainty = quantGrowthSpread + appendSpread * packed;
      String verdict =
          uncertainty >= 0.5 * floatKept ? "INCONCLUSIVE" : (quantGrowth < bound ? "PASS" : "FAIL");
      printf(
          "PACKED_ATTN_P2P3 cfg=%s peakFloat4096=%.0f peakPacked4096=%.0f p2=%s"
              + " floatGrowth=%.2f quantGrowth=%.2f quantGrowthSpread=%.2f bound=%.2f"
              + " (a*packed=%.2f floatHalf=%.2f scoreKept=%.2f) uncertainty=%.2f half=%.2f p3=%s%n",
          c.id(),
          f2.median(),
          q2.median(),
          q2.median() < f2.median() ? "PASS" : "FAIL",
          floatGrowth,
          quantGrowth,
          quantGrowthSpread,
          bound,
          appendMultiplier * packed,
          0.5 * floatKept,
          scoreKept,
          uncertainty,
          0.5 * floatKept,
          verdict);
    }
    assertTrue(true);
  }

  // ---- accuracy (P1) ------------------------------------------------------------------------

  @Test
  void accuracy() {
    for (Cfg c : CONFIGS) {
      for (int keys : new int[] {2048, 4096}) {
        for (int length : new int[] {1, 256}) {
          for (String maskKind : new String[] {"none", "causal", "sliding"}) {
            if (length == 1 && maskKind.equals("causal")) {
              continue;
            }
            if (length == 256 && maskKind.equals("none")) {
              continue;
            }
            try (MLXScope scope = new MLXScope()) {
              MLXArray k = random(scope, new int[] {1, KV_HEADS, keys, c.headDim()}, c.dtype());
              MLXArray v = random(scope, new int[] {1, KV_HEADS, keys, c.headDim()}, c.dtype());
              MLXArray q = random(scope, new int[] {1, HEADS, length, c.headDim()}, c.dtype());
              MLXArray mask =
                  switch (maskKind) {
                    case "causal" -> AttentionMask.slidingWindow(scope, length, keys, keys);
                    case "sliding" ->
                        AttentionMask.slidingWindow(scope, length, keys, Math.min(128, keys));
                    default -> null;
                  };
              printf(
                  "%s%n",
                  report("P1", c, keys, length, maskKind, compare(scope, c, q, k, v, mask)));
            }
          }
        }
      }
    }
    assertTrue(true);
  }

  /**
   * Informational: splits the P1 error into the quantizer's own floor (float SDPA over dequantized
   * K/V versus float SDPA) and what the attention path adds (packed path versus float SDPA over
   * dequantized K/V). Any candidate that consumes the same packed K/V shares the first term.
   */
  @Test
  void quantizerFloorDiagnostic() {
    for (Cfg c : CONFIGS) {
      for (int length : new int[] {1, 256}) {
        try (MLXScope scope = new MLXScope()) {
          int keys = 2048;
          MLXArray k = random(scope, new int[] {1, KV_HEADS, keys, c.headDim()}, c.dtype());
          MLXArray v = random(scope, new int[] {1, KV_HEADS, keys, c.headDim()}, c.dtype());
          MLXArray q = random(scope, new int[] {1, HEADS, length, c.headDim()}, c.dtype());
          MLXArray mask =
              length == 1 ? null : AttentionMask.slidingWindow(scope, length, keys, keys);
          PackedAttentionPrototype.Packed pk =
              PackedAttentionPrototype.pack(k, c.group(), c.bits());
          PackedAttentionPrototype.Packed pv =
              PackedAttentionPrototype.pack(v, c.group(), c.bits());
          MLXArray kd = dequantized(pk, c, c.headDim());
          MLXArray vd = dequantized(pv, c, c.headDim());
          MLXArray exact = PackedAttentionPrototype.floatAttend(q, k, v, mask, false, KV_HEADS);
          MLXArray viaDequant =
              PackedAttentionPrototype.floatAttend(q, kd, vd, mask, false, KV_HEADS);
          MLXArray packed =
              PackedAttentionPrototype.attend(
                  q, pk, pv, mask, KV_HEADS, c.headDim(), c.group(), c.bits());
          MLX.eval(exact, viaDequant, packed);
          double[] e = hostFloats(exact);
          double[] d = hostFloats(viaDequant);
          double[] p = hostFloats(packed);
          double total = 0;
          double floor = 0;
          double added = 0;
          for (int i = 0; i < e.length; i++) {
            total = Math.max(total, Math.abs(p[i] - e[i]));
            floor = Math.max(floor, Math.abs(d[i] - e[i]));
            added = Math.max(added, Math.abs(p[i] - d[i]));
          }
          double ref = maxAbs(e);
          printf(
              "PACKED_ATTN_FLOOR cfg=%s S=%d L=%d packedVsFloatPct=%.4f quantizerFloorPct=%.4f"
                  + " attentionAddedPct=%.4f outputMaxAbs=%.4f%n",
              c.id(), keys, length, 100 * total / ref, 100 * floor / ref, 100 * added / ref, ref);
        }
      }
    }
    assertTrue(true);
  }

  /**
   * Informational: the same split with peaked attention (queries scaled up), so output has signal.
   */
  @Test
  void quantizerFloorPeakedAttention() {
    for (Cfg c : CONFIGS) {
      try (MLXScope scope = new MLXScope()) {
        int keys = 2048;
        MLXArray k = random(scope, new int[] {1, KV_HEADS, keys, c.headDim()}, c.dtype());
        MLXArray v = random(scope, new int[] {1, KV_HEADS, keys, c.headDim()}, c.dtype());
        MLXArray q =
            MLXOps.multiply(
                random(scope, new int[] {1, HEADS, 1, c.headDim()}, c.dtype()),
                MLX.full(scope, new int[] {1}, 6f, c.dtype()));
        PackedAttentionPrototype.Packed pk = PackedAttentionPrototype.pack(k, c.group(), c.bits());
        PackedAttentionPrototype.Packed pv = PackedAttentionPrototype.pack(v, c.group(), c.bits());
        MLXArray exact = PackedAttentionPrototype.floatAttend(q, k, v, null, false, KV_HEADS);
        MLXArray packed =
            PackedAttentionPrototype.attend(
                q, pk, pv, null, KV_HEADS, c.headDim(), c.group(), c.bits());
        MLX.eval(exact, packed);
        double[] e = hostFloats(exact);
        double[] p = hostFloats(packed);
        double diff = 0;
        for (int i = 0; i < e.length; i++) {
          diff = Math.max(diff, Math.abs(p[i] - e[i]));
        }
        printf(
            "PACKED_ATTN_PEAKED cfg=%s S=%d L=1 queryScale=6 errorPct=%.4f outputMaxAbs=%.4f%n",
            c.id(), keys, 100 * diff / maxAbs(e), maxAbs(e));
      }
    }
    assertTrue(true);
  }

  private static MLXArray dequantized(PackedAttentionPrototype.Packed p, Cfg c, int headDim) {
    MLXArray full =
        se.alipsa.jmlx.core.MLXQuant.dequantize(
            p.w(), p.scales(), p.biases(), c.group(), c.bits(), "affine", null, null);
    int[] s = full.shape();
    return s[3] == headDim
        ? full
        : MLXShape.slice(full, new int[] {0, 0, 0, 0}, new int[] {s[0], s[1], s[2], headDim});
  }

  @Test
  void leftPaddedBatchWithPaddedQueryRows() {
    for (Cfg c : CONFIGS) {
      try (MLXScope scope = new MLXScope()) {
        int keyWidth = 504;
        int queryWidth = 4;
        MLXArray mask =
            AttentionMask.batched(
                scope,
                new int[] {500, 300},
                new int[] {0, 0},
                new int[] {4, 2},
                queryWidth,
                keyWidth,
                0);
        MLXArray k = random(scope, new int[] {2, KV_HEADS, keyWidth, c.headDim()}, c.dtype());
        MLXArray v = random(scope, new int[] {2, KV_HEADS, keyWidth, c.headDim()}, c.dtype());
        MLXArray q = random(scope, new int[] {2, HEADS, queryWidth, c.headDim()}, c.dtype());
        printf(
            "%s%n",
            report("P1", c, keyWidth, queryWidth, "batched", compare(scope, c, q, k, v, mask)));
      }
    }
    assertTrue(true);
  }

  private record Result(double errorPct, boolean finite) {}

  private Result compare(MLXScope scope, Cfg c, MLXArray q, MLXArray k, MLXArray v, MLXArray mask) {
    MLXArray reference = PackedAttentionPrototype.floatAttend(q, k, v, mask, false, KV_HEADS);
    PackedAttentionPrototype.Packed pk = PackedAttentionPrototype.pack(k, c.group(), c.bits());
    PackedAttentionPrototype.Packed pv = PackedAttentionPrototype.pack(v, c.group(), c.bits());
    MLXArray actual =
        PackedAttentionPrototype.attend(
            q, pk, pv, mask, KV_HEADS, c.headDim(), c.group(), c.bits());
    MLX.eval(reference, actual);
    double[] r = hostFloats(reference);
    double[] a = hostFloats(actual);
    boolean finite = true;
    double diff = 0;
    for (int i = 0; i < r.length; i++) {
      if (!Double.isFinite(a[i])) {
        finite = false;
      }
      diff = Math.max(diff, Math.abs(r[i] - a[i]));
    }
    return new Result(100.0 * diff / maxAbs(r), finite);
  }

  private static String report(String id, Cfg c, int keys, int length, String mask, Result result) {
    double limit = c.bits() == 8 ? 1.0 : 5.0;
    return String.format(
        Locale.ROOT,
        "PACKED_ATTN_%s cfg=%s S=%d L=%d mask=%s errorPct=%.4f limitPct=%.1f finite=%s %s",
        id,
        c.id(),
        keys,
        length,
        mask,
        result.errorPct(),
        limit,
        result.finite(),
        result.finite() && result.errorPct() <= limit ? "PASS" : "FAIL");
  }

  // ---- time (P4) ----------------------------------------------------------------------------

  @Test
  void decodeTime() {
    for (Cfg c : CONFIGS) {
      try (MLXScope scope = new MLXScope()) {
        int keys = 4096;
        MLXArray k = random(scope, new int[] {1, KV_HEADS, keys, c.headDim()}, c.dtype());
        MLXArray v = random(scope, new int[] {1, KV_HEADS, keys, c.headDim()}, c.dtype());
        PackedAttentionPrototype.Packed pk = packInto(scope, k, c.group(), c.bits());
        PackedAttentionPrototype.Packed pv = packInto(scope, v, c.group(), c.bits());
        MLX.eval(k, v);
        double[] ms =
            timeInterleaved(
                scope,
                c,
                q -> PackedAttentionPrototype.floatAttend(q, k, v, null, false, KV_HEADS),
                q ->
                    PackedAttentionPrototype.attend(
                        q, pk, pv, null, KV_HEADS, c.headDim(), c.group(), c.bits()));
        double floatMs = ms[0];
        double packedMs = ms[1];
        printf(
            "PACKED_ATTN_P4 cfg=%s floatMs=%.4f packedMs=%.4f ratio=%.2f warmup=%d timed=%d %s%n",
            c.id(),
            floatMs,
            packedMs,
            packedMs / floatMs,
            WARMUP_CALLS,
            TIMED_CALLS,
            packedMs <= 2.0 * floatMs ? "PASS" : "FAIL");
      }
    }
    assertTrue(true);
  }

  /**
   * Median milliseconds per decode-step attention call for {@code first} and {@code second},
   * returned as {@code {firstMs, secondMs}}. The two are timed in alternation, and which goes first
   * flips every iteration, so warm-up, GPU clock state and thermal drift hit both equally; timing
   * one fully before the other made the verdict depend on run order. Each call gets a fresh query
   * in a per-call child scope: ops allocate into the innermost operand scope, so the call's
   * intermediates (the repeated-head K/V copies on the float path, scores and weights on the packed
   * path) land in that child and are released when it closes, instead of accumulating in {@code
   * scope}. The query is evaluated before the clock starts so lazy random generation is not timed.
   */
  private static double[] timeInterleaved(
      MLXScope scope,
      Cfg c,
      Function<MLXArray, MLXArray> first,
      Function<MLXArray, MLXArray> second) {
    int[] queryShape = {1, HEADS, 1, c.headDim()};
    double[] firstTimes = new double[TIMED_CALLS];
    double[] secondTimes = new double[TIMED_CALLS];
    for (int i = 0; i < WARMUP_CALLS + TIMED_CALLS; i++) {
      boolean timed = i >= WARMUP_CALLS;
      boolean firstLeads = i % 2 == 0;
      double a = timeOne(scope, queryShape, c, firstLeads ? first : second);
      double b = timeOne(scope, queryShape, c, firstLeads ? second : first);
      if (timed) {
        firstTimes[i - WARMUP_CALLS] = firstLeads ? a : b;
        secondTimes[i - WARMUP_CALLS] = firstLeads ? b : a;
      }
    }
    return new double[] {median(firstTimes), median(secondTimes)};
  }

  private static double timeOne(
      MLXScope scope, int[] queryShape, Cfg c, Function<MLXArray, MLXArray> call) {
    try (MLXScope s = scope.newChild()) {
      MLXArray q = random(s, queryShape, c.dtype());
      MLX.eval(q);
      long t = System.nanoTime();
      MLX.eval(call.apply(q));
      return (System.nanoTime() - t) / 1e6;
    }
  }

  // ---- scale dtype and GQA broadcast --------------------------------------------------------

  @Test
  void scaleDtypeAndGroupedQueryFolding() {
    for (DType dtype : new DType[] {DType.FLOAT32, DType.BFLOAT16, DType.FLOAT16}) {
      try (MLXScope scope = new MLXScope()) {
        MLXArray x = random(scope, new int[] {1, KV_HEADS, 8, 64}, dtype);
        PackedAttentionPrototype.Packed p = PackedAttentionPrototype.pack(x, 64, 4);
        MLX.eval(p.arrays());
        printf(
            "PACKED_ATTN_DTYPE input=%s packed=%s scales=%s biases=%s%n",
            dtype, p.w().dtype(), p.scales().dtype(), p.biases().dtype());
      }
    }
    printf(
        "%s%n",
        "PACKED_ATTN_GQA the prototype folds queriesPerKv into the query axis, so quantized_matmul"
            + " sees [B,kvHeads,perKv*L,Dpad] against [B,kvHeads,S,*] with equal batch dims and"
            + " needs no head broadcast (exercised by every P1 line above)");
    assertTrue(true);
  }

  // ---- rounding-flip warning (informational) ------------------------------------------------

  @Test
  void roundingFlipWarning() {
    final int keys = 2048;
    final int length = 256;
    for (Cfg c :
        new Cfg[] {
          new Cfg(DType.FLOAT32, 64, 4, 64),
          new Cfg(DType.FLOAT32, 64, 8, 64),
          new Cfg(DType.BFLOAT16, 64, 4, 64),
          new Cfg(DType.BFLOAT16, 64, 8, 64),
          new Cfg(DType.FLOAT32, 16, 4, 32),
          new Cfg(DType.FLOAT32, 16, 8, 32)
        }) {
      try (MLXScope scope = new MLXScope()) {
        int[] kv = {1, KV_HEADS, keys, c.headDim()};
        MLXArray k = random(scope, kv, c.dtype());
        MLXArray v = random(scope, kv, c.dtype());
        MLXArray q = random(scope, new int[] {1, HEADS, length, c.headDim()}, c.dtype());
        Random rng = new Random(12345);
        int ulps = c.dtype() == DType.FLOAT32 ? 4 : 1;
        MLXArray k2 = perturb(scope, k, c.dtype(), ulps, rng);
        MLXArray v2 = perturb(scope, v, c.dtype(), ulps, rng);
        MLXArray mask = AttentionMask.slidingWindow(scope, length, keys, keys);
        MLXArray attnF = PackedAttentionPrototype.floatAttend(q, k, v, mask, false, KV_HEADS);
        MLXArray attnF2 = PackedAttentionPrototype.floatAttend(q, k2, v2, mask, false, KV_HEADS);
        MLXArray attnQ = quantizedAttend(c, q, k, v, mask);
        MLXArray attnQ2 = quantizedAttend(c, q, k2, v2, mask);
        MLX.eval(attnF, attnF2, attnQ, attnQ2);
        double[] f = hostFloats(attnF);
        double[] f2 = hostFloats(attnF2);
        double[] qa = hostFloats(attnQ);
        double[] q2 = hostFloats(attnQ2);
        double scale = maxAbs(f);
        double quantizedDiff = 0;
        double floatDiff = 0;
        double flipShare = 0;
        for (int i = 0; i < f.length; i++) {
          quantizedDiff = Math.max(quantizedDiff, Math.abs(q2[i] - qa[i]));
          floatDiff = Math.max(floatDiff, Math.abs(f2[i] - f[i]));
          flipShare = Math.max(flipShare, Math.abs((q2[i] - qa[i]) - (f2[i] - f[i])));
        }
        double bound = c.bits() == 8 ? 1.0 : 5.0;
        double flipPct = 100 * flipShare / scale;
        printf(
            "PACKED_ATTN_FLIP cfg=%s ulps=%d quantizedDiffPct=%.5f floatDiffPct=%.5f"
                + " flipSharePct=%.5f warnAbovePct=%.4f %s%n",
            c.id(),
            ulps,
            100 * quantizedDiff / scale,
            100 * floatDiff / scale,
            flipPct,
            0.25 * bound,
            flipPct > 0.25 * bound ? "WARN" : "ok");
      }
    }
    assertTrue(true);
  }

  private static MLXArray quantizedAttend(
      Cfg c, MLXArray q, MLXArray k, MLXArray v, MLXArray mask) {
    return PackedAttentionPrototype.attend(
        q,
        PackedAttentionPrototype.pack(k, c.group(), c.bits()),
        PackedAttentionPrototype.pack(v, c.group(), c.bits()),
        mask,
        KV_HEADS,
        c.headDim(),
        c.group(),
        c.bits());
  }

  /** Perturbs a seeded random 10% of elements by +/- ulps of the dtype's spacing at |x|. */
  private static MLXArray perturb(MLXScope scope, MLXArray x, DType dtype, int ulps, Random rng) {
    float[] data = MLX.astype(x, DType.FLOAT32).toFloatArray();
    int mantissa = dtype == DType.FLOAT32 ? 23 : (dtype == DType.BFLOAT16 ? 7 : 10);
    float smallestNormal = dtype == DType.FLOAT16 ? 6.1035156e-5f : Float.MIN_NORMAL;
    for (int i = 0; i < data.length; i++) {
      boolean pick = rng.nextDouble() < 0.10;
      boolean positive = rng.nextBoolean();
      if (!pick) {
        continue;
      }
      float abs = Math.abs(data[i]);
      float ulp =
          abs == 0f ? smallestNormal : (float) Math.scalb(1.0, Math.getExponent(abs) - mantissa);
      data[i] = data[i] + (positive ? 1 : -1) * ulps * ulp;
    }
    MLXArray out = MLX.array(scope, data, x.shape());
    return dtype == DType.FLOAT32 ? out : MLX.astype(out, dtype);
  }
}
