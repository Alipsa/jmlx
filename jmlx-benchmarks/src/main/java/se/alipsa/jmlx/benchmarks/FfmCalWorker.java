package se.alipsa.jmlx.benchmarks;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SegmentAllocator;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.nio.file.Path;
import java.time.Instant;
import java.util.LinkedHashSet;
import tools.jackson.databind.ObjectMapper;
import tools.jackson.databind.node.ArrayNode;
import tools.jackson.databind.node.ObjectNode;

/**
 * Opt-in FFM calibration worker (phase 8.2 WP2, plan section 5.1): in a fresh JVM, measures one
 * no-op downcall per representative shape of the generated mlx-c bindings, through the same
 * invocation form the generated wrappers use ({@code invokeExact} on a {@code MethodHandle} bound
 * in a private holder class's {@code static final} field — the jextract wrappers' own form, so the
 * measured call sites reference a JIT-trusted constant, not an array load).
 *
 * <p>The shapes, their C symbols and their register carrier classes are defined by {@code
 * src/main/resources/ffm_cal.c}. The 27 keys are the union of the native descriptor classes of the
 * bindings the controlled binding-count workloads actually call (review round 2, A; the derivation
 * is recorded in findings section 15.4): P = a pointer parameter, M = an 8-byte struct parameter or
 * return passed by value (every opaque mlx-c struct and every {@code mlx_optional_*} wrapper is
 * exactly that in the pinned header), and I/J/F/D/Z the scalar parameters. The M slots are declared
 * as the 8-byte struct layout, not a pointer, so the Java-side marshalling (the 8-byte struct read
 * from the segment) matches the generated wrappers' exactly; on AArch64 the C ABI carrier (one
 * 64-bit register) is identical either way.
 *
 * <p>Carrier and allocator fidelity (review round 2, A): the P/M argument carriers live on a
 * process-lifetime {@code Arena.ofShared()}, the same arena kind production {@code MLXScope.Holder}
 * uses for every native handle, and the struct-return shapes allocate their results through a
 * {@link CalScope} that mirrors {@code MLXScope.Holder.allocate} (synchronized shared-arena
 * allocation plus per-scope tracking) — the production per-call allocation cost, not a bare
 * confined arena's.
 *
 * <p>One sample is the total wall time of {@code calls} consecutive downcalls; a serial
 * (loop-carried-dependent) no-op Java loop over the same call count is measured as the timing floor
 * (warmed up like the batches).
 *
 * <p>This is a calibrated estimate of the FFM transition only (dispatch, marshalling, return
 * handling), not a causal partition of model work, and it does not measure Metal dispatches,
 * kernels or GPU synchronization (plan section 5.1).
 *
 * <p>Internal fresh-JVM worker, forked by {@link FfmCalibration}; not part of the publication
 * surface.
 */
public final class FfmCalWorker {
  private static final ObjectMapper MAPPER = new ObjectMapper();

  /** The by-value 8-byte struct layout of every opaque mlx-c struct (the M slots). */
  private static final MemoryLayout STRUCT8 = MemoryLayout.structLayout(ValueLayout.ADDRESS);

  private static final int INT_ARG = 7;
  private static final long LONG_ARG = 4096;
  private static final float FLOAT_ARG = 1.5f;
  private static final double DOUBLE_ARG = 2.5;
  private static final boolean BOOL_ARG = true;

  /** The calibration shape keys (native descriptor classes), in batch id order. */
  static final String[] KEYS = {
    "(M)I",
    "(M)J",
    "(M)P",
    "(PM)I",
    "(PMM)I",
    "(PMMM)I",
    "(PMMMM)I",
    "(PMMMMZM)I",
    "(PMMMMZMMPM)I",
    "(PMMMMMPMMM)I",
    "(PMZM)I",
    "(PMIZM)I",
    "(PMIZMFIMM)I",
    "(PMIIM)I",
    "(PMIM)I",
    "(PMMIM)I",
    "(PMMFM)I",
    "(PMMMFPMMZM)I",
    "(PMPJM)I",
    "(PMPJZM)I",
    "(PMPJPJPJM)I",
    "(PDDDIM)I",
    "(PPJMIM)I",
    "(F)M",
    "()M",
    "(PJ)M",
    "(PPII)M"
  };

  /** The no-op C symbols of the calibration shapes, in batch id order. */
  static final String[] SYMBOLS = {
    "cal_M_I",
    "cal_M_J",
    "cal_M_P",
    "cal_P_M_I",
    "cal_P_M_M_I",
    "cal_P_M_M_M_I",
    "cal_P_M_M_M_M_I",
    "cal_P_M_M_M_M_Z_M_I",
    "cal_P_M_M_M_M_Z_M_M_P_M_I",
    "cal_P_M_M_M_M_M_P_M_M_M_I",
    "cal_P_M_Z_M_I",
    "cal_P_M_I_Z_M_I",
    "cal_P_M_I_Z_M_F_I_M_M_I",
    "cal_P_M_I_I_M_I",
    "cal_P_M_I_M_I",
    "cal_P_M_M_I_M_I",
    "cal_P_M_M_F_M_I",
    "cal_P_M_M_M_F_P_M_M_Z_M_I",
    "cal_P_M_P_J_M_I",
    "cal_P_M_P_J_Z_M_I",
    "cal_P_M_P_J_P_J_P_J_M_I",
    "cal_P_D_D_D_I_M_I",
    "cal_P_P_J_M_I_M_I",
    "cal_F_M",
    "cal__M",
    "cal_P_J_M",
    "cal_P_P_I_I_M"
  };

  /**
   * The downcall descriptors, one per batch id; return layout first, then parameters. Every M
   * parameter slot is the 8-byte struct layout (by value, as the pinned header passes {@code
   * mlx_array} and friends — the generated wrappers declare the same layout); the M return slots
   * keep their own form (the pointer return of (M)P, the struct returns of the four
   * struct-producing shapes).
   */
  private static final FunctionDescriptor[] DESCS = {
    FunctionDescriptor.of(ValueLayout.JAVA_INT, STRUCT8),
    FunctionDescriptor.of(ValueLayout.JAVA_LONG, STRUCT8),
    FunctionDescriptor.of(ValueLayout.ADDRESS, STRUCT8),
    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, STRUCT8),
    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, STRUCT8, STRUCT8),
    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, STRUCT8, STRUCT8, STRUCT8),
    FunctionDescriptor.of(
        ValueLayout.JAVA_INT, ValueLayout.ADDRESS, STRUCT8, STRUCT8, STRUCT8, STRUCT8),
    FunctionDescriptor.of(
        ValueLayout.JAVA_INT,
        ValueLayout.ADDRESS,
        STRUCT8,
        STRUCT8,
        STRUCT8,
        STRUCT8,
        ValueLayout.JAVA_BOOLEAN,
        STRUCT8),
    FunctionDescriptor.of(
        ValueLayout.JAVA_INT,
        ValueLayout.ADDRESS,
        STRUCT8,
        STRUCT8,
        STRUCT8,
        STRUCT8,
        ValueLayout.JAVA_BOOLEAN,
        STRUCT8,
        STRUCT8,
        ValueLayout.ADDRESS,
        STRUCT8),
    FunctionDescriptor.of(
        ValueLayout.JAVA_INT,
        ValueLayout.ADDRESS,
        STRUCT8,
        STRUCT8,
        STRUCT8,
        STRUCT8,
        STRUCT8,
        ValueLayout.ADDRESS,
        STRUCT8,
        STRUCT8,
        STRUCT8),
    FunctionDescriptor.of(
        ValueLayout.JAVA_INT, ValueLayout.ADDRESS, STRUCT8, ValueLayout.JAVA_BOOLEAN, STRUCT8),
    FunctionDescriptor.of(
        ValueLayout.JAVA_INT,
        ValueLayout.ADDRESS,
        STRUCT8,
        ValueLayout.JAVA_INT,
        ValueLayout.JAVA_BOOLEAN,
        STRUCT8),
    FunctionDescriptor.of(
        ValueLayout.JAVA_INT,
        ValueLayout.ADDRESS,
        STRUCT8,
        ValueLayout.JAVA_INT,
        ValueLayout.JAVA_BOOLEAN,
        STRUCT8,
        ValueLayout.JAVA_FLOAT,
        ValueLayout.JAVA_INT,
        STRUCT8,
        STRUCT8),
    FunctionDescriptor.of(
        ValueLayout.JAVA_INT,
        ValueLayout.ADDRESS,
        STRUCT8,
        ValueLayout.JAVA_INT,
        ValueLayout.JAVA_INT,
        STRUCT8),
    FunctionDescriptor.of(
        ValueLayout.JAVA_INT, ValueLayout.ADDRESS, STRUCT8, ValueLayout.JAVA_INT, STRUCT8),
    FunctionDescriptor.of(
        ValueLayout.JAVA_INT, ValueLayout.ADDRESS, STRUCT8, STRUCT8, ValueLayout.JAVA_INT, STRUCT8),
    FunctionDescriptor.of(
        ValueLayout.JAVA_INT,
        ValueLayout.ADDRESS,
        STRUCT8,
        STRUCT8,
        ValueLayout.JAVA_FLOAT,
        STRUCT8),
    FunctionDescriptor.of(
        ValueLayout.JAVA_INT,
        ValueLayout.ADDRESS,
        STRUCT8,
        STRUCT8,
        STRUCT8,
        ValueLayout.JAVA_FLOAT,
        ValueLayout.ADDRESS,
        STRUCT8,
        STRUCT8,
        ValueLayout.JAVA_BOOLEAN,
        STRUCT8),
    FunctionDescriptor.of(
        ValueLayout.JAVA_INT,
        ValueLayout.ADDRESS,
        STRUCT8,
        ValueLayout.ADDRESS,
        ValueLayout.JAVA_LONG,
        STRUCT8),
    FunctionDescriptor.of(
        ValueLayout.JAVA_INT,
        ValueLayout.ADDRESS,
        STRUCT8,
        ValueLayout.ADDRESS,
        ValueLayout.JAVA_LONG,
        ValueLayout.JAVA_BOOLEAN,
        STRUCT8),
    FunctionDescriptor.of(
        ValueLayout.JAVA_INT,
        ValueLayout.ADDRESS,
        STRUCT8,
        ValueLayout.ADDRESS,
        ValueLayout.JAVA_LONG,
        ValueLayout.ADDRESS,
        ValueLayout.JAVA_LONG,
        ValueLayout.ADDRESS,
        ValueLayout.JAVA_LONG,
        STRUCT8),
    FunctionDescriptor.of(
        ValueLayout.JAVA_INT,
        ValueLayout.ADDRESS,
        ValueLayout.JAVA_DOUBLE,
        ValueLayout.JAVA_DOUBLE,
        ValueLayout.JAVA_DOUBLE,
        ValueLayout.JAVA_INT,
        STRUCT8),
    FunctionDescriptor.of(
        ValueLayout.JAVA_INT,
        ValueLayout.ADDRESS,
        ValueLayout.ADDRESS,
        ValueLayout.JAVA_LONG,
        STRUCT8,
        ValueLayout.JAVA_INT,
        STRUCT8),
    FunctionDescriptor.of(STRUCT8, ValueLayout.JAVA_FLOAT),
    FunctionDescriptor.of(STRUCT8),
    FunctionDescriptor.of(STRUCT8, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG),
    FunctionDescriptor.of(
        STRUCT8,
        ValueLayout.ADDRESS,
        ValueLayout.ADDRESS,
        ValueLayout.JAVA_INT,
        ValueLayout.JAVA_INT)
  };

  /**
   * The P/M argument carriers: ten 8-byte segments on a process-lifetime shared arena — the same
   * arena kind production {@code MLXScope.Holder} uses for every native handle (review round 2, A)
   * — so the measured marshalling reads shared-arena segments the way the real wrappers' arguments
   * are.
   */
  private static final Arena CARRIER_ARENA = Arena.ofShared();

  private static final MemorySegment[] CARRIERS = new MemorySegment[10];

  static {
    for (int i = 0; i < CARRIERS.length; i++) {
      CARRIERS[i] = CARRIER_ARENA.allocate(8);
    }
  }

  /** The shared symbol lookup; {@link #run} installs it before any {@code Handle} class loads. */
  private static final class Init {
    static SymbolLookup lookup;
  }

  /** Binds one shape's downcall (called from a holder class's static initializer). */
  private static MethodHandle downcall(int id) {
    try {
      return Linker.nativeLinker().downcallHandle(Init.lookup.findOrThrow(SYMBOLS[id]), DESCS[id]);
    } catch (RuntimeException e) {
      throw new IllegalStateException("failed to bind the " + KEYS[id] + " downcall", e);
    }
  }

  /**
   * Pre-touches every holder class so each downcall binds here — before warmup — and a binding
   * failure surfaces as a clean {@link IllegalStateException} instead of an {@code
   * ExceptionInInitializerError} on the first measured call.
   */
  private static void initializeHandles() {
    try {
      MethodHandle[] bound =
          new MethodHandle[] {
            Handle0.HANDLE,
            Handle1.HANDLE,
            Handle2.HANDLE,
            Handle3.HANDLE,
            Handle4.HANDLE,
            Handle5.HANDLE,
            Handle6.HANDLE,
            Handle7.HANDLE,
            Handle8.HANDLE,
            Handle9.HANDLE,
            Handle10.HANDLE,
            Handle11.HANDLE,
            Handle12.HANDLE,
            Handle13.HANDLE,
            Handle14.HANDLE,
            Handle15.HANDLE,
            Handle16.HANDLE,
            Handle17.HANDLE,
            Handle18.HANDLE,
            Handle19.HANDLE,
            Handle20.HANDLE,
            Handle21.HANDLE,
            Handle22.HANDLE,
            Handle23.HANDLE,
            Handle24.HANDLE,
            Handle25.HANDLE,
            Handle26.HANDLE
          };
      if (bound.length != KEYS.length) {
        throw new IllegalStateException("holder classes and shape keys mismatch");
      }
    } catch (ExceptionInInitializerError e) {
      Throwable cause = e.getCause();
      if (cause instanceof RuntimeException re) {
        throw re;
      }
      throw new IllegalStateException("failed to bind the calibration downcalls", cause);
    }
  }

  private FfmCalWorker() {}

  /**
   * Runs {@code dylib output-json iterations calls [warmup]}.
   *
   * @param args the dylib path, the raw-sample output path, the sample count, the calls per sample
   *     and the optional warmup call count
   * @throws Exception if the dylib, a downcall or the report write fails
   */
  public static void main(String[] args) throws Exception {
    if (args.length < 4 || args.length > 5) {
      throw new IllegalArgumentException(
          "usage: FfmCalWorker dylib output-json iterations calls [warmup]");
    }
    run(
        Path.of(args[0]),
        Path.of(args[1]),
        Integer.parseInt(args[2]),
        Integer.parseInt(args[3]),
        args.length == 5 ? Integer.parseInt(args[4]) : 2000);
  }

  /**
   * Loads the dylib, binds and warms every shape, measures one sample per shape per iteration
   * (rotating the shape order per iteration to avoid systematic order effects) plus the no-op
   * timing floor, and writes the raw samples as JSON.
   *
   * @param dylib the compiled calibration dylib
   * @param output the raw-sample JSON path
   * @param iterations one sample per shape per iteration
   * @param calls the downcalls per sample
   * @param warmup the warmup calls per shape before measuring
   * @throws Exception if the dylib, a downcall or the report write fails
   */
  static void run(Path dylib, Path output, int iterations, int calls, int warmup) throws Exception {
    if (iterations < 1 || calls < 1 || warmup < 0) {
      throw new IllegalArgumentException(
          "iterations and calls must be positive; warmup may be zero");
    }
    // The lookup's confined arena keeps the dylib loaded for the measurement; the carriers use a
    // process-lifetime shared arena (review round 2, A). Every sample gets a fresh CalScope,
    // closed after its batch, outside the timed region (review round 3, 1).
    try (Arena lookupArena = Arena.ofConfined()) {
      Init.lookup = SymbolLookup.libraryLookup(dylib, lookupArena);
      initializeHandles();

      long sink = 0;
      for (int w = 0; w < warmup; w++) {
        for (int i = 0; i < KEYS.length; i++) {
          CalScope sample = new CalScope();
          sink ^= batch(i, sample, 1) ^ floor(1);
          sample.close();
        }
      }
      SINK ^= sink;

      long[][] raw = new long[KEYS.length][iterations];
      long[] loop = new long[iterations];
      for (int i = 0; i < iterations; i++) {
        int offset = i % KEYS.length;
        for (int s = 0; s < KEYS.length; s++) {
          int id = (offset + s) % KEYS.length;
          CalScope sample = new CalScope();
          long t0 = System.nanoTime();
          SINK ^= batch(id, sample, calls);
          raw[id][i] = System.nanoTime() - t0;
          sample.close();
        }
        long t0 = System.nanoTime();
        SINK ^= floor(calls);
        loop[i] = System.nanoTime() - t0;
      }

      ObjectNode report = MAPPER.createObjectNode();
      report.put("schema", "jmlx-ffm-cal-raw-1");
      report.put("created_at", Instant.now().toString());
      report.put("java_version", System.getProperty("java.version"));
      report.put("os_name", System.getProperty("os.name"));
      report.put("os_version", System.getProperty("os.version"));
      report.put("os_arch", System.getProperty("os.arch"));
      report.put("dylib", dylib.toString());
      ObjectNode workload = report.putObject("workload");
      workload.put("iterations", iterations);
      workload.put("calls", calls);
      workload.put("warmup", warmup);
      ObjectNode shapes = report.putObject("shapes");
      for (int i = 0; i < KEYS.length; i++) {
        ObjectNode shape = shapes.putObject(KEYS[i]);
        shape.put("symbol", SYMBOLS[i]);
        ArrayNode nanos = shape.putArray("nanos");
        for (long sample : raw[i]) {
          nanos.add(sample);
        }
      }
      ArrayNode loopNanos = report.putObject("loop").putArray("nanos");
      for (long sample : loop) {
        loopNanos.add(sample);
      }
      MAPPER.writerWithDefaultPrettyPrinter().writeValue(output.toFile(), report);
    }
  }

  /** Keeps the measurement loop observable to the JIT (the native calls are never elided). */
  private static volatile long SINK;

  /**
   * The timing floor: {@code calls} no-op iterations. Each iteration is a linear congruential
   * recurrence ({@code sink = sink * C + (k + 1)}), so every iteration depends on the previous
   * one's result: C2 cannot process the iterations independently and the loop cannot be vectorized
   * (review round 2, F), while the per-iteration ALU work stays in the same order as the C no-op
   * bodies. The result lands in the volatile sink like the batch results.
   */
  private static long floor(int calls) {
    long sink = 0;
    for (int k = 0; k < calls; k++) {
      sink = sink * 0x9E3779B97F4A7C15L + (k + 1L);
    }
    return sink;
  }

  /**
   * The struct-return result allocator for one sample (review round 2, A, re-scoped in review round
   * 3, 1). It mirrors a production step scope's lifecycle: a per-scope shared arena plus a
   * per-scope {@code LinkedHashSet} tracking set (the production per-call cost of {@code
   * MLXScope.Holder.allocate}, not a bare confined arena's), created before the sample's timed
   * batch and closed after it, exactly as production closes a step scope after every step. With a
   * fork-lifetime scope the defaults would have kept the measured 4 x iterations x calls = 1.6M
   * segments in one set and one arena for the whole fork — the warmup rounds add one more per
   * struct shape apiece — so the set's doubling rehashes would have run inside the later samples'
   * timed batches and the growing live heap could have put GC pauses into any shape's samples,
   * biasing the p90 tails the calibration reads.
   */
  private static final class CalScope implements SegmentAllocator {
    private final Arena arena = Arena.ofShared();
    private final LinkedHashSet<MemorySegment> handles = new LinkedHashSet<>();

    @Override
    public MemorySegment allocate(long byteSize, long byteAlignment) {
      synchronized (this) {
        MemorySegment seg = arena.allocate(byteSize, byteAlignment);
        handles.add(seg);
        return seg;
      }
    }

    /**
     * Releases the sample's arena after its batch; the carriers are no-op structs, so there is no
     * native array to free (unlike a production scope close's per-array frees).
     */
    void close() {
      synchronized (this) {
        arena.close();
      }
    }
  }

  /**
   * One measured batch: {@code calls} consecutive downcalls of the shape with the given batch id.
   * Each call site below has exactly the generated wrapper's invocation form ({@code invokeExact}
   * on the holder's {@code static final} downcall handle with a matching signature).
   */
  static long batch(int id, SegmentAllocator allocator, int calls) {
    long sink = 0;
    try {
      for (int k = 0; k < calls; k++) {
        switch (id) {
          case 0:
            sink ^= call0(CARRIERS[0]);
            break;
          case 1:
            sink ^= call1(CARRIERS[0]);
            break;
          case 2:
            sink ^= call2(CARRIERS[0]).address();
            break;
          case 3:
            sink ^= call3(CARRIERS[0], CARRIERS[1]);
            break;
          case 4:
            sink ^= call4(CARRIERS[0], CARRIERS[1], CARRIERS[2]);
            break;
          case 5:
            sink ^= call5(CARRIERS[0], CARRIERS[1], CARRIERS[2], CARRIERS[3]);
            break;
          case 6:
            sink ^= call6(CARRIERS[0], CARRIERS[1], CARRIERS[2], CARRIERS[3], CARRIERS[4]);
            break;
          case 7:
            sink ^=
                call7(
                    CARRIERS[0],
                    CARRIERS[1],
                    CARRIERS[2],
                    CARRIERS[3],
                    CARRIERS[4],
                    BOOL_ARG,
                    CARRIERS[5]);
            break;
          case 8:
            sink ^=
                call8(
                    CARRIERS[0],
                    CARRIERS[1],
                    CARRIERS[2],
                    CARRIERS[3],
                    CARRIERS[4],
                    BOOL_ARG,
                    CARRIERS[5],
                    CARRIERS[6],
                    CARRIERS[7],
                    CARRIERS[8]);
            break;
          case 9:
            sink ^=
                call9(
                    CARRIERS[0],
                    CARRIERS[1],
                    CARRIERS[2],
                    CARRIERS[3],
                    CARRIERS[4],
                    CARRIERS[5],
                    CARRIERS[6],
                    CARRIERS[7],
                    CARRIERS[8],
                    CARRIERS[9]);
            break;
          case 10:
            sink ^= call10(CARRIERS[0], CARRIERS[1], BOOL_ARG, CARRIERS[2]);
            break;
          case 11:
            sink ^= call11(CARRIERS[0], CARRIERS[1], INT_ARG, BOOL_ARG, CARRIERS[2]);
            break;
          case 12:
            sink ^=
                call12(
                    CARRIERS[0],
                    CARRIERS[1],
                    INT_ARG,
                    BOOL_ARG,
                    CARRIERS[2],
                    FLOAT_ARG,
                    INT_ARG,
                    CARRIERS[3],
                    CARRIERS[4]);
            break;
          case 13:
            sink ^= call13(CARRIERS[0], CARRIERS[1], INT_ARG, INT_ARG, CARRIERS[2]);
            break;
          case 14:
            sink ^= call14(CARRIERS[0], CARRIERS[1], INT_ARG, CARRIERS[2]);
            break;
          case 15:
            sink ^= call15(CARRIERS[0], CARRIERS[1], CARRIERS[2], INT_ARG, CARRIERS[3]);
            break;
          case 16:
            sink ^= call16(CARRIERS[0], CARRIERS[1], CARRIERS[2], FLOAT_ARG, CARRIERS[3]);
            break;
          case 17:
            sink ^=
                call17(
                    CARRIERS[0],
                    CARRIERS[1],
                    CARRIERS[2],
                    CARRIERS[3],
                    FLOAT_ARG,
                    CARRIERS[4],
                    CARRIERS[5],
                    CARRIERS[6],
                    BOOL_ARG,
                    CARRIERS[7]);
            break;
          case 18:
            sink ^= call18(CARRIERS[0], CARRIERS[1], CARRIERS[2], LONG_ARG, CARRIERS[3]);
            break;
          case 19:
            sink ^= call19(CARRIERS[0], CARRIERS[1], CARRIERS[2], LONG_ARG, BOOL_ARG, CARRIERS[3]);
            break;
          case 20:
            sink ^=
                call20(
                    CARRIERS[0],
                    CARRIERS[1],
                    CARRIERS[2],
                    LONG_ARG,
                    CARRIERS[3],
                    LONG_ARG,
                    CARRIERS[4],
                    LONG_ARG,
                    CARRIERS[5]);
            break;
          case 21:
            sink ^= call21(CARRIERS[0], DOUBLE_ARG, DOUBLE_ARG, DOUBLE_ARG, INT_ARG, CARRIERS[1]);
            break;
          case 22:
            sink ^= call22(CARRIERS[0], CARRIERS[1], LONG_ARG, CARRIERS[2], INT_ARG, CARRIERS[3]);
            break;
          case 23:
            sink ^= call23(allocator, FLOAT_ARG).address();
            break;
          case 24:
            sink ^= call24(allocator).address();
            break;
          case 25:
            sink ^= call25(allocator, CARRIERS[0], LONG_ARG).address();
            break;
          default:
            sink ^= call26(allocator, CARRIERS[0], CARRIERS[1], INT_ARG, INT_ARG).address();
            break;
        }
      }
    } catch (Throwable t) {
      throw new IllegalStateException("calibration downcall failed", t);
    }
    return sink;
  }

  private static int call0(MemorySegment a) throws Throwable {
    return (int) Handle0.HANDLE.invokeExact(a);
  }

  private static long call1(MemorySegment a) throws Throwable {
    return (long) Handle1.HANDLE.invokeExact(a);
  }

  private static MemorySegment call2(MemorySegment a) throws Throwable {
    return (MemorySegment) Handle2.HANDLE.invokeExact(a);
  }

  private static int call3(MemorySegment a, MemorySegment b) throws Throwable {
    return (int) Handle3.HANDLE.invokeExact(a, b);
  }

  private static int call4(MemorySegment a, MemorySegment b, MemorySegment c) throws Throwable {
    return (int) Handle4.HANDLE.invokeExact(a, b, c);
  }

  private static int call5(MemorySegment a, MemorySegment b, MemorySegment c, MemorySegment d)
      throws Throwable {
    return (int) Handle5.HANDLE.invokeExact(a, b, c, d);
  }

  private static int call6(
      MemorySegment a, MemorySegment b, MemorySegment c, MemorySegment d, MemorySegment e)
      throws Throwable {
    return (int) Handle6.HANDLE.invokeExact(a, b, c, d, e);
  }

  private static int call7(
      MemorySegment a,
      MemorySegment b,
      MemorySegment c,
      MemorySegment d,
      MemorySegment e,
      boolean z,
      MemorySegment f)
      throws Throwable {
    return (int) Handle7.HANDLE.invokeExact(a, b, c, d, e, z, f);
  }

  private static int call8(
      MemorySegment a,
      MemorySegment b,
      MemorySegment c,
      MemorySegment d,
      MemorySegment e,
      boolean z,
      MemorySegment f,
      MemorySegment g,
      MemorySegment h,
      MemorySegment i)
      throws Throwable {
    return (int) Handle8.HANDLE.invokeExact(a, b, c, d, e, z, f, g, h, i);
  }

  private static int call9(
      MemorySegment a,
      MemorySegment b,
      MemorySegment c,
      MemorySegment d,
      MemorySegment e,
      MemorySegment f,
      MemorySegment g,
      MemorySegment h,
      MemorySegment i,
      MemorySegment j)
      throws Throwable {
    return (int) Handle9.HANDLE.invokeExact(a, b, c, d, e, f, g, h, i, j);
  }

  private static int call10(MemorySegment a, MemorySegment b, boolean z, MemorySegment c)
      throws Throwable {
    return (int) Handle10.HANDLE.invokeExact(a, b, z, c);
  }

  private static int call11(MemorySegment a, MemorySegment b, int i, boolean z, MemorySegment c)
      throws Throwable {
    return (int) Handle11.HANDLE.invokeExact(a, b, i, z, c);
  }

  private static int call12(
      MemorySegment a,
      MemorySegment b,
      int i,
      boolean z,
      MemorySegment c,
      float f,
      int j,
      MemorySegment d,
      MemorySegment e)
      throws Throwable {
    return (int) Handle12.HANDLE.invokeExact(a, b, i, z, c, f, j, d, e);
  }

  private static int call13(MemorySegment a, MemorySegment b, int i, int j, MemorySegment c)
      throws Throwable {
    return (int) Handle13.HANDLE.invokeExact(a, b, i, j, c);
  }

  private static int call14(MemorySegment a, MemorySegment b, int i, MemorySegment c)
      throws Throwable {
    return (int) Handle14.HANDLE.invokeExact(a, b, i, c);
  }

  private static int call15(
      MemorySegment a, MemorySegment b, MemorySegment c, int i, MemorySegment d) throws Throwable {
    return (int) Handle15.HANDLE.invokeExact(a, b, c, i, d);
  }

  private static int call16(
      MemorySegment a, MemorySegment b, MemorySegment c, float f, MemorySegment d)
      throws Throwable {
    return (int) Handle16.HANDLE.invokeExact(a, b, c, f, d);
  }

  private static int call17(
      MemorySegment a,
      MemorySegment b,
      MemorySegment c,
      MemorySegment d,
      float f,
      MemorySegment e,
      MemorySegment g,
      MemorySegment h,
      boolean z,
      MemorySegment i)
      throws Throwable {
    return (int) Handle17.HANDLE.invokeExact(a, b, c, d, f, e, g, h, z, i);
  }

  private static int call18(
      MemorySegment a, MemorySegment b, MemorySegment c, long j, MemorySegment d) throws Throwable {
    return (int) Handle18.HANDLE.invokeExact(a, b, c, j, d);
  }

  private static int call19(
      MemorySegment a, MemorySegment b, MemorySegment c, long j, boolean z, MemorySegment d)
      throws Throwable {
    return (int) Handle19.HANDLE.invokeExact(a, b, c, j, z, d);
  }

  private static int call20(
      MemorySegment a,
      MemorySegment b,
      MemorySegment c,
      long j,
      MemorySegment d,
      long k,
      MemorySegment e,
      long l,
      MemorySegment f)
      throws Throwable {
    return (int) Handle20.HANDLE.invokeExact(a, b, c, j, d, k, e, l, f);
  }

  private static int call21(
      MemorySegment a, double d1, double d2, double d3, int i, MemorySegment b) throws Throwable {
    return (int) Handle21.HANDLE.invokeExact(a, d1, d2, d3, i, b);
  }

  private static int call22(
      MemorySegment a, MemorySegment b, long j, MemorySegment c, int i, MemorySegment d)
      throws Throwable {
    return (int) Handle22.HANDLE.invokeExact(a, b, j, c, i, d);
  }

  private static MemorySegment call23(SegmentAllocator allocator, float f) throws Throwable {
    return (MemorySegment) Handle23.HANDLE.invokeExact(allocator, f);
  }

  private static MemorySegment call24(SegmentAllocator allocator) throws Throwable {
    return (MemorySegment) Handle24.HANDLE.invokeExact(allocator);
  }

  private static MemorySegment call25(SegmentAllocator allocator, MemorySegment a, long j)
      throws Throwable {
    return (MemorySegment) Handle25.HANDLE.invokeExact(allocator, a, j);
  }

  private static MemorySegment call26(
      SegmentAllocator allocator, MemorySegment a, MemorySegment b, int i, int j) throws Throwable {
    return (MemorySegment) Handle26.HANDLE.invokeExact(allocator, a, b, i, j);
  }

  // The 27 per-shape downcall holders: the jextract-generated wrappers' own form — a private
  // holder class binding its MethodHandle in <clinit> as a static final field. A call site
  // referencing HandleN.HANDLE can be constant-folded by C2 (a trusted final of a private
  // class); an array load of the handle could not, so the measurement would have included
  // address-indexed dispatch the generated wrappers do not pay.
  private static final class Handle0 {
    static final MethodHandle HANDLE = downcall(0);
  }

  private static final class Handle1 {
    static final MethodHandle HANDLE = downcall(1);
  }

  private static final class Handle2 {
    static final MethodHandle HANDLE = downcall(2);
  }

  private static final class Handle3 {
    static final MethodHandle HANDLE = downcall(3);
  }

  private static final class Handle4 {
    static final MethodHandle HANDLE = downcall(4);
  }

  private static final class Handle5 {
    static final MethodHandle HANDLE = downcall(5);
  }

  private static final class Handle6 {
    static final MethodHandle HANDLE = downcall(6);
  }

  private static final class Handle7 {
    static final MethodHandle HANDLE = downcall(7);
  }

  private static final class Handle8 {
    static final MethodHandle HANDLE = downcall(8);
  }

  private static final class Handle9 {
    static final MethodHandle HANDLE = downcall(9);
  }

  private static final class Handle10 {
    static final MethodHandle HANDLE = downcall(10);
  }

  private static final class Handle11 {
    static final MethodHandle HANDLE = downcall(11);
  }

  private static final class Handle12 {
    static final MethodHandle HANDLE = downcall(12);
  }

  private static final class Handle13 {
    static final MethodHandle HANDLE = downcall(13);
  }

  private static final class Handle14 {
    static final MethodHandle HANDLE = downcall(14);
  }

  private static final class Handle15 {
    static final MethodHandle HANDLE = downcall(15);
  }

  private static final class Handle16 {
    static final MethodHandle HANDLE = downcall(16);
  }

  private static final class Handle17 {
    static final MethodHandle HANDLE = downcall(17);
  }

  private static final class Handle18 {
    static final MethodHandle HANDLE = downcall(18);
  }

  private static final class Handle19 {
    static final MethodHandle HANDLE = downcall(19);
  }

  private static final class Handle20 {
    static final MethodHandle HANDLE = downcall(20);
  }

  private static final class Handle21 {
    static final MethodHandle HANDLE = downcall(21);
  }

  private static final class Handle22 {
    static final MethodHandle HANDLE = downcall(22);
  }

  private static final class Handle23 {
    static final MethodHandle HANDLE = downcall(23);
  }

  private static final class Handle24 {
    static final MethodHandle HANDLE = downcall(24);
  }

  private static final class Handle25 {
    static final MethodHandle HANDLE = downcall(25);
  }

  private static final class Handle26 {
    static final MethodHandle HANDLE = downcall(26);
  }
}
