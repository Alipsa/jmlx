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
 * src/main/resources/ffm_cal.c}. Every mlx-c opaque struct parameter is an 8-byte struct passed
 * <em>by value</em> in the pinned header (jextract maps it to the struct's 8-byte layout, not to a
 * pointer), so the M parameter slots are declared as an 8-byte struct layout — the AArch64 C ABI
 * carrier (one 64-bit register) is identical to a pointer's, and the Java-side marshalling (the
 * 8-byte struct read from the segment) matches the generated wrappers' exactly. One sample is the
 * total wall time of {@code calls} consecutive downcalls; an empty Java loop over the same call
 * count is measured as the timing floor (warmed up like the batches).
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
  private static final MemoryLayout STRUCT8 = MemoryLayout.structLayout(ValueLayout.ADDRESS);
  private static final int INT_ARG = 7;
  private static final long LONG_ARG = 4096;
  private static final boolean BOOL_ARG = true;

  /** The calibration shape keys, in batch id order. */
  static final String[] KEYS = {
    "(M)I",
    "(M)J",
    "(M)Z",
    "(M)V",
    "(M)M",
    "(AM)M",
    "(MM)I",
    "(MMM)I",
    "(MMMM)I",
    "(MMMMM)I",
    "(MI)I",
    "(MMIM)I",
    "(MMIIM)I",
    "(MZM)I",
    "(MMMJM)I",
    "(MMIMM)I"
  };

  /** The no-op C symbols of the calibration shapes, in batch id order. */
  static final String[] SYMBOLS = {
    "cal_M_I",
    "cal_M_J",
    "cal_M_Z",
    "cal_M_V",
    "cal_M_M",
    "cal_AM_M",
    "cal_MM_I",
    "cal_MMM_I",
    "cal_MMMM_I",
    "cal_MMMMM_I",
    "cal_MI_I",
    "cal_MMIM_I",
    "cal_MMIIM_I",
    "cal_MZM_I",
    "cal_MMMJM_I",
    "cal_MMIMM_I"
  };

  /**
   * The downcall descriptors, one per batch id; return layout first, then parameters. Every M
   * parameter slot is the 8-byte struct layout (by value, as the pinned header passes {@code
   * mlx_array} and friends — the generated wrappers declare the same layout); the M return slots
   * keep their own form (the pointer return of (M)M, the struct return of (AM)M).
   */
  private static final FunctionDescriptor[] DESCS = {
    FunctionDescriptor.of(ValueLayout.JAVA_INT, STRUCT8),
    FunctionDescriptor.of(ValueLayout.JAVA_LONG, STRUCT8),
    FunctionDescriptor.of(ValueLayout.JAVA_BOOLEAN, STRUCT8),
    FunctionDescriptor.ofVoid(STRUCT8),
    FunctionDescriptor.of(ValueLayout.ADDRESS, STRUCT8),
    FunctionDescriptor.of(STRUCT8, STRUCT8),
    FunctionDescriptor.of(ValueLayout.JAVA_INT, STRUCT8, STRUCT8),
    FunctionDescriptor.of(ValueLayout.JAVA_INT, STRUCT8, STRUCT8, STRUCT8),
    FunctionDescriptor.of(ValueLayout.JAVA_INT, STRUCT8, STRUCT8, STRUCT8, STRUCT8),
    FunctionDescriptor.of(ValueLayout.JAVA_INT, STRUCT8, STRUCT8, STRUCT8, STRUCT8, STRUCT8),
    FunctionDescriptor.of(ValueLayout.JAVA_INT, STRUCT8, ValueLayout.JAVA_INT),
    FunctionDescriptor.of(ValueLayout.JAVA_INT, STRUCT8, STRUCT8, ValueLayout.JAVA_INT, STRUCT8),
    FunctionDescriptor.of(
        ValueLayout.JAVA_INT,
        STRUCT8,
        STRUCT8,
        ValueLayout.JAVA_INT,
        ValueLayout.JAVA_INT,
        STRUCT8),
    FunctionDescriptor.of(ValueLayout.JAVA_INT, STRUCT8, ValueLayout.JAVA_BOOLEAN, STRUCT8),
    FunctionDescriptor.of(
        ValueLayout.JAVA_INT, STRUCT8, STRUCT8, STRUCT8, ValueLayout.JAVA_LONG, STRUCT8),
    FunctionDescriptor.of(
        ValueLayout.JAVA_INT, STRUCT8, STRUCT8, ValueLayout.JAVA_INT, STRUCT8, STRUCT8)
  };

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
            Handle15.HANDLE
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
    try (Arena arena = Arena.ofConfined()) {
      Init.lookup = SymbolLookup.libraryLookup(dylib, arena);
      initializeHandles();
      MemorySegment[] carriers = new MemorySegment[5];
      for (int i = 0; i < carriers.length; i++) {
        carriers[i] = arena.allocate(8);
      }
      SegmentAllocator resultAllocator = arena::allocate;

      long sink = 0;
      for (int w = 0; w < warmup; w++) {
        for (int i = 0; i < KEYS.length; i++) {
          sink ^= batch(i, carriers, resultAllocator, 1) ^ floor(1);
        }
      }
      SINK ^= sink;

      long[][] raw = new long[KEYS.length][iterations];
      long[] loop = new long[iterations];
      for (int i = 0; i < iterations; i++) {
        int offset = i % KEYS.length;
        for (int s = 0; s < KEYS.length; s++) {
          int id = (offset + s) % KEYS.length;
          long t0 = System.nanoTime();
          SINK ^= batch(id, carriers, resultAllocator, calls);
          raw[id][i] = System.nanoTime() - t0;
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
   * The timing floor: {@code calls} no-op iterations. Each iteration runs a per-iteration xor-fold
   * (two ALU ops — the same order of work the C no-op body does), so C2 has to execute every
   * iteration instead of folding an empty loop body away; the result lands in the volatile sink
   * like the batch results.
   */
  private static long floor(int calls) {
    long sink = 0;
    for (int k = 0; k < calls; k++) {
      sink ^= (k + 1L) * 0x9E3779B97F4A7C15L;
    }
    return sink;
  }

  /**
   * One measured batch: {@code calls} consecutive downcalls of the shape with the given batch id.
   * Each call site below has exactly the generated wrapper's invocation form ({@code invokeExact}
   * on the holder's {@code static final} downcall handle with a matching signature).
   */
  static long batch(int id, MemorySegment[] carriers, SegmentAllocator allocator, int calls) {
    long sink = 0;
    try {
      for (int k = 0; k < calls; k++) {
        switch (id) {
          case 0:
            sink ^= call0(carriers[0]);
            break;
          case 1:
            sink ^= call1(carriers[0]);
            break;
          case 2:
            sink ^= call2(carriers[0]) ? 1L : 0L;
            break;
          case 3:
            call3(carriers[0]);
            break;
          case 4:
            sink ^= call4(carriers[0]).address();
            break;
          case 5:
            sink ^= call5(allocator, carriers[0]).address();
            break;
          case 6:
            sink ^= call6(carriers[0], carriers[1]);
            break;
          case 7:
            sink ^= call7(carriers[0], carriers[1], carriers[2]);
            break;
          case 8:
            sink ^= call8(carriers[0], carriers[1], carriers[2], carriers[3]);
            break;
          case 9:
            sink ^= call9(carriers[0], carriers[1], carriers[2], carriers[3], carriers[4]);
            break;
          case 10:
            sink ^= call10(carriers[0], INT_ARG);
            break;
          case 11:
            sink ^= call11(carriers[0], carriers[1], INT_ARG, carriers[2]);
            break;
          case 12:
            sink ^= call12(carriers[0], carriers[1], INT_ARG, INT_ARG, carriers[2]);
            break;
          case 13:
            sink ^= call13(carriers[0], BOOL_ARG, carriers[1]);
            break;
          case 14:
            sink ^= call14(carriers[0], carriers[1], carriers[2], LONG_ARG, carriers[3]);
            break;
          default:
            sink ^= call15(carriers[0], carriers[1], INT_ARG, carriers[2], carriers[3]);
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

  private static boolean call2(MemorySegment a) throws Throwable {
    return (boolean) Handle2.HANDLE.invokeExact(a);
  }

  private static void call3(MemorySegment a) throws Throwable {
    Handle3.HANDLE.invokeExact(a);
  }

  private static MemorySegment call4(MemorySegment a) throws Throwable {
    return (MemorySegment) Handle4.HANDLE.invokeExact(a);
  }

  private static MemorySegment call5(SegmentAllocator allocator, MemorySegment a) throws Throwable {
    return (MemorySegment) Handle5.HANDLE.invokeExact(allocator, a);
  }

  private static int call6(MemorySegment a, MemorySegment b) throws Throwable {
    return (int) Handle6.HANDLE.invokeExact(a, b);
  }

  private static int call7(MemorySegment a, MemorySegment b, MemorySegment c) throws Throwable {
    return (int) Handle7.HANDLE.invokeExact(a, b, c);
  }

  private static int call8(MemorySegment a, MemorySegment b, MemorySegment c, MemorySegment d)
      throws Throwable {
    return (int) Handle8.HANDLE.invokeExact(a, b, c, d);
  }

  private static int call9(
      MemorySegment a, MemorySegment b, MemorySegment c, MemorySegment d, MemorySegment e)
      throws Throwable {
    return (int) Handle9.HANDLE.invokeExact(a, b, c, d, e);
  }

  private static int call10(MemorySegment a, int i) throws Throwable {
    return (int) Handle10.HANDLE.invokeExact(a, i);
  }

  private static int call11(MemorySegment a, MemorySegment b, int i, MemorySegment c)
      throws Throwable {
    return (int) Handle11.HANDLE.invokeExact(a, b, i, c);
  }

  private static int call12(MemorySegment a, MemorySegment b, int i, int j, MemorySegment c)
      throws Throwable {
    return (int) Handle12.HANDLE.invokeExact(a, b, i, j, c);
  }

  private static int call13(MemorySegment a, boolean z, MemorySegment b) throws Throwable {
    return (int) Handle13.HANDLE.invokeExact(a, z, b);
  }

  private static int call14(
      MemorySegment a, MemorySegment b, MemorySegment c, long j, MemorySegment d) throws Throwable {
    return (int) Handle14.HANDLE.invokeExact(a, b, c, j, d);
  }

  private static int call15(
      MemorySegment a, MemorySegment b, int i, MemorySegment c, MemorySegment d) throws Throwable {
    return (int) Handle15.HANDLE.invokeExact(a, b, i, c, d);
  }

  // The 16 per-shape downcall holders: the jextract-generated wrappers' own form — a private
  // holder class binding its MethodHandle in <clinit> as a static final field. A call site
  // referencing HandleN.HANDLE can be constant-folded by C2 (a trusted final of a private
  // class); the previous MethodHandle[] array load could not, so the measurement would have
  // included address-indexed dispatch the generated wrappers do not pay.
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
}
