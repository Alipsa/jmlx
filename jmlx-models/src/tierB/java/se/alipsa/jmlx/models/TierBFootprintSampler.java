package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;

/**
 * Samples the test JVM's macOS physical footprint every 20 ms while inference runs; the sampler
 * thread never calls MLX.
 *
 * <p>{@code ps -o rss} counts only the process's private resident pages and misses memory the MLX
 * allocator and Metal hold for the process (the recorded Tier-B RSS sampled below the checkpoint's
 * own weight size), so it is not useful memory evidence. {@code ri_phys_footprint} from libproc's
 * {@code proc_pid_rusage(RUSAGE_INFO_V2)} is the figure Activity Monitor and {@code top} report as
 * the process's memory: it covers the JVM's private and shared pages plus file-backed mappings, so
 * it is what the hosted runner's RAM must actually hold.
 *
 * <p>Binding: {@code /usr/lib/libproc.dylib} is loaded with {@link System#load} and resolved
 * through {@link SymbolLookup#loaderLookup()}, the same mechanism the committed jextract bindings
 * use for libmlxc. In the macOS SDK's {@code struct rusage_info_v2} (sys/resource.h) {@code
 * ri_phys_footprint} sits at byte 72 -- the 16-byte {@code ri_uuid} plus the seven 8-byte fields
 * {@code ri_user_time} through {@code ri_resident_size} precede it -- verified against the SDK
 * header's layout on this machine. The buffer over-allocates the whole struct, and each sample is
 * rejected below a plausibility floor so a layout shift that moves or zeroes the field fails the
 * gate instead of passing it with a tiny reading.
 */
final class TierBFootprintSampler implements AutoCloseable {
  // proc_pid_rusage(int pid, int flavor, void *buffer); RUSAGE_INFO_V2 = 2.
  private static final int FLAVOR_RUSAGE_INFO_V2 = 2;
  // struct rusage_info_v2 in sys/resource.h: ri_phys_footprint at byte 72.
  private static final int PHYS_FOOTPRINT_OFFSET = 72;
  // A running test JVM always holds tens of MiB of footprint; a smaller reading means the
  // field was misparsed, not that the process shrank.
  private static final long MINIMUM_PLAUSIBLE_BYTES = 64L * 1024 * 1024;

  // libproc must be loaded before its symbol is resolved; both run once at class
  // initialization, before any sample.
  private static final MethodHandle PROC_PID_RUSAGE = bindProcPidRusage();

  private static MethodHandle bindProcPidRusage() {
    System.load("/usr/lib/libproc.dylib");
    return Linker.nativeLinker()
        .downcallHandle(
            SymbolLookup.loaderLookup()
                .or(Linker.nativeLinker().defaultLookup())
                .findOrThrow("proc_pid_rusage"),
            FunctionDescriptor.of(
                ValueLayout.JAVA_INT,
                ValueLayout.JAVA_INT,
                ValueLayout.JAVA_INT,
                ValueLayout.ADDRESS));
  }

  private volatile boolean running = true;
  private volatile long maximum;
  private volatile Exception failure;
  private final Thread worker;

  TierBFootprintSampler() {
    worker =
        new Thread(
            () -> {
              try {
                while (running) {
                  sample();
                  Thread.sleep(20);
                }
              } catch (Exception exception) {
                failure = exception;
              }
            },
            "tier-b-footprint-sampler");
    worker.start();
  }

  long maximum() throws InterruptedException {
    sample();
    return maximum;
  }

  private synchronized void sample() throws InterruptedException {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment buffer = arena.allocate(256, 8);
      int status;
      try {
        status =
            (int)
                PROC_PID_RUSAGE.invoke(
                    (int) ProcessHandle.current().pid(), FLAVOR_RUSAGE_INFO_V2, buffer);
      } catch (InterruptedException exception) {
        throw exception;
      } catch (Throwable exception) {
        throw new IllegalStateException("proc_pid_rusage downcall failed", exception);
      }
      if (status != 0) {
        throw new IllegalStateException("proc_pid_rusage failed with status " + status);
      }
      long footprint = buffer.get(ValueLayout.JAVA_LONG, PHYS_FOOTPRINT_OFFSET);
      if (footprint < MINIMUM_PLAUSIBLE_BYTES) {
        throw new IllegalStateException(
            "implausible phys_footprint "
                + footprint
                + " bytes at offset "
                + PHYS_FOOTPRINT_OFFSET);
      }
      maximum = Math.max(maximum, footprint);
    }
  }

  @Override
  public void close() throws InterruptedException {
    running = false;
    worker.join();
    assertEquals(null, failure, "footprint sampler failure");
    assertTrue(maximum >= MINIMUM_PLAUSIBLE_BYTES);
  }
}
