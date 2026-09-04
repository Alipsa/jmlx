package se.alipsa.jmlx.ffi;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.net.URLConnection;
import java.net.URLStreamHandler;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.FileTime;
import java.time.Duration;
import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.api.parallel.ResourceLock;
import org.junit.jupiter.api.parallel.Resources;

/**
 * Exercises {@link ClasspathNativeExtractor}'s extraction mechanics using tiny fake fixture files
 * under {@code src/test/resources/se/alipsa/jmlx/native-fixture/} -- a deliberately different
 * resource root than the real {@code se/alipsa/jmlx/native/macos-aarch64} one, so these fixtures
 * can never collide with (or be mistaken for) the real ~180MB bundled binaries. Real end-to-end
 * extraction of the genuine artifact is covered separately by the {@code extractionFallbackTest}
 * Gradle task (see {@code jmlx-ffi/build.gradle}), on real hardware.
 */
class ClasspathNativeExtractorTest {

  private static final String FIXTURE_ROOT = "se/alipsa/jmlx/native-fixture/macos-aarch64";
  private static final String ALT_FIXTURE_ROOT = "se/alipsa/jmlx/native-fixture/macos-aarch64-alt";
  private static final String SIZE_FIXTURE_ROOT =
      "se/alipsa/jmlx/native-fixture/macos-aarch64-size";
  private static final String MISSING_PIN_FIXTURE_ROOT =
      "se/alipsa/jmlx/native-fixture/macos-aarch64-missing-pin";
  private static final List<String> BINARY_NAMES =
      List.of("libmlxc.dylib", "libmlx.dylib", "libjaccl.dylib", "mlx.metallib");

  private static ClassLoader loader() {
    return ClasspathNativeExtractorTest.class.getClassLoader();
  }

  private static ClassLoader loaderWithUnknownContentLengths() {
    return new ClassLoader(loader()) {
      @Override
      public URL getResource(String name) {
        URL delegate = super.getResource(name);
        if (delegate == null || !name.startsWith(FIXTURE_ROOT)) {
          return delegate;
        }
        try {
          return new URL(
              null,
              "fixture:" + delegate,
              new URLStreamHandler() {
                @Override
                protected URLConnection openConnection(URL ignored) {
                  return new URLConnection(ignored) {
                    @Override
                    public void connect() {}

                    @Override
                    public InputStream getInputStream() throws IOException {
                      return delegate.openStream();
                    }

                    @Override
                    public long getContentLengthLong() {
                      return -1;
                    }
                  };
                }
              });
        } catch (IOException e) {
          throw new AssertionError(e);
        }
      }
    };
  }

  @Test
  void returnsEmptyWhenTheResourceRootIsNotOnTheClasspath(@TempDir Path cacheRoot) {
    Optional<Path> result =
        ClasspathNativeExtractor.extractIfAvailable(
            loader(), "se/alipsa/jmlx/native-fixture/does-not-exist", cacheRoot);
    assertTrue(result.isEmpty());
  }

  @Test
  void reportsNativeArtifactWithoutPayload(@TempDir Path cacheRoot) {
    ClassLoader nativeArtifactOnlyLoader =
        new ClassLoader(null) {
          @Override
          public URL getResource(String name) {
            if (name.equals("se/alipsa/jmlx/nativelib/macosarm64/NativeArtifact.class")) {
              return ClasspathNativeExtractorTest.class.getResource(
                  "ClasspathNativeExtractorTest.class");
            }
            return null;
          }
        };

    ClasspathNativeExtractor.NativeExtractionException failure =
        assertThrows(
            ClasspathNativeExtractor.NativeExtractionException.class,
            () ->
                ClasspathNativeExtractor.extractIfAvailable(
                    nativeArtifactOnlyLoader, "se/alipsa/jmlx/native/macos-aarch64", cacheRoot));

    assertTrue(failure.getMessage().contains("contains no native payload"));
  }

  private static final Map<String, String> EXPECTED_FIXTURE_CONTENT =
      Map.of(
          "libmlxc.dylib", "FAKE-TEST-FIXTURE-NOT-REAL-MLX-libmlxc",
          "libmlx.dylib", "FAKE-TEST-FIXTURE-NOT-REAL-MLX-libmlx",
          "libjaccl.dylib", "FAKE-TEST-FIXTURE-NOT-REAL-MLX-libjaccl",
          "mlx.metallib", "FAKE-TEST-FIXTURE-NOT-REAL-MLX-metallib");

  @Test
  void extractsAllFourBinariesFlatIntoOneDirectory(@TempDir Path cacheRoot) throws IOException {
    Path dir =
        ClasspathNativeExtractor.extractIfAvailable(loader(), FIXTURE_ROOT, cacheRoot)
            .orElseThrow();
    for (String name : BINARY_NAMES) {
      assertEquals(EXPECTED_FIXTURE_CONTENT.get(name), Files.readString(dir.resolve(name)));
    }
  }

  @Test
  void extractsWhenAReadableUrlDoesNotReportItsContentLength(@TempDir Path cacheRoot)
      throws IOException {
    Path dir =
        ClasspathNativeExtractor.extractIfAvailable(
                loaderWithUnknownContentLengths(), FIXTURE_ROOT, cacheRoot)
            .orElseThrow();

    assertEquals(
        EXPECTED_FIXTURE_CONTENT.get("mlx.metallib"),
        Files.readString(dir.resolve("mlx.metallib")));
  }

  @Test
  void secondCallIsIdempotentAndDoesNotRewriteFiles(@TempDir Path cacheRoot) throws IOException {
    Path first =
        ClasspathNativeExtractor.extractIfAvailable(loader(), FIXTURE_ROOT, cacheRoot)
            .orElseThrow();
    long mtimeBefore = Files.getLastModifiedTime(first.resolve("mlx.metallib")).toMillis();

    Path second =
        ClasspathNativeExtractor.extractIfAvailable(loader(), FIXTURE_ROOT, cacheRoot)
            .orElseThrow();
    long mtimeAfter = Files.getLastModifiedTime(second.resolve("mlx.metallib")).toMillis();

    assertEquals(first, second);
    assertEquals(
        mtimeBefore, mtimeAfter, "second call must not rewrite an already-complete cache entry");
  }

  @Test
  void concurrentExtractionRacesConvergeOnOneUncorruptedDirectory(@TempDir Path cacheRoot)
      throws Exception {
    int threadCount = 8;
    ExecutorService pool = Executors.newFixedThreadPool(threadCount);
    try {
      Callable<Path> task =
          () ->
              ClasspathNativeExtractor.extractIfAvailable(loader(), FIXTURE_ROOT, cacheRoot)
                  .orElseThrow();
      List<Future<Path>> futures = pool.invokeAll(java.util.Collections.nCopies(threadCount, task));
      List<Path> results = new java.util.ArrayList<>();
      for (Future<Path> f : futures) {
        results.add(f.get());
      }
      assertEquals(
          1, results.stream().distinct().count(), "every racing call must agree on one path");

      Path dir = results.get(0);
      for (String name : BINARY_NAMES) {
        Path file = dir.resolve(name);
        assertTrue(Files.isRegularFile(file), "missing " + file);
        assertTrue(Files.size(file) > 0, "empty " + file);
      }
      // No leftover .tmp-* siblings from a losing race.
      try (var entries = Files.list(cacheRoot)) {
        List<Path> leftovers =
            entries
                .filter(p -> p.getFileName().toString().startsWith(".tmp-"))
                .collect(Collectors.toList());
        assertTrue(leftovers.isEmpty(), "leftover temp directories: " + leftovers);
      }
    } finally {
      pool.shutdown();
      assertTrue(pool.awaitTermination(10, TimeUnit.SECONDS));
    }
  }

  @Test
  void differentPinContentExtractsToADifferentCacheSubdirectory(@TempDir Path cacheRoot)
      throws IOException {
    Path a =
        ClasspathNativeExtractor.extractIfAvailable(loader(), FIXTURE_ROOT, cacheRoot)
            .orElseThrow();
    Path b =
        ClasspathNativeExtractor.extractIfAvailable(loader(), ALT_FIXTURE_ROOT, cacheRoot)
            .orElseThrow();
    assertFalse(
        a.equals(b), "different native-pin.properties content must yield different cache keys");
    assertTrue(Files.isDirectory(b));
  }

  @Test
  void samePinWithDifferentBinarySizesExtractsToADifferentCacheSubdirectory(@TempDir Path cacheRoot)
      throws IOException {
    Path a =
        ClasspathNativeExtractor.extractIfAvailable(loader(), FIXTURE_ROOT, cacheRoot)
            .orElseThrow();
    Path b =
        ClasspathNativeExtractor.extractIfAvailable(loader(), SIZE_FIXTURE_ROOT, cacheRoot)
            .orElseThrow();
    assertFalse(a.equals(b), "different binary sizes must affect the cache key");
    assertTrue(Files.isDirectory(b));
  }

  @Test
  void truncatedCacheFileIsReplaced(@TempDir Path cacheRoot) throws IOException {
    Path extracted =
        ClasspathNativeExtractor.extractIfAvailable(loader(), FIXTURE_ROOT, cacheRoot)
            .orElseThrow();
    Files.writeString(extracted.resolve("libmlxc.dylib"), "truncated");

    Path repaired =
        ClasspathNativeExtractor.extractIfAvailable(loader(), FIXTURE_ROOT, cacheRoot)
            .orElseThrow();

    assertEquals(extracted, repaired);
    assertEquals(
        EXPECTED_FIXTURE_CONTENT.get("libmlxc.dylib"),
        Files.readString(repaired.resolve("libmlxc.dylib")));
  }

  @Test
  void repairSweepsStaleRootStagingAndLeavesOnlyNativeFiles(@TempDir Path cacheRoot)
      throws IOException {
    Path extracted =
        ClasspathNativeExtractor.extractIfAvailable(loader(), FIXTURE_ROOT, cacheRoot)
            .orElseThrow();
    Path staleStaging = cacheRoot.resolve(".tmp-repair-orphan");
    Files.writeString(staleStaging, "abandoned repair staging");
    Files.setLastModifiedTime(
        staleStaging, FileTime.from(Instant.now().minus(Duration.ofHours(2))));
    Files.writeString(extracted.resolve("libmlxc.dylib"), "truncated");

    Path repaired =
        ClasspathNativeExtractor.extractIfAvailable(loader(), FIXTURE_ROOT, cacheRoot)
            .orElseThrow();

    assertFalse(Files.exists(staleStaging), "stale root staging was not swept");
    try (var entries = Files.list(repaired)) {
      assertEquals(
          Set.of(
              "libmlxc.dylib",
              "libmlx.dylib",
              "libjaccl.dylib",
              "mlx.metallib",
              "native-pin.properties"),
          entries.map(path -> path.getFileName().toString()).collect(Collectors.toSet()));
    }
  }

  @Test
  void interruptedRepairStagesInTheCacheRoot(@TempDir Path cacheRoot) throws IOException {
    Path extracted =
        ClasspathNativeExtractor.extractIfAvailable(loader(), FIXTURE_ROOT, cacheRoot)
            .orElseThrow();
    Files.writeString(extracted.resolve("libmlxc.dylib"), "truncated");
    AtomicReference<Path> staged = new AtomicReference<>();

    assertThrows(
        IllegalStateException.class,
        () ->
            ClasspathNativeExtractor.extractIfAvailable(
                loader(),
                FIXTURE_ROOT,
                cacheRoot,
                path -> {
                  staged.set(path);
                  throw new IllegalStateException("simulate an interrupted repair");
                }));

    assertEquals(cacheRoot, staged.get().getParent());
    assertFalse(staged.get().startsWith(extracted));
  }

  @Test
  @ResourceLock(Resources.SYSTEM_PROPERTIES)
  void validatesAndBoundsConfiguredLockTimeout(@TempDir Path cacheRoot) {
    String property = "jmlx.native.lock.timeout.seconds";
    String previous = System.getProperty(property);
    try {
      System.setProperty(property, "9223372036854775807");
      assertTrue(
          ClasspathNativeExtractor.extractIfAvailable(loader(), FIXTURE_ROOT, cacheRoot)
              .isPresent());

      System.setProperty(property, "not-a-duration");
      ClasspathNativeExtractor.NativeExtractionException failure =
          assertThrows(
              ClasspathNativeExtractor.NativeExtractionException.class,
              () -> ClasspathNativeExtractor.extractIfAvailable(loader(), FIXTURE_ROOT, cacheRoot));
      assertTrue(failure.getMessage().contains(property));

      System.setProperty(property, "0");
      ClasspathNativeExtractor.NativeExtractionException nonPositiveFailure =
          assertThrows(
              ClasspathNativeExtractor.NativeExtractionException.class,
              () -> ClasspathNativeExtractor.extractIfAvailable(loader(), FIXTURE_ROOT, cacheRoot));
      assertTrue(nonPositiveFailure.getMessage().contains(property));
    } finally {
      if (previous == null) {
        System.clearProperty(property);
      } else {
        System.setProperty(property, previous);
      }
    }
  }

  @Test
  void nonDirectoryCacheTargetIsReplaced(@TempDir Path cacheRoot) throws IOException {
    Path extracted =
        ClasspathNativeExtractor.extractIfAvailable(loader(), FIXTURE_ROOT, cacheRoot)
            .orElseThrow();
    Files.delete(extracted.resolve("libmlxc.dylib"));
    Files.delete(extracted.resolve("libmlx.dylib"));
    Files.delete(extracted.resolve("libjaccl.dylib"));
    Files.delete(extracted.resolve("mlx.metallib"));
    Files.delete(extracted.resolve("native-pin.properties"));
    Files.delete(extracted);
    Files.writeString(extracted, "stray cache entry");

    Path repaired =
        ClasspathNativeExtractor.extractIfAvailable(loader(), FIXTURE_ROOT, cacheRoot)
            .orElseThrow();

    assertEquals(extracted, repaired);
    assertTrue(Files.isDirectory(repaired));
    assertEquals(
        EXPECTED_FIXTURE_CONTENT.get("libmlxc.dylib"),
        Files.readString(repaired.resolve("libmlxc.dylib")));
  }

  @Test
  void malformedNativeArtifactFailureIsReported(@TempDir Path cacheRoot) {
    assertThrows(
        ClasspathNativeExtractor.NativeExtractionException.class,
        () ->
            ClasspathNativeExtractor.extractIfAvailable(
                loader(), MISSING_PIN_FIXTURE_ROOT, cacheRoot));
  }

  @Test
  void isSupportedPlatformOnlyAcceptsMacosAarch64() {
    assertTrue(ClasspathNativeExtractor.isSupportedPlatform("Mac OS X", "aarch64"));
    assertFalse(ClasspathNativeExtractor.isSupportedPlatform("Mac OS X", "x86_64"));
    assertFalse(ClasspathNativeExtractor.isSupportedPlatform("Linux", "aarch64"));
    assertFalse(ClasspathNativeExtractor.isSupportedPlatform("Windows 11", "amd64"));
  }
}
