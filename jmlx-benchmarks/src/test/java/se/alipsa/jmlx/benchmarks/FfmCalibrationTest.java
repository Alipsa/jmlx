package se.alipsa.jmlx.benchmarks;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;
import tools.jackson.databind.node.ObjectNode;

/**
 * Exercises the FFM calibration end-to-end (source extraction, {@code cc} build, in-process worker
 * run, aggregation). Skipped when {@code cc} is not on the PATH: the calibration is an opt-in local
 * tool, and ordinary {@code check} must gain no compiler prerequisite.
 */
class FfmCalibrationTest {
  private static final ObjectMapper MAPPER = new ObjectMapper();

  @TempDir Path dir;

  @Test
  void calibrationMeasuresDowncallShapes() throws Exception {
    assumeTrue(ccAvailable(), "cc is not on the PATH; the FFM calibration is opt-in");
    Path source = FfmCalibration.extractSource(dir);
    Path dylib = dir.resolve("libffm_cal.dylib");
    FfmCalibration.build(dylib, source);
    Path rawFile = dir.resolve("raw.json");
    FfmCalWorker.run(dylib, rawFile, 20, 400, 100);
    JsonNode raw = MAPPER.readTree(rawFile.toFile());
    assertEquals("jmlx-ffm-cal-raw-1", raw.path("schema").asString());
    for (String key : FfmCalWorker.KEYS) {
      JsonNode samples = raw.path("shapes").path(key).path("nanos");
      assertEquals(20, samples.size(), key);
      for (JsonNode value : samples) {
        assertTrue(value.asDouble() > 0, key);
      }
    }
    ObjectNode buildInfo = FfmCalibration.buildInfo(dir, source, dylib);
    assertEquals(64, buildInfo.path("source_sha256").asString("").length(), "source sha");
    assertEquals(64, buildInfo.path("binary_sha256").asString("").length(), "binary sha");
    assertEquals("cc", buildInfo.path("command").get(0).asString());
    ObjectNode summary = FfmCalibration.aggregate(List.of(raw), buildInfo, 20, 400, 1, 100);
    assertEquals("jmlx-ffm-cal-1", summary.path("schema").asString());
    double loopMedian = summary.path("loop").path("median_ns").asDouble();
    assertTrue(loopMedian > 0);
    for (String key : FfmCalWorker.KEYS) {
      JsonNode shape = summary.path("shapes").path(key);
      assertEquals(20, shape.path("n").asInt(), key);
      assertTrue(shape.path("p10_ns").asDouble() <= shape.path("median_ns").asDouble(), key);
      assertTrue(shape.path("median_ns").asDouble() <= shape.path("p90_ns").asDouble(), key);
      // A real downcall costs more than the empty loop over the same call count.
      assertTrue(shape.path("median_ns").asDouble() > loopMedian, key);
    }
  }

  private static boolean ccAvailable() {
    try {
      Process process = new ProcessBuilder("cc", "--version").start();
      process.getInputStream().readAllBytes();
      return process.waitFor(15, TimeUnit.SECONDS) && process.exitValue() == 0;
    } catch (Exception e) {
      return false;
    }
  }
}
