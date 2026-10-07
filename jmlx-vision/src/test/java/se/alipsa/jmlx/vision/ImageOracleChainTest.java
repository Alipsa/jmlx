package se.alipsa.jmlx.vision;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.charset.StandardCharsets;
import org.junit.jupiter.api.Test;
import tools.jackson.databind.JsonNode;

/**
 * Byte-exact (uint8) and float32 (1e-6, the pinned normalized tolerance) comparison of the full
 * SmolVLM/Idefics3 preprocessing chain against the committed oracle fixtures.
 */
class ImageOracleChainTest {

  private static final float NORMALIZED_TOLERANCE = 1e-6f;

  private static SmolVlmProcessorConfig chainConfig(int resample, boolean doImageSplitting) {
    String json = configText();
    json = json.replace("\"longest_edge\": 2048", "\"longest_edge\": 96");
    json = json.replace("\"longest_edge\": 512", "\"longest_edge\": 32");
    json = json.replace("\"resample\": 1", "\"resample\": " + resample);
    json =
        json.replace("\"do_image_splitting\": true", "\"do_image_splitting\": " + doImageSplitting);
    return SmolVlmProcessorConfig.fromJson(json.getBytes(StandardCharsets.UTF_8));
  }

  private static String configText() {
    try {
      return OracleFixtures.configText();
    } catch (Exception e) {
      throw new AssertionError(e);
    }
  }

  @Test
  void chainMatchesThePinnedOracleForEveryCase() throws Exception {
    JsonNode expected = OracleFixtures.read("chain.expected.json");
    JsonNode expectedConfig = expected.required("config");
    assertEquals(96, expectedConfig.required("size").intValue());
    assertEquals(32, expectedConfig.required("maxImageSize").intValue());
    assertEquals(0.00392156862745098, expectedConfig.required("rescaleFactor").asDouble());
    for (String flag : new String[] {"doResize", "doRescale", "doNormalize", "doPad"}) {
      assertTrue(expectedConfig.required(flag).asBoolean(), flag);
    }
    assertArrayEquals(new float[] {0.5f, 0.5f, 0.5f}, floats(expectedConfig.required("imageMean")));
    assertArrayEquals(new float[] {0.5f, 0.5f, 0.5f}, floats(expectedConfig.required("imageStd")));

    JsonNode cases = OracleFixtures.read("chain.input.json").required("cases");
    assertEquals(cases.size(), expected.required("cases").size());
    assertTrue(cases.size() > 0);
    for (int i = 0; i < cases.size(); i++) {
      JsonNode testCase = cases.get(i);
      String name = testCase.required("name").asString();
      JsonNode expectedCase = OracleFixtures.findCase(expected.required("cases"), name);
      int resample = testCase.required("resample").intValue();
      boolean splitting = testCase.required("doImageSplitting").asBoolean();
      assertEquals(expectedCase.required("resample").intValue(), resample, name);
      assertEquals(expectedCase.required("doImageSplitting").asBoolean(), splitting, name);

      SmolVlmImageProcessor processor = new SmolVlmImageProcessor(chainConfig(resample, splitting));
      RgbImage source =
          ImageDecoder.decode(
              OracleFixtures.fixtures().resolve(testCase.required("image").asString()));
      SmolVlmImageProcessor.ChainIntermediates chain = processor.intermediates(source);

      JsonNode stage1 = expectedCase.required("stage1");
      assertEquals(stage1.required("width").intValue(), chain.stage1().width(), name);
      assertEquals(stage1.required("height").intValue(), chain.stage1().height(), name);
      assertArrayEquals(
          OracleFixtures.base64(stage1.required("pixelsBase64").asString()),
          chain.stage1().pixels(),
          name + " stage1");

      if (splitting) {
        JsonNode stage2 = expectedCase.required("stage2");
        assertNotNull(chain.stage2(), name);
        assertEquals(stage2.required("width").intValue(), chain.stage2().width(), name);
        assertEquals(stage2.required("height").intValue(), chain.stage2().height(), name);
        assertArrayEquals(
            OracleFixtures.base64(stage2.required("pixelsBase64").asString()),
            chain.stage2().pixels(),
            name + " stage2");
      } else {
        assertNull(chain.stage2(), name);
      }

      assertEquals(expectedCase.required("rows").intValue(), chain.rows(), name);
      assertEquals(expectedCase.required("cols").intValue(), chain.cols(), name);
      JsonNode tiles = expectedCase.required("tiles");
      assertEquals(tiles.size(), chain.frames().size(), name);
      for (int t = 0; t < tiles.size(); t++) {
        JsonNode tile = tiles.get(t);
        RgbImage frame = chain.frames().get(t);
        assertEquals(tile.required("width").intValue(), frame.width(), name + " tile " + t);
        assertEquals(tile.required("height").intValue(), frame.height(), name + " tile " + t);
        if (tile.has("crop")) {
          // A split crop must be a byte slice of stage2 at the recorded rectangle.
          JsonNode crop = tile.required("crop");
          byte[] expectedCrop =
              OracleFixtures.crop(
                  chain.stage2(),
                  crop.get(0).intValue(),
                  crop.get(1).intValue(),
                  crop.get(2).intValue(),
                  crop.get(3).intValue());
          assertArrayEquals(expectedCrop, frame.pixels(), name + " tile " + t + " crop");
        } else {
          assertTrue(tile.required("global").asBoolean(), name + " tile " + t);
          assertArrayEquals(
              OracleFixtures.base64(tile.required("pixelsBase64").asString()),
              frame.pixels(),
              name + " tile " + t + " global");
        }
      }

      SmolVlmImageProcessorResult result = processor.process(source);
      assertEquals(tiles.size(), result.tiles().size(), name);
      assertEquals(expectedCase.required("rows").intValue(), result.rows(), name);
      assertEquals(expectedCase.required("cols").intValue(), result.cols(), name);
      JsonNode masks = expectedCase.required("masks");
      assertEquals(tiles.size(), masks.size(), name);
      float maxError = 0f;
      for (int t = 0; t < tiles.size(); t++) {
        JsonNode tile = tiles.get(t);
        ImageTensor tensor = result.tiles().get(t);
        assertArrayEquals(
            new int[] {1, tile.required("height").intValue(), tile.required("width").intValue(), 3},
            tensor.shape(),
            name + " tile " + t);
        assertEquals(Layout.NHWC, tensor.layout(), name + " tile " + t);
        float[] expectedValues =
            OracleFixtures.float32(
                OracleFixtures.base64(tile.required("normalizedBase64").asString()));
        float[] values = tensor.values();
        assertEquals(expectedValues.length, values.length, name + " tile " + t);
        for (int p = 0; p < values.length; p++) {
          maxError = Math.max(maxError, Math.abs(values[p] - expectedValues[p]));
        }
        int[] mask = result.pixelMasks().get(t);
        JsonNode maskRecord = masks.get(t);
        assertTrue(maskRecord.required("allOnes").asBoolean(), name + " tile " + t);
        assertEquals(
            maskRecord.required("width").intValue() * maskRecord.required("height").intValue(),
            mask.length,
            name + " tile " + t);
        for (int v : mask) {
          assertEquals(1, v, name + " tile " + t);
        }
      }
      assertTrue(maxError <= NORMALIZED_TOLERANCE, name + " normalized max error " + maxError);
    }
  }

  private static float[] floats(JsonNode array) {
    float[] out = new float[array.size()];
    for (int i = 0; i < out.length; i++) {
      out[i] = (float) array.get(i).asDouble();
    }
    return out;
  }
}
