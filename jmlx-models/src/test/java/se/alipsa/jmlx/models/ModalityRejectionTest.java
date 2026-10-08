package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static se.alipsa.jmlx.models.SchedulerFixtures.checkpoint;
import static se.alipsa.jmlx.models.SchedulerFixtures.config;
import static se.alipsa.jmlx.models.SchedulerFixtures.greedy;
import static se.alipsa.jmlx.models.SchedulerFixtures.start;

import java.util.List;
import java.util.Set;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.vision.RgbImage;

/**
 * Every delivered model is text-only today: generation and scheduler admission reject image
 * requests before any native work, naming the model type.
 */
@EnabledIfNativeAvailable
class ModalityRejectionTest {

  private static RgbImage image() {
    return new RgbImage(2, 2, new byte[12]);
  }

  @Test
  void textOnlyDecodersRejectImagesBeforeNativeWork() throws Exception {
    try (MLXScope scope = new MLXScope()) {
      DecoderModel model = (DecoderModel) TextGenerationModels.load(scope, checkpoint("llama"));
      assertEquals(Set.of(InputModality.TEXT), model.metadata().inputModalities());
      IllegalArgumentException e =
          assertThrows(
              IllegalArgumentException.class,
              () ->
                  model.generate(
                      greedy(new int[] {1, 7, 42}, 4).withImages(List.of(image())), ignored -> {}));
      assertEquals("model_type llama does not accept images", e.getMessage());
    }
  }

  @Test
  void textOnlySeq2SeqRejectsImagesBeforeNativeWork() throws Exception {
    try (MLXScope scope = new MLXScope()) {
      T5Model model = T5Model.load(scope, checkpoint("t5-gated"));
      assertEquals(Set.of(InputModality.TEXT), model.metadata().inputModalities());
      IllegalArgumentException e =
          assertThrows(
              IllegalArgumentException.class,
              () ->
                  model.generate(
                      greedy(new int[] {5, 7, 9, 1}, 4).withImages(List.of(image())),
                      ignored -> {}));
      assertEquals("model_type t5 does not accept images", e.getMessage());
    }
  }

  @Test
  void schedulerAdmissionRejectsImagesSynchronously() {
    try (BatchGenerationScheduler scheduler = start("llama", config(2, 4))) {
      IllegalArgumentException e =
          assertThrows(
              IllegalArgumentException.class,
              () ->
                  scheduler.submit(
                      greedy(new int[] {1, 7, 42}, 4).withImages(List.of(image())), ignored -> {}));
      assertEquals("model_type llama does not accept images", e.getMessage());
    }
  }
}
