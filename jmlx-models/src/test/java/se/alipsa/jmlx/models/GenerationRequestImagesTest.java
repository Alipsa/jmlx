package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.tokenizer.ChatTemplateOptions;
import se.alipsa.jmlx.tokenizer.HfTokenizer;
import se.alipsa.jmlx.vision.RgbImage;

/** Image-carrying requests: default emptiness, deep copy semantics, and copy-method retention. */
class GenerationRequestImagesTest {

  private static RgbImage image(int width, int height, int seed) {
    byte[] pixels = new byte[width * height * 3];
    for (int i = 0; i < pixels.length; i++) {
      pixels[i] = (byte) (seed + i);
    }
    return new RgbImage(width, height, pixels);
  }

  private static GenerationRequest base() {
    return new GenerationRequest(
        new int[] {4, 5}, GenerationConfig.greedyDefaults(3, Set.of()), CancellationToken.NONE);
  }

  @Test
  void requestsStartImageFree() {
    assertTrue(base().images().isEmpty());
    assertTrue(
        GenerationRequest.chat(
                mistralTokenizer(),
                List.of(Map.of("role", "user", "content", "hello")),
                ChatTemplateOptions.defaults(true),
                GenerationConfig.greedyDefaults(3, Set.of()),
                CancellationToken.NONE)
            .images()
            .isEmpty());
  }

  @Test
  void withImagesIsDeeplyImmutableToCallerMutation() {
    byte[] backing = new byte[6 * 3];
    for (int i = 0; i < backing.length; i++) {
      backing[i] = (byte) i;
    }
    RgbImage first = new RgbImage(2, 3, backing);
    List<RgbImage> callerList = new ArrayList<>(List.of(first, image(1, 1, 9)));

    GenerationRequest request = base().withImages(callerList);

    backing[0] = 0; // the image constructor already copied the buffer
    callerList.add(image(4, 4, 7));
    callerList.remove(0);

    assertEquals(2, request.images().size());
    assertEquals(first, request.images().get(0));
    assertEquals(List.of(first, image(1, 1, 9)), request.images());
    assertThrows(UnsupportedOperationException.class, () -> request.images().add(image(1, 1, 1)));
  }

  @Test
  void withImagesRejectsANullListOrNullElements() {
    assertThrows(NullPointerException.class, () -> base().withImages(null));
    assertThrows(
        NullPointerException.class, () -> base().withImages(List.of(image(1, 1, 1), null)));
  }

  @Test
  void copyMethodsRetainEveryFieldInBothOrders() {
    GenerationRequest chat =
        GenerationRequest.chat(
            mistralTokenizer(),
            List.of(Map.of("role", "user", "content", "hello")),
            ChatTemplateOptions.defaults(true),
            GenerationConfig.greedyDefaults(3, Set.of()),
            CancellationToken.NONE);
    RgbImage image = image(2, 2, 3);
    List<RgbImage> images = List.of(image, image(1, 1, 5));
    GenerationCachePolicy policy = GenerationCachePolicy.slidingWindow(8);

    GenerationRequest imagesFirst = chat.withImages(images).withCachePolicy(policy);
    GenerationRequest policyFirst = chat.withCachePolicy(policy).withImages(images);

    for (GenerationRequest copy : List.of(imagesFirst, policyFirst)) {
      assertArrayEquals(chat.promptTokenIds(), copy.promptTokenIds());
      assertEquals(chat.config(), copy.config());
      assertEquals(chat.cancellationToken(), copy.cancellationToken());
      assertEquals(chat.promptSpecialTokens(), copy.promptSpecialTokens());
      assertEquals(images, copy.images());
      assertEquals(policy, copy.cachePolicy());
    }
    // The original is untouched by either copy chain.
    assertTrue(chat.images().isEmpty());
    assertEquals(GenerationCachePolicy.full(), chat.cachePolicy());
  }

  @Test
  void textOnlyMetadataDefaultsToText() {
    // Pure metadata: no native library involved, so this runs (and must run) on bootstrapped and
    // unbootstrapped checkouts alike.
    assertEquals(
        Set.of(InputModality.TEXT), new DecoderMetadata("llama", 32_000, 32).inputModalities());
    assertEquals(
        Set.of(InputModality.TEXT), new EncoderMetadata("bert", 30_522, 6).inputModalities());
    assertEquals(
        Set.of(InputModality.TEXT), new Seq2SeqMetadata("t5", 32_128, 8, 8).inputModalities());
    // The set is immutable.
    assertThrows(
        UnsupportedOperationException.class,
        () -> new DecoderMetadata("llama", 32_000, 32).inputModalities().add(InputModality.IMAGE));
  }

  private static HfTokenizer mistralTokenizer() {
    Path root = Path.of(System.getProperty("jmlx.repository.root"));
    return HfTokenizer.fromDirectory(
        root.resolve("jmlx-tokenizer/src/test/resources/families/mistral"));
  }
}
