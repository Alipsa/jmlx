package se.alipsa.jmlx.models;

import java.util.Objects;
import se.alipsa.jmlx.tokenizer.HfTokenizer;

/**
 * The four validated Qwen3-VL prompt-side token IDs used by the pretokenized wrapper grammar.
 *
 * <p>Unlike SmolVLM (7.3a), these sentinels are real vocabulary tokens in the pinned artifact: each
 * one is present in the tokenizer's added-token metadata and encodes to exactly one ID. The
 * reference chat template renders each image as a three-token wrapper around a single image
 * sentinel -- vision start, one {@link #IMAGE_TOKEN_CONTENT} sentinel, and vision end -- and the
 * processor's expansion replaces that one sentinel with the image's merged feature-row count of
 * copies of it, in prompt order. The checkpoint's outer {@code config.json} is the authoritative
 * source of the four numeric IDs ({@code image_token_id} / {@code video_token_id} / {@code
 * vision_start_token_id} / {@code vision_end_token_id}).
 *
 * <p>{@link #resolve(HfTokenizer, int, int, int, int)} is the model-load path: it resolves each
 * sentinel spelling to its ID through the tokenizer and cross-checks the result against the ID the
 * outer config declares for that role. Candidate numeric IDs are never hard-coded in production
 * code; the constants here carry only the sentinel spellings, and the golden and the tests carry
 * the pinned numeric values.
 *
 * @param imageTokenId the image sentinel token declared by the checkpoint
 * @param videoTokenId the video sentinel token, or -1 when the checkpoint declares no video token
 * @param visionStartTokenId the wrapper's start token
 * @param visionEndTokenId the wrapper's end token
 */
record Qwen3VlPromptTokens(
    int imageTokenId, int videoTokenId, int visionStartTokenId, int visionEndTokenId) {

  /** The image sentinel spelling as it appears in the checkpoint's tokenizer. */
  static final String IMAGE_TOKEN_CONTENT = "<|image_pad|>";

  /** The video sentinel spelling as it appears in the checkpoint's tokenizer. */
  static final String VIDEO_TOKEN_CONTENT = "<|video_pad|>";

  /** The wrapper's start spelling as it appears in the checkpoint's tokenizer. */
  static final String VISION_START_TOKEN_CONTENT = "<|vision_start|>";

  /** The wrapper's end spelling as it appears in the checkpoint's tokenizer. */
  static final String VISION_END_TOKEN_CONTENT = "<|vision_end|>";

  public Qwen3VlPromptTokens {
    if (imageTokenId < 0 || visionStartTokenId < 0 || visionEndTokenId < 0) {
      throw new IllegalArgumentException(
          "image, vision start and vision end token IDs must be non-negative: "
              + imageTokenId
              + ", "
              + visionStartTokenId
              + ", "
              + visionEndTokenId);
    }
    if (videoTokenId < -1) {
      throw new IllegalArgumentException(
          "videoTokenId must be -1 (no video token) or a token ID: " + videoTokenId);
    }
    if (imageTokenId == visionStartTokenId
        || imageTokenId == visionEndTokenId
        || visionStartTokenId == visionEndTokenId) {
      throw new IllegalArgumentException(
          "image, vision start and vision end token IDs must be distinct: "
              + imageTokenId
              + ", "
              + visionStartTokenId
              + ", "
              + visionEndTokenId);
    }
    if (videoTokenId >= 0
        && (videoTokenId == imageTokenId
            || videoTokenId == visionStartTokenId
            || videoTokenId == visionEndTokenId)) {
      throw new IllegalArgumentException(
          "videoTokenId must be distinct from the image markers: " + videoTokenId);
    }
  }

  /**
   * Resolves the four sentinel IDs from the tokenizer and cross-checks each against the ID the
   * checkpoint's outer config declares for its role.
   *
   * <p>Each sentinel spelling ({@link #IMAGE_TOKEN_CONTENT}, {@link #VIDEO_TOKEN_CONTENT}, {@link
   * #VISION_START_TOKEN_CONTENT} and {@link #VISION_END_TOKEN_CONTENT}) is resolved through the
   * tokenizer's name-to-ID direction, in the order image, video, vision start, vision end. A
   * sentinel that is not a vocabulary token fails the load (the tokenizer's own exception is left
   * to propagate), and a sentinel whose resolved ID differs from the config's declaration for that
   * role is rejected rather than silently overridden.
   *
   * @param tokenizer the model's tokenizer; null is rejected
   * @param configImageTokenId the outer config's image token ID
   * @param configVideoTokenId the outer config's video token ID
   * @param configVisionStartTokenId the outer config's vision start token ID
   * @param configVisionEndTokenId the outer config's vision end token ID
   * @return the resolved, cross-checked token set
   * @throws IllegalArgumentException when a resolved ID differs from the config's declaration for
   *     its role
   */
  static Qwen3VlPromptTokens resolve(
      HfTokenizer tokenizer,
      int configImageTokenId,
      int configVideoTokenId,
      int configVisionStartTokenId,
      int configVisionEndTokenId) {
    Objects.requireNonNull(tokenizer, "tokenizer");
    int image = resolveRole(tokenizer, IMAGE_TOKEN_CONTENT, configImageTokenId, "image");
    int video = resolveRole(tokenizer, VIDEO_TOKEN_CONTENT, configVideoTokenId, "video");
    int start =
        resolveRole(
            tokenizer, VISION_START_TOKEN_CONTENT, configVisionStartTokenId, "vision start");
    int end =
        resolveRole(tokenizer, VISION_END_TOKEN_CONTENT, configVisionEndTokenId, "vision end");
    return new Qwen3VlPromptTokens(image, video, start, end);
  }

  private static int resolveRole(
      HfTokenizer tokenizer, String sentinel, int configId, String role) {
    int resolved = tokenizer.eosTokenId(sentinel).getAsInt();
    if (resolved != configId) {
      throw new IllegalArgumentException(
          "the "
              + role
              + " sentinel resolves to ID "
              + resolved
              + " but the checkpoint config declares ID "
              + configId);
    }
    return resolved;
  }
}
