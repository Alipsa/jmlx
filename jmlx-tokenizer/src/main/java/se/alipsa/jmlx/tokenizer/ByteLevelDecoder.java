package se.alipsa.jmlx.tokenizer;

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Objects;

/**
 * Decodes a list of token strings back to text by byte-decoding every token, added or not, and
 * concatenating their raw bytes before a single UTF-8 (lossy) conversion over the whole sequence.
 *
 * <p>Does not special-case added-token strings with a literal pass-through, unlike an earlier
 * version of this port modeled on {@code Decoder.swift}'s {@code ByteLevelDecoder}
 * (swift-transformers): HF's own {@code TokenizerImpl::decode} (`huggingface/tokenizers`'s Rust
 * source) resolves every id -- added-vocabulary or model-vocabulary -- into the same flat {@code
 * Vec<String>} and passes all of it to {@code decoder.decode(tokens)} with no distinction based on
 * origin; {@code ByteLevel}'s own {@code decode_chain} then applies the identical per-character
 * byte-level mapping (falling back to a token's own raw UTF-8 bytes on any unmapped character, see
 * {@link ByteLevelCoding#decodeToBytes}) to every token before the one final {@code
 * from_utf8_lossy} over the fully-concatenated buffer (PR #14 review round 6, finding 1). For
 * Qwen2.5's and Llama-3's real {@code added_tokens} -- every one of which is built from plain
 * printable-ASCII characters, each already byte-level-identity-mapped -- this makes no observable
 * difference from the literal pass-through it replaces; it matters for a hypothetical added token
 * whose content contains a character outside that identity range, which HF would still byte-decode
 * (or fall back to raw UTF-8 bytes for) rather than pass through verbatim.
 */
public final class ByteLevelDecoder {

  private ByteLevelDecoder() {}

  /**
   * Decodes {@code tokens} to text.
   *
   * @param tokens byte-level token strings
   * @return decoded text
   */
  public static String decode(List<String> tokens) {
    Objects.requireNonNull(tokens, "ByteLevelDecoder.decode: tokens must not be null");
    ByteArrayOutputStream pending = new ByteArrayOutputStream();
    for (String token : tokens) {
      pending.writeBytes(ByteLevelCoding.decodeToBytes(token));
    }
    return new String(pending.toByteArray(), StandardCharsets.UTF_8);
  }
}
