package se.alipsa.jmlx.tokenizer;

/** Request-local generated-token decoder. */
public interface IncrementalTokenDecoder {
  /**
   * Accepts one token ID and returns text that is stable to emit now.
   *
   * @param tokenId generated token ID
   * @return stable, possibly empty text delta
   */
  String append(int tokenId);

  /**
   * Flushes any incomplete terminal sequence. May be called once.
   *
   * @return final, possibly empty text delta
   */
  String finish();
}
