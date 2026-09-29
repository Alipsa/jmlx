package se.alipsa.jmlx.tokenizer;

/** Request-local generated-token decoder. */
public interface IncrementalTokenDecoder {
  /**
   * Whether {@link #append} can emit non-empty text before {@link #finish} is called. When {@code
   * false}, the underlying {@code tokenizer.json} decoder shape has no incremental implementation
   * here, so every {@link #append} call returns {@code ""} and the complete decoded text is only
   * available from {@link #finish}. A caller streaming {@code textDelta} per generated token (the
   * pattern this module's own README promotes) can check this up front to know whether that pattern
   * will actually produce visible output as generation proceeds, rather than silently seeing
   * nothing until the end with no signal why.
   *
   * @return whether this decoder streams incremental output
   */
  boolean streams();

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
