package se.alipsa.jmlx.tokenizer;

/** Internal token text plus its original-input offset and optional resolved ID. */
record TokenPiece(
    String text, TokenOffset offset, Integer id, int typeId, boolean special, boolean padding) {
  TokenPiece(String text, TokenOffset offset, Integer id, int typeId, boolean special) {
    this(text, offset, id, typeId, special, false);
  }

  TokenPiece(String text, TokenOffset offset) {
    this(text, offset, null, 0, false, false);
  }
}
