package se.alipsa.jmlx.tokenizer;

/** Thrown when a tokenizer.json file, chat template, or token stream cannot be processed. */
public final class TokenizerException extends RuntimeException {

  /** Creates an exception carrying a description of the tokenizer failure. */
  public TokenizerException(String message) {
    super(message);
  }

  /** Creates an exception carrying {@code cause}'s failure as context. */
  public TokenizerException(String message, Throwable cause) {
    super(message, cause);
  }
}
