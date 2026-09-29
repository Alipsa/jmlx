package se.alipsa.jmlx.models;

/** How prompt special tokens were or should be applied. */
public enum PromptSpecialTokens {
  /** Apply the tokenizer post-processor's special tokens. */
  ADD,
  /** Tokenize without adding post-processor special tokens. */
  OMIT,
  /** Prompt IDs were supplied directly by the caller. */
  PRETOKENIZED
}
