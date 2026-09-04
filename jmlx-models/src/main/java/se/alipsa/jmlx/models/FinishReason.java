package se.alipsa.jmlx.models;

/** Why a generation ended. */
public enum FinishReason {
  EOS,
  STOP_TOKEN,
  MAX_TOKENS,
  CANCELLED
}
