package se.alipsa.jmlx.tokenizer;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.List;
import org.junit.jupiter.api.Test;

class AddedTokenSplitterTest {

  private static final List<AddedToken> QWEN_ADDED_TOKENS =
      List.of(
          new AddedToken(151644, "<|im_start|>", true), new AddedToken(151645, "<|im_end|>", true));

  @Test
  void splitsPlainTextAroundLiteralAddedTokens() {
    AddedTokenSplitter splitter = new AddedTokenSplitter(QWEN_ADDED_TOKENS);
    List<AddedTokenSplitter.Segment> segments = splitter.split("<|im_start|>user\nhi<|im_end|>");
    assertEquals(
        List.of(
            new AddedTokenSplitter.Segment("<|im_start|>", true),
            new AddedTokenSplitter.Segment("user\nhi", false),
            new AddedTokenSplitter.Segment("<|im_end|>", true)),
        segments);
  }

  @Test
  void plainTextWithNoAddedTokensIsOneSegment() {
    AddedTokenSplitter splitter = new AddedTokenSplitter(QWEN_ADDED_TOKENS);
    assertEquals(
        List.of(new AddedTokenSplitter.Segment("hello world", false)),
        splitter.split("hello world"));
  }
}
