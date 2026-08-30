package se.alipsa.jmlx.tokenizer;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.net.URISyntaxException;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;

class HfTokenizerTest {

  private Path fixture(String name) {
    try {
      return Path.of(getClass().getResource(name).toURI());
    } catch (URISyntaxException e) {
      throw new IllegalStateException(e);
    }
  }

  @Test
  void qwenStyleEncodeDecodeRoundTrips() {
    HfTokenizer tokenizer = HfTokenizer.fromFile(fixture("qwen-style.tokenizer.json"));
    List<Integer> ids = tokenizer.encode("low the", false);
    assertEquals(List.of(13, 16), ids);
    assertEquals("low the", tokenizer.decode(ids, false));
  }

  @Test
  void llama3StylePrependsBosTokenWhenAddSpecialTokensIsTrue() {
    HfTokenizer tokenizer = HfTokenizer.fromFile(fixture("llama3-style.tokenizer.json"));
    List<Integer> ids = tokenizer.encode("low the", true);
    assertEquals(List.of(128000, 13, 16), ids);
  }

  @Test
  void llama3StyleOmitsBosTokenWhenAddSpecialTokensIsFalse() {
    HfTokenizer tokenizer = HfTokenizer.fromFile(fixture("llama3-style.tokenizer.json"));
    assertEquals(List.of(13, 16), tokenizer.encode("low the", false));
  }

  @Test
  void decodeSkipsSpecialTokensWhenRequested() {
    HfTokenizer tokenizer = HfTokenizer.fromFile(fixture("llama3-style.tokenizer.json"));
    List<Integer> ids = tokenizer.encode("low the", true);
    assertEquals("low the", tokenizer.decode(ids, true));
  }
}
