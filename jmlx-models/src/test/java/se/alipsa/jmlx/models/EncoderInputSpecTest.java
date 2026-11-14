package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.file.Path;
import java.util.Collections;
import java.util.List;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.tokenizer.TokenOffset;
import se.alipsa.jmlx.tokenizer.TokenizerEncoding;

/** Caller validation uses the same ID, type, mask and CLS rules as native BERT inference. */
class EncoderInputSpecTest {
  private static TokenizerEncoding input(
      List<Integer> ids, List<Integer> types, List<Integer> masks, List<Integer> special) {
    return new TokenizerEncoding(
        ids, types, masks, special, Collections.nCopies(2, TokenOffset.NONE));
  }

  @Test
  void validatesAllEncoderColumnsWithoutLoadingNativeCode() throws Exception {
    var spec =
        TextEncoderModels.inputSpec(
            Path.of(
                System.getProperty("jmlx.repository.root"),
                "tools",
                "hf-reference",
                "goldens",
                "checkpoints",
                "bert"));
    List<TokenizerEncoding> invalid =
        List.of(
            input(List.of(2, 99), List.of(0, 0), List.of(1, 1), List.of(1, 0)),
            input(List.of(2, 3), List.of(0, 2), List.of(1, 1), List.of(1, 0)),
            input(List.of(2, 3), List.of(0, 0), List.of(1, 2), List.of(1, 0)),
            input(List.of(2, 3), List.of(0, 0), List.of(1, 1), List.of(1, 2)),
            input(List.of(2, 3), List.of(0, 0), List.of(0, 1), List.of(1, 0)));
    for (TokenizerEncoding encoding : invalid) {
      assertThrows(
          IllegalArgumentException.class,
          () -> spec.validate(encoding, new Pooling(Pooling.Mode.CLS, false)));
    }
    spec.validate(
        input(List.of(2, 3), List.of(0, 0), List.of(0, 1), List.of(1, 0)),
        new Pooling(Pooling.Mode.MEAN, false));
  }
}
