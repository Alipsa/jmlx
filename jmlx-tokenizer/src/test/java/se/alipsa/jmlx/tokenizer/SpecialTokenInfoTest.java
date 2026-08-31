package se.alipsa.jmlx.tokenizer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

class SpecialTokenInfoTest {

  @Test
  void bothEmptyIdsAndTokensThrowInsteadOfSilentlyVanishingTheSpecialToken() {
    // Both-empty passes an ids.size() != tokens.size() check (0 == 0), but would still make
    // PostProcessorApplier.applyTemplate loop zero times over the special token -- silently
    // dropping it (e.g. a BOS token) instead of failing loudly (PR #14 review round 3, finding 1).
    assertThrows(TokenizerException.class, () -> new SpecialTokenInfo("<s>", List.of(), List.of()));
  }

  @Test
  void mismatchedNonEmptyLengthsThrow() {
    assertThrows(
        TokenizerException.class, () -> new SpecialTokenInfo("<s>", List.of(1, 2), List.of("<s>")));
  }

  @Test
  void equalNonEmptyLengthsConstructSuccessfully() {
    new SpecialTokenInfo("<s>", List.of(1), List.of("<s>"));
  }

  @Test
  void mutatingTheCallersListsAfterConstructionDoesNotAffectTheStoredCopies() {
    // The compact constructor's List.copyOf defends against exactly this: without it, a record
    // stores its constructor arguments by reference, so a caller mutating the mutable list it
    // passed in after construction would silently violate the invariant just validated (PR #14
    // review round 4, finding 10; coverage added PR #14 review round 5, finding 4).
    List<Integer> ids = new ArrayList<>(List.of(1));
    List<String> tokens = new ArrayList<>(List.of("<s>"));
    SpecialTokenInfo info = new SpecialTokenInfo("<s>", ids, tokens);
    ids.clear();
    tokens.clear();
    assertEquals(List.of(1), info.ids());
    assertEquals(List.of("<s>"), info.tokens());
  }
}
