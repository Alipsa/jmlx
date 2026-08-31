package se.alipsa.jmlx.tokenizer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class VocabularyTest {

  @Test
  void constructorThrowsWhenModelVocabHasTwoDifferentTokensSharingOneId() {
    // Without this check, idOf("a") and idOf("b") both resolved to 5 while tokenOf(5) returned
    // only whichever token modelVocab.forEach happened to iterate last -- silently breaking the
    // mutual-inverse guarantee this constructor's own javadoc promises, with the winner
    // depending on HashMap iteration order. TokenizerJsonLoader has its own, better-worded check
    // for a real tokenizer.json file; this is Vocabulary's own defense-in-depth line for a direct
    // caller like this test itself (PR #14 review round 8, finding 5).
    assertThrows(TokenizerException.class, () -> new Vocabulary(Map.of("a", 5, "b", 5), List.of()));
  }

  @Test
  void idOfAndTokenOfAreInverseWhenAnAddedTokenSharesAnIdWithADifferentModelVocabToken() {
    // Model vocab has "x" -> 5; an added token "y" also claims id 5. The added token must win
    // both directions: idOf("y") == 5 and tokenOf(5) == "y", with "x" no longer resolvable by
    // either lookup (PR #14 review, finding 8).
    Vocabulary vocabulary = new Vocabulary(Map.of("x", 5), List.of(new AddedToken(5, "y", false)));
    assertEquals(5, vocabulary.idOf("y"));
    assertEquals("y", vocabulary.tokenOf(5));
    assertFalse(vocabulary.hasId(6));
  }

  @Test
  void idOfAndTokenOfAreInverseWhenAnAddedTokenSharesATokenStringWithADifferentModelVocabId() {
    // Model vocab has "x" -> 5; an added token also spells "x" but claims id 7. The added token
    // must win both directions: idOf("x") == 7 and tokenOf(7) == "x", with id 5 no longer
    // resolving back to "x".
    Vocabulary vocabulary = new Vocabulary(Map.of("x", 5), List.of(new AddedToken(7, "x", false)));
    assertEquals(7, vocabulary.idOf("x"));
    assertEquals("x", vocabulary.tokenOf(7));
    assertFalse(vocabulary.hasId(5));
  }

  @Test
  void hasIdIsFalseForAnIdWithNoVocabularyEntry() {
    Vocabulary vocabulary = new Vocabulary(Map.of("x", 5), List.of());
    assertTrue(vocabulary.hasId(5));
    assertFalse(vocabulary.hasId(999));
  }

  @Test
  void maxKnownIdIsTheLargestModelVocabIdWhenNoAddedTokenExceedsIt() {
    Vocabulary vocabulary = new Vocabulary(Map.of("x", 5, "y", 3), List.of());
    assertEquals(5, vocabulary.maxKnownId());
  }

  @Test
  void maxKnownIdExcludesAModelVocabIdVacatedByAnAddedTokenCollision() {
    // "b" -> 2 in model.vocab is vacated when an added token claims content "b" under a
    // different id (1): id 2 is briefly assigned during construction but ends up with no
    // vocabulary entry at all, so maxKnownId (1, from the surviving ids 0 and 1) must not still
    // count it, or HfTokenizer#decode would treat a genuine above-vocab id as an in-range hole
    // instead (PR #14 review round 5, finding 4, adding coverage for round 4 finding 6's fix).
    Vocabulary vocabulary =
        new Vocabulary(Map.of("a", 0, "b", 2), List.of(new AddedToken(1, "b", false)));
    assertFalse(vocabulary.hasId(2));
    assertEquals(1, vocabulary.maxKnownId());
  }

  @Test
  void maxKnownIdIncludesAnAddedTokenThatExceedsEveryModelVocabId() {
    Vocabulary vocabulary =
        new Vocabulary(Map.of("x", 5), List.of(new AddedToken(100, "<|extra|>", true)));
    assertEquals(100, vocabulary.maxKnownId());
  }

  @Test
  void stringCollisionThatVacatesAnIdAlsoClearsThatIdsSpecialFlag() {
    // "<a>" first claims id 5 (special); a second added token reuses the same content "<a>" but
    // claims a different id, 7 -- vacating id 5 from idToToken via the token-string collision
    // path, not the id-collision path the test below already covers. specialIds must be pruned
    // there too, or isSpecial(5) stays true even though hasId(5) is now false (PR #14 review
    // round 3, finding 5).
    Vocabulary vocabulary =
        new Vocabulary(
            Map.of(), List.of(new AddedToken(5, "<a>", true), new AddedToken(7, "<a>", false)));
    assertFalse(vocabulary.hasId(5));
    assertFalse(vocabulary.isSpecial(5));
  }

  @Test
  void laterNonSpecialAddedTokenClearsStaleSpecialFlagFromAnEarlierCollidingAddedToken() {
    // Two added tokens collide on id 5: the first is special, the second (which wins the
    // collision) is not. isSpecial(5) must follow whichever token currently occupies the id, not
    // remain stuck true from the first one processed (PR #14 review, finding 5).
    Vocabulary vocabulary =
        new Vocabulary(
            Map.of(), List.of(new AddedToken(5, "<a>", true), new AddedToken(5, "<b>", false)));
    assertEquals("<b>", vocabulary.tokenOf(5));
    assertFalse(vocabulary.isSpecial(5));
  }
}
