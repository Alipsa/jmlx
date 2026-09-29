package se.alipsa.jmlx.tokenizer;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/** Text represented as Unicode scalars carrying original UTF-8 byte ranges. */
record AlignedText(List<Unit> units) {
  AlignedText {
    units = List.copyOf(units);
  }

  static AlignedText original(String text) {
    List<Unit> units = new ArrayList<>();
    int byteIndex = 0;
    for (int index = 0; index < text.length(); ) {
      int codePoint = text.codePointAt(index);
      String value = new String(Character.toChars(codePoint));
      int nextByte = byteIndex + value.getBytes(StandardCharsets.UTF_8).length;
      units.add(new Unit(value, byteIndex, nextByte));
      byteIndex = nextByte;
      index += Character.charCount(codePoint);
    }
    return new AlignedText(units);
  }

  String text() {
    StringBuilder result = new StringBuilder();
    units.forEach(unit -> result.append(unit.value()));
    return result.toString();
  }

  TokenOffset offset() {
    if (units.isEmpty()) {
      return TokenOffset.NONE;
    }
    return new TokenOffset(units.getFirst().startByte(), units.getLast().endByte());
  }

  record Unit(String value, int startByte, int endByte) {}
}
