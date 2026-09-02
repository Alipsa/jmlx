package se.alipsa.jmlx.jinja.internal.util;

import java.time.ZonedDateTime;
import java.util.Locale;

/**
 * The deterministic {@code strftime_now} directive formatter pinned to the upstream {@code
 * @huggingface/jinja} 0.5.9 token set: {@code %Y}, {@code %m}, {@code %d}, {@code %b}, {@code %B},
 * {@code %H}, {@code %M}, and {@code %%}. All other {@code %x} pairs, a terminal {@code %}, and
 * ordinary characters are copied through literally.
 */
public final class PosixStrftime {
  private static final String[] SHORT_MONTH_NAMES = {
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
  };
  private static final String[] LONG_MONTH_NAMES = {
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December"
  };

  private PosixStrftime() {}

  /**
   * Formats {@code dateTime} according to {@code format}, recognizing only the pinned directive
   * set. Month names are enumerated explicitly under {@link Locale#ROOT} C/POSIX English names, so
   * JVM default locale data never influences the result.
   *
   * @param dateTime the zoned instant to format
   * @param format the {@code strftime}-style format string
   * @return the formatted string
   */
  public static String format(ZonedDateTime dateTime, String format) {
    var out = new StringBuilder();
    for (int i = 0; i < format.length(); i++) {
      char c = format.charAt(i);
      if (c != '%' || i + 1 == format.length()) {
        out.append(c);
        continue;
      }
      char directive = format.charAt(++i);
      out.append(
          switch (directive) {
            case 'Y' -> Integer.toString(dateTime.getYear());
            case 'm' -> String.format(Locale.ROOT, "%02d", dateTime.getMonthValue());
            case 'd' -> String.format(Locale.ROOT, "%02d", dateTime.getDayOfMonth());
            case 'b' -> SHORT_MONTH_NAMES[dateTime.getMonthValue() - 1];
            case 'B' -> LONG_MONTH_NAMES[dateTime.getMonthValue() - 1];
            case 'H' -> String.format(Locale.ROOT, "%02d", dateTime.getHour());
            case 'M' -> String.format(Locale.ROOT, "%02d", dateTime.getMinute());
            case '%' -> "%";
            default -> "%" + directive;
          });
    }
    return out.toString();
  }
}
