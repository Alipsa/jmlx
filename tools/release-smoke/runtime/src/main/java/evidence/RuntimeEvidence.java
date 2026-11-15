package evidence;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Properties;

/** Records the JVM actually executing a successful smoke consumer. */
public final class RuntimeEvidence {
  private RuntimeEvidence() {}

  /**
   * Writes measured runtime properties to the Gradle-supplied evidence file.
   *
   * @throws IOException if the runtime evidence cannot be written
   */
  public static void write() throws IOException {
    Path output = Path.of(System.getProperty("smoke.runtime.evidence"));
    Properties properties = new Properties();
    for (String name :
        new String[] {
          "java.version", "java.runtime.version", "java.vendor", "java.home", "java.vm.name"
        }) {
      properties.setProperty(name, System.getProperty(name));
    }
    properties.setProperty("java.feature", Integer.toString(Runtime.version().feature()));
    Files.createDirectories(output.getParent());
    try (OutputStream stream = Files.newOutputStream(output)) {
      properties.storeToXML(stream, "Measured smoke consumer JVM", "UTF-8");
    }
  }
}
