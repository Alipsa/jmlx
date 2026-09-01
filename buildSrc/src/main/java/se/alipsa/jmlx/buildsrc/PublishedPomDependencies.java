package se.alipsa.jmlx.buildsrc;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import javax.xml.parsers.DocumentBuilderFactory;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

/**
 * Reads the {@code <dependency>} elements of a generated Maven publication POM.
 *
 * <p>Shared by every publishable module's release-verification tasks (root build.gradle's {@code
 * verifyNoSnapshotDependencies}, jmlx-tokenizer's {@code verifyPublishedDependencies}, and
 * jmlx-jinja's {@code verifyPublicationMetadata}/{@code dependencyReview}) so the same
 * childNodes-walk isn't hand-copied into each one.
 */
public final class PublishedPomDependencies {

  private PublishedPomDependencies() {}

  /** One {@code <dependency>} element's groupId/artifactId/version; any of them may be null. */
  public record Dependency(String groupId, String artifactId, String version) {

    /** {@code groupId:artifactId}. */
    public String groupArtifact() {
      return groupId + ":" + artifactId;
    }

    /** {@code groupId:artifactId}, plus {@code :version} when a version is present. */
    public String coordinates() {
      return version == null ? groupArtifact() : groupArtifact() + ":" + version;
    }
  }

  /** Reads every {@code <dependency>} element in {@code pomFile}. */
  public static List<Dependency> read(File pomFile) throws Exception {
    var document = DocumentBuilderFactory.newInstance().newDocumentBuilder().parse(pomFile);
    NodeList dependencyNodes = document.getElementsByTagName("dependency");
    List<Dependency> result = new ArrayList<>();
    for (int i = 0; i < dependencyNodes.getLength(); i++) {
      Node dependency = dependencyNodes.item(i);
      result.add(
          new Dependency(
              field(dependency, "groupId"),
              field(dependency, "artifactId"),
              field(dependency, "version")));
    }
    return result;
  }

  private static String field(Node dependency, String name) {
    for (Node child = dependency.getFirstChild(); child != null; child = child.getNextSibling()) {
      if (child.getNodeType() == Node.ELEMENT_NODE && child.getNodeName().equals(name)) {
        return child.getTextContent();
      }
    }
    return null;
  }
}
