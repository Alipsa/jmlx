package se.alipsa.jmlx.buildsrc;

import java.io.File;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import javax.xml.XMLConstants;
import javax.xml.parsers.DocumentBuilderFactory;
import org.w3c.dom.Node;

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

  /** Reads the direct {@code <project>/<dependencies>/<dependency>} elements in {@code pomFile}. */
  public static List<Dependency> read(File pomFile) throws Exception {
    var document = document(pomFile);
    List<Dependency> result = new ArrayList<>();
    Node dependencies = child(document.getDocumentElement(), "dependencies");
    if (dependencies == null) {
      return result;
    }
    for (Node dependency = dependencies.getFirstChild();
        dependency != null;
        dependency = dependency.getNextSibling()) {
      if (dependency.getNodeType() == Node.ELEMENT_NODE
          && dependency.getNodeName().equals("dependency")) {
        result.add(
            new Dependency(
                field(dependency, "groupId"),
                field(dependency, "artifactId"),
                field(dependency, "version")));
      }
    }
    return result;
  }

  /** Returns the direct {@code <project>} child-element texts, keyed by element name. */
  public static Map<String, String> directChildTexts(File pomFile) throws Exception {
    var document = document(pomFile);
    Map<String, String> result = new LinkedHashMap<>();
    Node project = document.getDocumentElement();
    for (Node child = project.getFirstChild(); child != null; child = child.getNextSibling()) {
      if (child.getNodeType() == Node.ELEMENT_NODE) {
        result.put(child.getNodeName(), child.getTextContent());
      }
    }
    return result;
  }

  private static org.w3c.dom.Document document(File pomFile) throws Exception {
    DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
    factory.setFeature(XMLConstants.FEATURE_SECURE_PROCESSING, true);
    factory.setAttribute(XMLConstants.ACCESS_EXTERNAL_DTD, "");
    factory.setAttribute(XMLConstants.ACCESS_EXTERNAL_SCHEMA, "");
    factory.setXIncludeAware(false);
    factory.setExpandEntityReferences(false);
    return factory.newDocumentBuilder().parse(pomFile);
  }

  private static Node child(Node parent, String name) {
    for (Node child = parent.getFirstChild(); child != null; child = child.getNextSibling()) {
      if (child.getNodeType() == Node.ELEMENT_NODE && child.getNodeName().equals(name)) {
        return child;
      }
    }
    return null;
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
