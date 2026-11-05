#!/usr/bin/env groovy
// Captures the sanitized Central Portal availability fixture required by the combined
// release workflow. Uploads (releaseAll) are fail-closed: before an absence-based decision
// (a coordinate is "not published") is trusted, the fixture must confirm that (a) the
// published exemplar in this namespace reports published:true and (b) a never-uploaded
// coordinate reports published:false, via the same authenticated /publisher/published
// endpoint the plan and upload use.
//
// Credentials come from SONATYPE_USERNAME/SONATYPE_PASSWORD or
// gradle.properties under $GRADLE_USER_HOME (default ~/.gradle). Only the request parameters,
// HTTP status and boolean body are recorded; no credential material is written.
//
// The default output path is resolved from this script's location (the repository root's
// req/releaseAll-portal-fixture.json), so the script works from any working directory; an
// explicit argument still overrides it.
import groovy.json.JsonOutput
import groovy.json.JsonSlurper
import java.net.http.*
import java.time.Duration

Properties properties = new Properties()
String gradleUserHome = System.getenv('GRADLE_USER_HOME') ?: new File(System.getProperty('user.home'), '.gradle').path
File configuration = new File(gradleUserHome, 'gradle.properties')
if (configuration.exists()) configuration.withInputStream { properties.load(it) }
String user = System.getenv('SONATYPE_USERNAME') ?: properties.getProperty('sonatypeUsername')
String password = System.getenv('SONATYPE_PASSWORD') ?: properties.getProperty('sonatypePassword')
if (!user || !password) throw new IllegalStateException('Set SONATYPE_USERNAME/SONATYPE_PASSWORD or Gradle sonatypeUsername/sonatypePassword')
String namespace = 'se.alipsa'
// jmlx-jinja 0.6.0 is live on Maven Central; the 0.0.0 coordinate was never uploaded.
String publishedName = 'jmlx-jinja'
String publishedVersion = '0.6.0'
String absentVersion = '0.0.0-release-all-never-uploaded'
def client = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(15)).build()
def request = { String name, String version ->
  String query = [namespace: namespace, name: name, version: version].collect { k, v -> "${k}=${URLEncoder.encode(v, 'UTF-8')}" }.join('&')
  def response = client.send(HttpRequest.newBuilder(URI.create('https://central.sonatype.com/api/v1/publisher/published?' + query))
    .timeout(Duration.ofSeconds(30)).header('Authorization', 'Bearer ' + Base64.encoder.encodeToString("${user}:${password}".getBytes('UTF-8'))).GET().build(), HttpResponse.BodyHandlers.ofString())
  if (response.statusCode() != 200) throw new IllegalStateException("Fixture request returned HTTP ${response.statusCode()}")
  def body = new JsonSlurper().parseText(response.body())
  if (!(body.published instanceof Boolean)) throw new IllegalStateException('Fixture response missing boolean published')
  [parameters: [namespace: namespace, name: name, version: version], status: response.statusCode(), body: [published: body.published]]
}
def fixture = [source: 'https://central.sonatype.com/api-doc#checkComponentPublished', captured: java.time.Instant.now().toString(),
  namespacePolicy: 'groupId', published: request(publishedName, publishedVersion), absent: request(publishedName, absentVersion)]
if (fixture.published.body.published != true || fixture.absent.body.published != false) throw new IllegalStateException('Portal semantics not confirmed; discovery remains blocked')
File scriptDir = new File(getClass().protectionDomain.codeSource.location.toURI()).parentFile
File output = args ? new File(args[0]) : new File(scriptDir, '../../req/releaseAll-portal-fixture.json').canonicalFile
output.text = JsonOutput.prettyPrint(JsonOutput.toJson(fixture))
println 'Confirmed exact Maven groupId namespace and published:false for never-uploaded coordinate; saved sanitized fixture.'
