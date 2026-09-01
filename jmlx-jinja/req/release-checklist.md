# Release checklist

- [ ] Confirm JDK 21 and the exact Node version in `upstream/upstream-lock.json`, then run `./gradlew :jmlx-jinja:releaseVerification` from a clean source checkout. Use `-PreleaseVerificationAllowDirty` only for development; its report is not a release candidate.
- [ ] Read `jmlx-jinja/build/reports/release-verification.md` and its JSON companion. The task prepares an isolated Gradle home, runs the candidate matrix offline, and compares two clean archive builds.
- [ ] The isolated verification repository is created per candidate; do not use `publishToMavenLocal` for this verification step. The verifier publishes with `:jmlx-jinja:publishMavenPublicationToReleaseVerificationRepository`. (The actual release below does use `publishToMavenLocal`, via the nexus-release-plugin's `bundle` task -- that is expected there, just not while verifying.)
- [ ] Review `NOTICE`, `req/model-fixture-policy.md`, `CHANGELOG.md`, public Javadoc, and the generated POM.
- [ ] Review dependency updates with `./gradlew :jmlx-jinja:dependencyUpdates`; do not upgrade the pinned Node oracle implicitly.
- [ ] Confirm `jmlx-jinja/build/reports/corpus-coverage.md` contains source, runtime-surface, and error-family evidence.
- [ ] Once the above is clean, run `./jmlx-jinja/release.sh` to build, run `check`, and publish the release bundle to Maven Central.
