# Release checklist

- [ ] Confirm JDK 21 and the exact Node version in `upstream/upstream-lock.json`, then run `./gradlew releaseVerification` from a clean source checkout. Use `-PreleaseVerificationAllowDirty` only for development; its report is not a release candidate.
- [ ] Read `build/reports/release-verification.md` and its JSON companion. The task prepares an isolated Gradle home, runs the candidate matrix offline, and compares two clean archive builds.
- [ ] The isolated verification repository is created per candidate; do not use `publishToMavenLocal`. The verifier publishes with `publishMavenPublicationToReleaseVerificationRepository`.
- [ ] Review `NOTICE`, `req/model-fixture-policy.md`, `CHANGELOG.md`, public Javadoc, and the generated POM.
- [ ] Review dependency updates with `./gradlew dependencyUpdates`; do not upgrade the pinned Node oracle implicitly.
- [ ] Confirm `build/reports/corpus-coverage.md` contains source, runtime-surface, and error-family evidence.
