# Release checklist

- [ ] Start from a clean checkout. Set `jmlx-jinja/build.gradle`'s version to the intended release version (not `-SNAPSHOT`), roll `CHANGELOG.md`'s `[Unreleased]` entries into a dated `## [x.y.z]` section, and commit those release-preparation changes. `releaseVerification` refuses a dirty tree.
- [ ] Confirm JDK 21 or newer and the exact Node version in `upstream/upstream-lock.json`, then run `./gradlew :jmlx-jinja:releaseVerification` from that clean release commit. Use `-PreleaseVerificationAllowDirty` only for development; its report is not a release candidate.
- [ ] Read `jmlx-jinja/build/reports/release-verification.md` and its JSON companion. The task prepares an isolated Gradle home, runs the candidate matrix offline, and compares two clean archive builds.
- [ ] The isolated verification repository is created per candidate; do not use `publishToMavenLocal` for this verification step. The verifier publishes with `:jmlx-jinja:publishMavenPublicationToReleaseVerificationRepository`. (The actual release below does use `publishToMavenLocal`, via the nexus-release-plugin's `bundle` task -- that is expected there, just not while verifying.)
- [ ] Review `NOTICE`, `req/model-fixture-policy.md`, `CHANGELOG.md`, public Javadoc, and the generated POM.
- [ ] Review dependency updates with `./gradlew :jmlx-jinja:dependencyUpdates`; do not upgrade the pinned Node oracle implicitly.
- [ ] Confirm `jmlx-jinja/build/reports/corpus-coverage.md` contains source, runtime-surface, and error-family evidence.
- [ ] Run `./jmlx-jinja/release.sh` to build, run `check`, and publish the release bundle to Maven Central.
- [ ] If releasing `jmlx-tokenizer` too, keep jinja at its released version, set and commit tokenizer's release version, then run `./jmlx-tokenizer/release.sh`. Its published POM must resolve the released jinja coordinate.
- [ ] After both releases, bump jinja and tokenizer to their next `-SNAPSHOT` versions, add jinja's fresh `[Unreleased]` changelog section, and commit. Do not bump jinja before the tokenizer release.
