#!/usr/bin/env zsh
#
# Read-only check for available updates to everything jmlx pins: the Gradle
# plugins and Maven Central dependencies declared in build.gradle/
# settings.gradle/gradle/libs.versions.toml, the Gradle wrapper itself, and
# the native mlx-metal wheel version pinned in scripts/bootstrap-native.sh.
#
# For mlx-metal specifically: mlx-c is versioned independently of MLX (mlx-c
# v0.6.0 pins MLX v0.31.2 via its own CMakeLists.txt GIT_TAG -- see
# req/initial-plan.md Decision 9), so a newer mlx-metal wheel does not imply
# any particular mlx-c tag. When a newer mlx-metal version is found, this
# script also searches mlx-c's tags (newest first) for the one whose
# CMakeLists.txt pins that exact version, and prints the matching commit --
# hand it to scripts/updateMlx.zsh to act on.
#
# Makes no changes anywhere. Needs network access; a resolution failure for
# any single check is reported as a warning, not a fatal error, so the rest
# of the report still runs.
#
# Usage: ./scripts/checkDependencies.zsh
set -uo pipefail

REPO_ROOT="${0:A:h}/.."
REPO_ROOT="${REPO_ROOT:A}"
BUILD_GRADLE="$REPO_ROOT/build.gradle"
SETTINGS_GRADLE="$REPO_ROOT/settings.gradle"
LIBS_TOML="$REPO_ROOT/gradle/libs.versions.toml"
WRAPPER_PROPS="$REPO_ROOT/gradle/wrapper/gradle-wrapper.properties"
BOOTSTRAP_SCRIPT="$REPO_ROOT/scripts/bootstrap-native.sh"
MLXC_REPO_URL="https://github.com/ml-explore/mlx-c.git"
MLXC_RAW_BASE="https://raw.githubusercontent.com/ml-explore/mlx-c/refs/tags"

log() { print -r -- ">>> $*"; }
warn() { print -r -- "WARNING: $*" >&2; }
# A bare ">>>" between sections -- not `log ""`, which would leave a
# trailing space after the arrow.
sep() { print -r -- ">>>"; }

for tool in curl git grep sed awk sort; do
  command -v "$tool" >/dev/null 2>&1 || { print -r -- "ERROR: required tool '$tool' not found on PATH" >&2; exit 1; }
done

typeset -i UPDATES_FOUND=0

# Extracts <release> (falling back to <latest>) from a Maven repo's
# maven-metadata.xml. Prints nothing on any failure -- callers treat an empty
# result as "could not resolve", not as an error, so one bad lookup doesn't
# abort the whole report.
latest_from_maven_metadata() {
  local url="$1" xml release
  xml="$(curl -fsSL "$url" 2>/dev/null)" || return 0
  release="$(print -r -- "$xml" | grep -oE '<release>[^<]+</release>' | head -1 | sed -E 's#</?release>##g')"
  if [[ -n "$release" ]]; then
    print -r -- "$release"
  else
    print -r -- "$xml" | grep -oE '<latest>[^<]+</latest>' | head -1 | sed -E 's#</?latest>##g'
  fi
}

# Prints a line and flips UPDATES_FOUND only when there's something to
# report; an up-to-date result prints nothing here at all -- the caller
# tracks that silence per section and prints a single "(all up to date)"
# line when every item in the section was silent (see the section bodies
# below). Returns 1 whenever it printed something, 0 when up to date, so a
# caller can accumulate "did this section have any news" with `|| news=1`.
#
# Empty $current (the pinned-version regex didn't match -- a parse failure,
# not "no update") and empty $latest (the upstream lookup failed) are each
# reported as their own warning, never compared against each other: doing
# so would either report a parse failure as an available update, or compare
# two empty strings and silently call it up to date.
report() {
  local name="$1" current="$2" latest="$3"
  if [[ -z "$current" ]]; then
    warn "$name: could not parse the currently pinned version -- the declaration format may have changed"
    return 1
  elif [[ -z "$latest" ]]; then
    warn "$name: could not resolve the latest version (network issue, or the coordinates moved)"
    return 1
  elif [[ "$current" == "$latest" ]]; then
    return 0
  else
    log "$name: $current -> $latest available"
    UPDATES_FOUND=1
    return 1
  fi
}

# Upper bound on how many of mlx-c's tags (newest first) to inspect before
# giving up. ~19 tags exist at the time of writing; this leaves headroom for
# years of releases while keeping the no-match case -- the common one, since
# mlx-c reliably lags a fresh MLX release -- from growing into an unbounded
# sweep of sequential network requests as more tags accumulate.
MAX_MLXC_TAGS_TO_SCAN=40

# Searches mlx-c's own tags (newest first, via sort -V, since mlx-c's version
# numbers do not track MLX's) for the tag whose CMakeLists.txt GIT_TAG pins
# exactly "v$1". Prints "<tag> <commit>" and returns 0 on a resolved match;
# prints nothing and returns 0 if no tag matches (or the scan cap is hit --
# either way, the caller's own "no tag found" message is accurate); prints
# nothing but returns 2 if a tag DID match and its commit could not be
# resolved -- that is a different failure than "no matching tag", so the
# caller must not report its generic message in that case (this function
# already warned specifically, and the generic message would flatly
# contradict it: mlx-c does have the release, only the SHA lookup broke).
find_mlx_c_commit_for_mlx_version() {
  local target_tag="v${1}" tag cmake_txt pinned_tag
  local commit peeled_output
  local -i peeled_status scanned=0
  while IFS= read -r tag; do
    [[ -z "$tag" ]] && continue
    scanned+=1
    if (( scanned > MAX_MLXC_TAGS_TO_SCAN )); then
      warn "mlx-c: gave up after scanning $MAX_MLXC_TAGS_TO_SCAN tags without a match -- raise MAX_MLXC_TAGS_TO_SCAN in this script if mlx-c now has more releases than that, or check https://github.com/ml-explore/mlx-c manually"
      return 0
    fi
    cmake_txt="$(curl -fsSL "$MLXC_RAW_BASE/${tag}/CMakeLists.txt" 2>/dev/null)" || continue
    pinned_tag="$(print -r -- "$cmake_txt" | grep -oE 'GIT_TAG v[0-9.]+' | head -1 | awk '{print $2}')"
    if [[ "$pinned_tag" == "$target_tag" ]]; then
      # mlx-c's tags are annotated, so the plain ref is a tag OBJECT, not a
      # commit -- "^{}" peels it to the commit it points at. Only fall back
      # to the unpeeled ref when the peel query itself succeeded and simply
      # found nothing (a lightweight tag, where the plain ref already is the
      # commit); a *failed* peel query (network hiccup) must not fall
      # through to that same branch, or a transient failure silently yields
      # a tag-object SHA that looks like a valid 40-hex-char commit to
      # updateMlx.zsh but can never match `git rev-parse HEAD`.
      peeled_output="$(git ls-remote "$MLXC_REPO_URL" "refs/tags/${tag}^{}")"
      peeled_status=$?
      commit=""
      if [[ $peeled_status -eq 0 ]]; then
        commit="$(print -r -- "$peeled_output" | awk '{print $1}')"
        [[ -z "$commit" ]] && commit="$(git ls-remote "$MLXC_REPO_URL" "refs/tags/${tag}" 2>/dev/null | awk '{print $1}')"
      fi
      # A match was found -- the empty-result case below is not "no tag
      # pins this version" (the caller's fallback message would be wrong
      # here: mlx-c *does* have the release, only this SHA lookup broke),
      # so it gets its own warning rather than falling through silently.
      if [[ -n "$commit" ]]; then
        print -r -- "$tag $commit"
        return 0
      fi
      warn "mlx-c: $tag pins MLX $target_tag, but resolving its commit failed (see the git error above) -- retry, or read the SHA off the tag manually at https://github.com/ml-explore/mlx-c/releases/tag/$tag"
      return 2
    fi
  done < <(git ls-remote --tags --refs "$MLXC_REPO_URL" 2>/dev/null | awk '{print $2}' | sed 's#refs/tags/##' | sort -rV)
  return 0
}

# --- Gradle plugins (Gradle Plugin Portal publishes a marker artifact per
# plugin id to its own Maven repo -- same maven-metadata.xml shape as Maven
# Central) ---------------------------------------------------------------

log "Checking Gradle plugins..."
typeset -i SECTION_NEWS=0

SPOTLESS_CURRENT="$(grep -oE "id 'com\.diffplug\.spotless' version '[^']+'" "$BUILD_GRADLE" | grep -oE "[0-9][^']*" | head -1)"
SPOTLESS_LATEST="$(latest_from_maven_metadata 'https://plugins.gradle.org/m2/com/diffplug/spotless/com.diffplug.spotless.gradle.plugin/maven-metadata.xml')"
report "Gradle plugin com.diffplug.spotless" "$SPOTLESS_CURRENT" "$SPOTLESS_LATEST" || SECTION_NEWS=1

FOOJAY_CURRENT="$(grep -oE "id 'org\.gradle\.toolchains\.foojay-resolver-convention' version '[^']+'" "$SETTINGS_GRADLE" | grep -oE "[0-9][^']*" | head -1)"
FOOJAY_LATEST="$(latest_from_maven_metadata 'https://plugins.gradle.org/m2/org/gradle/toolchains/foojay-resolver-convention/org.gradle.toolchains.foojay-resolver-convention.gradle.plugin/maven-metadata.xml')"
report "Gradle plugin org.gradle.toolchains.foojay-resolver-convention" "$FOOJAY_CURRENT" "$FOOJAY_LATEST" || SECTION_NEWS=1

[[ "$SECTION_NEWS" -eq 0 ]] && log "(all up to date)"
sep

# --- Maven Central dependencies ------------------------------------------

log "Checking Maven Central dependencies..."
typeset -i SECTION_NEWS=0

JUNIT_CURRENT="$(grep -E '^junit-jupiter[[:space:]]*=' "$LIBS_TOML" | grep -oE '"[^"]+"' | tr -d '"' | head -1)"
JUNIT_LATEST="$(latest_from_maven_metadata 'https://repo1.maven.org/maven2/org/junit/jupiter/junit-jupiter-api/maven-metadata.xml')"
report "org.junit.jupiter:junit-jupiter-api" "$JUNIT_CURRENT" "$JUNIT_LATEST" || SECTION_NEWS=1

# build.gradle's own comment on this line: the JVM Test Suite plugin's
# useJUnitJupiter(...) pins the bundled-platform version separately from
# libs.versions.toml's junit-jupiter accessor, and is kept in sync with it by
# hand -- so check that sync here rather than just the upstream version.
USE_JUNIT_CURRENT="$(grep -oE "useJUnitJupiter\('[^']+'\)" "$BUILD_GRADLE" | grep -oE "[0-9][^']*" | head -1)"
if [[ -n "$USE_JUNIT_CURRENT" && -n "$JUNIT_CURRENT" && "$USE_JUNIT_CURRENT" != "$JUNIT_CURRENT" ]]; then
  warn "build.gradle's useJUnitJupiter('$USE_JUNIT_CURRENT') has drifted from gradle/libs.versions.toml's junit-jupiter ($JUNIT_CURRENT) -- these are meant to be kept in sync by hand (see the comment above useJUnitJupiter in build.gradle)"
  SECTION_NEWS=1
fi

CHECKSTYLE_CURRENT="$(grep -oE "toolVersion = '[^']+'" "$BUILD_GRADLE" | grep -oE "[0-9][^']*" | head -1)"
CHECKSTYLE_LATEST="$(latest_from_maven_metadata 'https://repo1.maven.org/maven2/com/puppycrawl/tools/checkstyle/maven-metadata.xml')"
report "com.puppycrawl.tools:checkstyle" "$CHECKSTYLE_CURRENT" "$CHECKSTYLE_LATEST" || SECTION_NEWS=1

[[ "$SECTION_NEWS" -eq 0 ]] && log "(all up to date)"
sep

# --- Gradle wrapper --------------------------------------------------------

log "Checking Gradle wrapper..."
typeset -i SECTION_NEWS=0

WRAPPER_CURRENT="$(grep -oE 'gradle-[0-9][0-9.]*-bin\.zip' "$WRAPPER_PROPS" | head -1 | sed -E 's/^gradle-//; s/-bin\.zip$//')"
WRAPPER_LATEST="$(curl -fsSL 'https://services.gradle.org/versions/current' 2>/dev/null | grep -oE '"version"[[:space:]]*:[[:space:]]*"[^"]+"' | head -1 | sed -E 's/.*"([^"]+)"$/\1/')"
report "Gradle wrapper" "$WRAPPER_CURRENT" "$WRAPPER_LATEST" || SECTION_NEWS=1

[[ "$SECTION_NEWS" -eq 0 ]] && log "(all up to date)"
sep

# --- mlx-metal / mlx-c (native/, pinned in scripts/bootstrap-native.sh) ---

log "Checking mlx-metal (native/ dependency, pinned in scripts/bootstrap-native.sh)..."

MLX_METAL_CURRENT="$(grep -oE 'MLX_METAL_VERSION="[^"]+"' "$BOOTSTRAP_SCRIPT" | grep -oE '"[^"]+"' | tr -d '"' | head -1)"
MLX_C_COMMIT_CURRENT="$(grep -oE 'MLX_C_COMMIT="[^"]+"' "$BOOTSTRAP_SCRIPT" | grep -oE '"[^"]+"' | tr -d '"' | head -1)"
MLX_METAL_LATEST="$(curl -fsSL 'https://pypi.org/pypi/mlx-metal/json' 2>/dev/null | grep -oE '"version"[[:space:]]*:[[:space:]]*"[^"]+"' | head -1 | sed -E 's/.*"([^"]+)"$/\1/')"

if [[ -z "$MLX_METAL_LATEST" ]]; then
  warn "mlx-metal: could not resolve the latest version from PyPI"
elif [[ "$MLX_METAL_CURRENT" == "$MLX_METAL_LATEST" ]]; then
  log "mlx-metal: $MLX_METAL_CURRENT (up to date; mlx-c pinned at $MLX_C_COMMIT_CURRENT)"
else
  log "mlx-metal: $MLX_METAL_CURRENT -> $MLX_METAL_LATEST available"
  UPDATES_FOUND=1
  log "Looking for the mlx-c tag that pins MLX v$MLX_METAL_LATEST (mlx-c's own CMakeLists.txt GIT_TAG, req/initial-plan.md Decision 9)..."

  MATCH="$(find_mlx_c_commit_for_mlx_version "$MLX_METAL_LATEST")"
  FIND_STATUS=$?
  if [[ -n "$MATCH" ]]; then
    MATCHED_TAG="${MATCH%% *}"
    MATCHED_COMMIT="${MATCH#* }"
    log "Found: mlx-c $MATCHED_TAG ($MATCHED_COMMIT) pins MLX v$MLX_METAL_LATEST"
    log "Next: ./scripts/updateMlx.zsh $MATCHED_COMMIT"
  elif [[ "$FIND_STATUS" -ne 2 ]]; then
    # Exit status 2 means a tag DID match and already printed its own,
    # more specific warning above -- this generic message would contradict
    # it (see the function's own header comment).
    warn "mlx-c: no tag found whose CMakeLists.txt pins MLX v$MLX_METAL_LATEST -- mlx-c's own release may simply lag the new MLX version yet; check https://github.com/ml-explore/mlx-c manually"
  fi
fi
sep

if [[ "$UPDATES_FOUND" -eq 0 ]]; then
  log "All checked dependencies are up to date."
else
  log "Updates are available -- see above."
fi
