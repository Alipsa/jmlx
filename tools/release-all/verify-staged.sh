#!/usr/bin/env bash
# Verifies staged release components against the plan manifest.
#
# releaseAllPrepareMaven invokes this (as `bash verify-staged.sh --manifest ... --staging ...
# --local ...`) only when the plan selected a static Maven component -- a component staged by
# `mvn install` rather than by a Gradle publish task. jmlx currently publishes only Gradle-built
# modules, so the plan never selects a static component and this script is not executed in the
# jmlx workflow; it exists because the plugin declares it as a required input file and because a
# future static component would be verified against the same manifest contract.
set -euo pipefail

manifest=
staging=
local=
while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest) manifest=${2:?}; shift 2 ;;
    --staging) staging=${2:?}; shift 2 ;;
    --local) local=${2:?}; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -f "$manifest" && -d "$staging" ]] || { echo 'verify-staged requires --manifest <file> and --staging <dir>' >&2; exit 2; }

groovy -e '
  def manifest = new groovy.json.JsonSlurper().parse(new File(args[0]))
  def staging = new File(args[1])
  def missing = []
  (manifest.selected ?: []).each { component ->
    def coordinate = component.groupId.toString().split("\.").join("/") + "/" + component.artifactId + "/" + component.version
    (component.files ?: []).each { String name ->
      ["", ".asc"].each { String suffix ->
        def file = new File(staging, coordinate + "/" + name + suffix)
        if (!file.isFile()) missing << (coordinate + "/" + name + suffix)
      }
    }
  }
  if (!missing.isEmpty()) {
    throw new IllegalStateException("Staged release files missing:\n" + missing.join("\n"))
  }
  println "verify-staged: all staged files present for " + (manifest.selected ?: []).size() + " component(s)"
' "$manifest" "$staging"
