"""Compare the saved RC3 native publications with PR #115's approved byte pins."""

import argparse
import datetime
import hashlib
import io
import json
from pathlib import Path
import zipfile


APPROVAL_SHA256 = "ba210deade57f036aa52880ef066b1b21a32bd82bfc41ce7605d47ba544dc257"
BUNDLE_SHA256 = "139bff18f638136a34a1e61de942322a1d5523a821deca8b282b05b9ca3a500c"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--approval", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    approval_bytes = args.approval.read_bytes()
    if sha(approval_bytes) != APPROVAL_SHA256 or sha(args.bundle.read_bytes()) != BUNDLE_SHA256:
        parser.error("bundle or approval differs from the reviewed RC3 inputs")
    approval = json.loads(approval_bytes)
    artifacts = approval.get("localArtifactsReviewed", [])
    if len(artifacts) != 3:
        parser.error("approval must identify exactly three native archives")
    expected_archives = {"jmlx-native-macos-arm64-0.1.0" + suffix
                         for suffix in (".jar", "-sources.jar", "-javadoc.jar")}
    if {Path(pin["path"]).name for pin in artifacts} != expected_archives:
        parser.error("approval must identify runtime, sources and Javadoc archives")
    metadata = {}
    for pin in approval.get("approvedMetadata", []):
        name = Path(pin["path"]).name
        for suffix, expected_name in ((".pom", "native-pom.xml"),
                                      (".module", "native-module.json")):
            if name == expected_name:
                if suffix in metadata:
                    parser.error("duplicate native metadata pin: " + suffix)
                metadata[suffix] = pin
    if len(metadata) != 2:
        parser.error("approval must identify exactly one native POM and module metadata pin")
    report = {
        "comparedAt": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "bundleSha256": BUNDLE_SHA256,
        "approvalRecordSha256": APPROVAL_SHA256,
        "nativeArtifacts": [],
        "nativeMetadata": [],
        "javadocProvenance": {
            "approvedProducer": "unestablished in the historical approval evidence",
            "rc3Production": "root check restored native javadoc FROM-CACHE; bundle reused it "
                             "UP-TO-DATE; release JDK does not identify the cache producer",
        },
        "scope": "Exact archive/metadata/notice byte comparison; no historical JVM or legal inference",
    }
    prefix = "se/alipsa/jmlx-native-macos-arm64/0.1.0/"
    with zipfile.ZipFile(args.bundle) as bundle:
        for pin in artifacts:
            entry = prefix + Path(pin["path"]).name
            data = bundle.read(entry)
            if sha(data) != pin["sha256"] or len(data) != pin["bytes"]:
                parser.error("native archive differs: " + entry)
            legal = {}
            with zipfile.ZipFile(io.BytesIO(data)) as jar:
                encrypted = [item.filename for item in jar.infolist() if item.flag_bits & 1]
                if encrypted:
                    parser.error("encrypted native archive entries: " + entry)
                for name, expected in approval["approvedLocalLegalTexts"].items():
                    body = jar.read("META-INF/" + name)
                    if sha(body) != expected["sha256"] or len(body) != expected["bytes"]:
                        parser.error("legal text differs: " + entry + "/" + name)
                    legal[name] = {"bytes": len(body), "sha256": sha(body)}
            report["nativeArtifacts"].append({
                "entry": entry, "bytes": len(data), "sha256": sha(data),
                "matchesApprovedWholeJar": True, "encryptedEntries": encrypted,
                "legalTexts": legal,
            })
        for suffix, pin in sorted(metadata.items()):
            entry = prefix + "jmlx-native-macos-arm64-0.1.0" + suffix
            data = bundle.read(entry)
            if sha(data) != pin["sha256"] or len(data) != pin["bytes"]:
                parser.error("native metadata differs: " + entry)
            report["nativeMetadata"].append({
                "entry": entry, "bytes": len(data), "sha256": sha(data),
                "matchesApprovedLiteral": True,
            })
    if len(report["nativeArtifacts"]) != 3 or len(report["nativeMetadata"]) != 2:
        parser.error("comparison must contain three archives and two metadata records")
    report["javadocProvenance"]["archiveMatchesApprovedBytes"] = True
    report["allFiveApprovedNativeArtifactsMatch"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
