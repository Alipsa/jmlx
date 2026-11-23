# Phase 8.4 Portal capacity check

Reviewed 2026-11-16 at PR #101 merge `102e03ea43e508042ec4d43a61802b448a525267`.
Revised 2026-11-17 (review rounds 1 and 2).
Status: RC2 historical bundle measurement complete; receipt revalidation blocked by missing
candidate inputs. Current account capacity remains pending.
This procedure does not authorize publication or close the native-license gate.

## Current official sources

The [upload documentation](https://central.sonatype.org/publish/publish-portal-upload/)
allows a ZIP up to 1 GB containing multiple components. Our existing bundle verifier caps
the actual ZIP at 1,000,000,000 bytes. No separate per-file cap was established from this page.

The [publishing limits](https://central.sonatype.org/publish/maven-central-publishing-limits/)
track monthly organization-wide file count, release size and release count. Current thresholds
come from the account, rather than a universal ten-deployment allowance. The public percentile
table is context, not proof of this account's entitlement. The policy also distinguishes
community publishing from commercial-nature artifacts; the latter require Publisher Pro
from 2026-10-01 independently of volume. Confirm the applicable account status before release.

The [Usage Center documentation](https://central.sonatype.org/publish/publish-portal-usage-center/)
defines size as published-file bytes, files individually, and release events as distinct
publish operations. Select the organization containing `se.alipsa` at
[Publishing → Usage Center](https://central.sonatype.com/publishing/usage).
Retain a dated non-secret snapshot of current usage and limits for all three metrics,
organization/namespace scope, plan, adjusted-limit or exemption expiry, and any grace/enforcement
banner. The limits page states that exceeding a threshold starts a 30-day grace period that
resets "after 2 consecutive months of being below limits", so read any grace/enforcement
banner together with its snapshot, not in isolation. Account for other releases in the same
month and refresh immediately before publishing.

## Measure the signed candidate

After the runbook's preliminary-record and bundle validation succeeds, run this from the
repository root of the candidate checkout, passing the runbook's preliminary record
(`build/releaseAll/preliminary-<invocation>.json`). It reads the ZIP without extracting it
and writes only local evidence. The script asserts the bundle digest matches the preliminary
record; a stale or rebuilt bundle fails before anything is measured, and the eventual final
receipt repeats the comparison.

```sh
RECORD='build/releaseAll/preliminary-<invocation>.json'   # set to the runbook's record
python3 - "$RECORD" <<'PY'
import hashlib
import json
import sys
import zipfile
from pathlib import Path

bundle = Path("build/releaseAll/jmlx-mega-bundle.zip")
record = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
digest = hashlib.sha256()
with bundle.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
if record.get("bundleSha256") != digest.hexdigest():
    raise SystemExit("bundle digest differs from the preliminary record")
with zipfile.ZipFile(bundle) as archive:
    if archive.testzip() is not None:
        raise SystemExit("bundle ZIP failed CRC verification")
    files = [entry for entry in archive.infolist() if not entry.is_dir()]
    if not files or len({entry.filename for entry in files}) != len(files):
        raise SystemExit("expected non-empty bundle without duplicate files")
    report = {
        "bundleSha256": digest.hexdigest(),
        "bundleBytes": bundle.stat().st_size,
        "withinConservativeUploadLimit": bundle.stat().st_size <= 1_000_000_000,
        "zipFileCount": len(files),
        "zipEntryBytes": sum(entry.file_size for entry in files),
        "files": [{"name": entry.filename, "bytes": entry.file_size}
                  for entry in sorted(files, key=lambda entry: entry.filename)],
    }
Path("build/releaseAll/portal-bundle-measurement.json").write_text(
    json.dumps(report, indent=2) + "\n", encoding="utf-8")
if not report["withinConservativeUploadLimit"]:
    raise SystemExit("bundle exceeds conservative Portal upload limit")
PY
```

`zipEntryBytes` reports the sizes recorded in the ZIP directory for the files inside the
upload ZIP, including already-compressed JARs; it does not expand the JARs' native payloads,
and the script's `testzip` verifies the archive's recorded CRCs before either number is
trusted. `zipFileCount` includes the `.asc`, `.md5` and `.sha1` sidecars the pinned
nexus-release-plugin generates, and Sonatype's publishing-limits FAQ (its "Why does file
count include signatures, checksums, source jars, javadocs, and related files?" entry)
states these files "are already reflected in the usage data the thresholds are based on" —
they are not optional. The plugin's `BundleTask` computes MD5 and SHA-1 for every artifact
and its `BundleValidator` checks `.asc`, `.md5` and `.sha1` entries; the 8.1 combined bundle
(nexus-release-plugin 2.5.1; its coordinates are published on Central and its composition is
recorded in `req/phase8-1-release-notes.md`) carries exactly 30 files per coordinate — five
artifacts and five signatures, each with a `.md5` and a `.sha1`, including checksums of the
`.asc` files themselves. Expect 270 files for the proposed nine-module tuple and confirm the
exact figure from the 2.6.0 measurement. These are local inventory measurements, not
authenticated Portal usage.

Compare the inventory with remaining monthly capacity for the selected scope. The combined
flow submits one deployment; confirm its release-event accounting rather than equating nine
coordinates with nine events. The cited pages do not establish rejected-upload accounting;
obtain clarification if retries affect available capacity. Retain the snapshot, measurement,
candidate SHA/digest and accounting decision outside `build/` before running `clean`.
Close the checklist gate only when the exact signed bundle and current account evidence agree.

## RC2 measurement (2026-11-22)

The [RC2 review](phase8-4-rc2-final-review.md) identifies the historically verified signed ZIP
and receipt. Missing candidate checkout/producer inputs now block receipt revalidation,
acceptance and publication. The inventory below remains evidence for the retained RC2 ZIP;
any replacement bundle requires a new measurement and capacity assessment against its digest.
Its [retained inventory](plans/evidence/rc2-final-review-2026-11-22/portal-bundle-measurement.json)
records 106,730,874 compressed bytes, 270 files and 107,448,768 bytes inside the upload ZIP.
The digest matches the preliminary record, ZIP CRCs pass and filenames are unique. The
measurement passes the existing conservative upload cap; it is not a capacity verdict.

The three official sources above were rechecked on 2026-11-22. They still direct publishers
to the authenticated Usage Center for current organization/scope limits and usage. The
account snapshot, plan/classification confirmation and any adjusted limits or enforcement
conditions remain pending. No account-specific capacity verdict or publication authorization
has been recorded. Refresh that snapshot immediately before any authorized publication.
