# Phase 8.4 Portal capacity check

Reviewed 2026-11-16 at PR #101 merge `102e03ea43e508042ec4d43a61802b448a525267`.
Status: research complete; account capacity and signed-candidate measurement pending.
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
banner. Account for other releases in the same month and refresh immediately before publishing.

## Measure the signed candidate

After the runbook's preliminary-record and bundle validation succeeds, run this from the
candidate checkout. It reads the ZIP without extracting it and writes only local evidence.
The bundle digest must match the preliminary record and eventual final receipt.

```sh
python3 - <<'PY'
import hashlib
import json
import zipfile
from pathlib import Path

bundle = Path("build/releaseAll/jmlx-mega-bundle.zip")
digest = hashlib.sha256()
with bundle.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
with zipfile.ZipFile(bundle) as archive:
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

`zipEntryBytes` measures files inside the upload ZIP, including already-compressed JARs;
it does not expand the JARs' native payloads. `zipFileCount` includes signatures and checksum
sidecars if present. These are local inventory measurements, not authenticated Portal usage.
Confirm how Portal accounts for sidecars or generated files when reconciling the account.
The manifest requires five artifacts and five signatures per new coordinate: at least 90
files for the proposed nine-module tuple, before optional checksum sidecars.

Compare the inventory with remaining monthly capacity for the selected scope. The combined
flow submits one deployment; confirm its release-event accounting rather than equating nine
coordinates with nine events. The cited pages do not establish rejected-upload accounting;
obtain clarification if retries affect available capacity. Retain the snapshot, measurement,
candidate SHA/digest and accounting decision outside `build/` before running `clean`.
Close the checklist gate only when the exact signed bundle and current account evidence agree.
No signed candidate or account-specific capacity verdict exists yet.
