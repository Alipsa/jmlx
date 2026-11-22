# Phase 8.4 minimum-host runner setup

Provisioned and connectivity checked: 2026-11-16. Account revalidated: 2026-11-22.
No candidate has been executed.
This implements the runner prerequisites in
[the plan](plans/phase8-4-plan.md) and
[release workflow](../.github/workflows/release-candidate.yml).

## Available host and existing runner

Read-only local inspection found:

| Item | Observed value |
| --- | --- |
| OS | macOS 27.2.1, build 26B131 |
| Architecture | arm64 |
| CPU and GPU model | Apple M5 Max |
| `kern.hv_vmm_present` | `0` |
| Existing repository runner | `local-m5-max`, runner ID 22, online and idle at inspection |
| Existing labels | `self-hosted`, `macOS`, `ARM64`, `self-macos-xl` |
| Existing service account | `ghrunner`, UID 502, not in the admin group |

These observations establish an available host at the declared 27.2.x support floor.
They do not establish Metal execution of a signed candidate. The runner's online/idle
status is transient. No machine serial number or registration credential is retained here.

The existing runner serves `tier-b.yml` and `vlm-first-use-smoke.yml`, including the latter's
same-repository PR-label trigger. It is not the dedicated manual-only release runner required
by the plan. Leave its registration and labels intact. Use a separate standard macOS account
and a separate runner installation; two installations under the same account would still
share that account's state. The owner approved creating `jmlxrunner`; it is now provisioned
with UID 503 and a private (`0700`) home. At provisioning its runner was
`jmlx-release-minimum-m5-max`, ID 23, with labels `self-hosted`, `macOS`, `ARM64`,
`jmlx-release-minimum`. The original runner's registration and labels were preserved.

## Completed setup and operation

The [retained evidence](plans/evidence/minimum-host-runner-2026-11-16/setup.json)
records host measurements made as `jmlxrunner`, account permission checks, runner package
identity and connectivity. GitHub runner v2.338.2 was downloaded from its official release
and SHA-256 checked before installation. The new listener connected and reached
`Listening for Jobs` under `jmlxrunner`; GitHub also reported it online and idle.
It was then stopped and GitHub reported it offline. No job was dispatched.

The generated account password was not retained. The account is used through a non-root
service identity, rather than interactive password login; an administrator can reset its
password if interactive access is later needed. No owner profiles or credentials were copied.
Permission tests as the new account confirmed it cannot read the owner's Gradle properties
or traverse the owner's SSH and signing directories. Its own corresponding credential
locations are absent. These are scoped filesystem checks, not a machine-wide secret scan.

The custom [launchd definition](plans/evidence/minimum-host-runner-2026-11-16/com.alipsa.jmlx-release-runner.plist)
is installed at `/Library/LaunchDaemons/com.alipsa.jmlx-release-runner.plist`.
The connectivity check used the documented `launchctl bootstrap` and `kickstart` commands,
including the configured log directory; it did not use `./run.sh`.
The service launches GitHub's `runsvc.sh` with `UserName=jmlxrunner` and an explicit account
home. `runsvc.sh` replaces launchd's PATH with the configuration-time `.path`; the listener
also restores `.env`. Inspection as the account found `.path` equal to the deliberately
supplied clean PATH and `.env` empty. Their digests are retained in
[the review evidence](plans/evidence/minimum-host-runner-2026-11-16/review.json).
Neither contains the owner's home or an inherited `JAVA_HOME`. Configuration used `env -i`,
not the owner's shell environment. Both `RunAtLoad` and `KeepAlive` are disabled.
At the 2026-11-16 check, the definition remained installed but unloaded.
The 2026-11-22 state is recorded below; automatic startup remains disabled.
GitHub's own runner authentication files are private to the account. Reinspect these saved
files before each release window; the plist alone does not control the effective environment.

Python and Git resolve to **owner-managed Homebrew** installations under `/opt/homebrew`.
The review records selected/resolved executable paths, owners, groups, modes and versions.
This setup isolates release-account credentials; it does not isolate the toolchain from
the owner who can modify Homebrew. Upgrades can change executable paths and versions.
Resolve and record them again before each window and review any change before acceptance.

The owner's Gradle properties file is now mode `0600`, rather than relying solely on
parent-directory permissions. The account still cannot read or write it. The sharing group
`com.apple.sharepoint.group.1` includes `everyone` through a nested group, explaining the
new account's membership. Inspection found a guest share configured; `smbd` was not loaded
at inspection. Public evidence retains only that boolean, without share identity or write mode.
No new share was created for `jmlxrunner`. This is not proof that File Sharing is globally
disabled or that all network sharing access has been audited; keep the private account home
and recheck sharing at window time.

## Account revalidation (2026-11-22)

The owner ran the read-only probe as `jmlxrunner`. The retained
[probe script](plans/evidence/minimum-host-runner-2026-11-22/account-probe.py) and the evidence's
method fields identify the command, script hash and source commit. `recordedAt` dates evidence
assembly and supplemental inspection; the original probe recorded only its observation date.
The
[retained account measurements](plans/evidence/minimum-host-runner-2026-11-22/account-probe.json)
confirm UID 503, a `0700` home and no admin membership on Apple M5 Max ARM64 macOS 27.2.1
(build 26B131; `kern.hv_vmm_present=0`). The account cannot read or write the owner's
Gradle properties or traverse the owner's SSH and signing directories. These scoped
permission checks do not establish machine-wide secret isolation.

Local runner ID 23 remains `jmlx-release-minimum-m5-max`, listener 2.338.2. Its listener
SHA-256 is `732456cbedb8c09870b82a63c05ba5505ad5cc3fb536ee570879a0af1fdb5087`.
The saved `.env` is empty; `.path` is 63 bytes. Neither contains the owner's home;
versions, resolved tool paths, modes and environment hashes are retained in the probe.
Python 3.15.2 and Git 2.56.1 still resolve through owner-managed Homebrew.

A read-only GitHub registration check confirmed ID 23 offline and idle, with labels
`self-hosted`, `macOS`, `ARM64` and `jmlx-release-minimum`. A local launchd inspection
found the service loaded, with state `not running`, active count zero and no starts.
The host booted on 2026-11-22 at 10:14:17 local time, after the unloaded November 16 check.
Loading the plist from `/Library/LaunchDaemons` at boot is consistent with this state;
no historical launchd event log was examined to establish the cause. `RunAtLoad` and
`KeepAlive` remain false, and the service reports no starts. Supplemental evidence records
these observations, the chip model and the GitHub registration snapshot separately from
measurements made as `jmlxrunner`.
No listener was started or job dispatched during this revalidation. If still loaded at
window time, skip `bootstrap` and use `kickstart` only after the window checks pass.

The account's `com.apple.sharepoint.group.1` membership comes through nested `everyone`;
it is the Public-folder share group, not an account-specific grant. A later read-only
`sharing -l` check found the owner's Public-folder SMB share enabled with guest access
and read-write permissions. Its state at the earlier account probe time was not established.
This exposure remains an open sharing-review item: resolve it or explicitly assess it before
the window, then recheck sharing and private-home isolation. No sharing settings were changed.

The queued-job review, sharing review and immediate pre-window revalidation remain required.
This probe does not close the readiness gate or establish signed-candidate Metal execution.

## Starting and stopping the reviewed window

For a reviewed release verification window, a host administrator can start it with:

```sh
sudo launchctl bootstrap system /Library/LaunchDaemons/com.alipsa.jmlx-release-runner.plist
sudo launchctl kickstart system/com.alipsa.jmlx-release-runner
```

After the job finishes, stop and unload it with:

```sh
sudo launchctl bootout system/com.alipsa.jmlx-release-runner
```

`bootout` unloads the service for the current boot. With the plist still installed in
`/Library/LaunchDaemons`, launchd can load it again after reboot. Disabled `RunAtLoad` and
`KeepAlive` prevent automatic startup; they do not keep the definition unloaded.
These administrative commands manage launchd; the runner process itself runs as UID 503.
Recheck account isolation and the candidate workflow before each window. Exact-bundle Metal
execution, measured Java 21/25 consumer runtimes and acceptance reports remain outstanding.

## Provisioning reference

1. Have a host administrator create a **Standard** account with its own home. Do not enable
   administrator access, passwordless sudo, iCloud synchronization or shared personal folders.
   For interactive use, keep its password in the host's normal password manager, outside
   release evidence. The current service-only setup uses an unretained generated password.
2. Run setup as that account. Provision Git and Python 3.11+ on its PATH. The workflow installs
   Java 21 and 25 through `actions/setup-java`; it does not need the owner's JDK or Gradle home.
   Provide network access for GitHub Actions, the Gradle wrapper and Maven dependencies.
3. Verify the account is non-root and non-admin with `id -u` and `id -Gn`. Confirm it cannot
   read the owner's Gradle properties, SSH private keys or signing key material. Check access
   using file permission tests, without printing their contents. Do not copy credentials or
   shell profiles from the owner's account. Any failed isolation check leaves readiness open.
4. In the new account, use the current download and checksum commands from
   [Alipsa/jmlx → Settings → Actions → Runners → New self-hosted runner](https://github.com/Alipsa/jmlx/settings/actions/runners/new),
   selecting **macOS / ARM64**. Install into that account's own runner directory.
   [GitHub's registration guide](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/add-runners)
   explains the repository-admin requirement and short-lived registration token. Enter the token
   locally; do not paste it into a PR, report or assistant message.
5. Register the runner as `jmlx-release-minimum-m5-max`, retaining the default
   `self-hosted`, `macOS`, `ARM64` labels and adding **only** `jmlx-release-minimum` as its custom
   label. Do not give it `self-macos-xl`. Restrict use to the reviewed candidate workflow;
   labels route jobs but are not an authorization boundary. Preserve repository approval
   controls and review changes that could route PR code onto the release runner.
6. Start it only for the reviewed release verification window. For an interactive session,
   run `./run.sh` as the dedicated account. If using a service, follow
   [GitHub's macOS service instructions](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/configure-the-application?platform=mac)
   as that account, or use the custom launchd definition above. Do not register or run the
   worker as root or the owner.

Runner registration necessarily creates the runner's own authentication files. The account
must remain free of signing keys, Sonatype credentials, SSH keys and owner credentials.
The candidate job receives only read access; draft-asset access stays in its separate hosted
transfer job. The workflow uses checkout without persisted Git credentials and removes its
job-local Gradle home. Stop the runner after verification and review residual workspace and
temporary evidence before another candidate; retain reports outside disposable build output.

## Verify registration, then execute the candidate

GitHub [automatically removes persistent runners](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/remove-runners)
after more than 14 days without a connection (ephemeral registrations have a one-day limit).
The 2026-11-16 connection therefore does not reserve ID 23 beyond approximately 2026-11-30.
Keep the worker stopped; plan to re-register at window time if the registration has expired,
using GitHub's removal/re-registration instructions and a fresh short-lived token.
Record the replacement ID and repeat connectivity/identity checks. There is no periodic
keepalive job. Historical setup evidence and the checked support-floor decision remain historical.

Check the repository runner list **before starting** and compare its ID with both the last
reviewed window record and the local `.runner` agent ID. A missing or different ID requires
re-registration or review, not silently treating ID 23 as current. Record labels, account and date:

```sh
gh api repos/Alipsa/jmlx/actions/runners \
  --jq '.runners[] | {id,name,status,busy,labels:[.labels[].name]}'
```

Automatic runner updates remain enabled. The archive digest records the **initial install**,
not a permanently pinned running binary. GitHub documents its
[update behavior](https://docs.github.com/en/actions/reference/runners/self-hosted-runners#runner-software-updates-on-self-hosted-runners).
As the dedicated account, record `./bin/Runner.Listener --version` and its executable SHA-256
before starting and again after connection or any update, before dispatch. An update may
still occur on job assignment: also retain the actual run's **Set up job** log and compare
its `Current runner version` line with the pre-dispatch version. If it differs, re-measure
the installed listener and digest and review the updated version before accepting the report.
The earlier binary digest must not be attributed to the updated runner; if the job-time binary
cannot be identified, repeat verification on an identified version. No actual candidate run
exists yet, so no job-time version comparison has been performed.
Review version changes alongside the saved environment and tool paths. Persistent registration
and manual workspace cleanup remain the chosen setup; ephemeral registration alone would not wipe files.

List active workflow runs and their jobs matching the release label before bringing the
runner online. The following read-only commands paginate both collections and list matching
unfinished jobs that the API currently exposes. They also list approval-required runs even
when jobs or labels are not yet exposed. Jobs blocked on `needs:` may not yet be visible with
labels; this check does not claim to enumerate those jobs. Review the active workflows and
their possible downstream jobs before starting the runner, and do not approve fork runs
during the release window.

```sh
python3 - <<'PY'
import json
import subprocess

def gh(*args):
    return subprocess.check_output(["gh", "api", "--paginate", *args], text=True, timeout=60)

run_ids = set()
for status in ("queued", "in_progress", "waiting", "requested", "pending", "action_required"):
    run_ids.update(gh(f"repos/Alipsa/jmlx/actions/runs?status={status}",
                      "--jq", ".workflow_runs[].id").split())
for run_id in sorted(run_ids, key=int):
    run = json.loads(gh(f"repos/Alipsa/jmlx/actions/runs/{run_id}"))
    approval_required = (run.get("status") == "action_required" or
                         run.get("conclusion") == "action_required")
    if approval_required:
        print(json.dumps({key: run[key] for key in ("id", "path", "head_branch", "head_sha", "event")}))
        print("Approval required: review this run even if job labels are not exposed.")
    jobs = gh(f"repos/Alipsa/jmlx/actions/runs/{run_id}/jobs", "--jq",
              '.jobs[] | select(.status != "completed") | '
              'select(.labels | index("jmlx-release-minimum")) | {id,name,status,labels}')
    if jobs.strip() and not approval_required:
        print(json.dumps({key: run[key] for key in ("id", "path", "head_branch", "head_sha", "event")}))
    if jobs.strip():
        print(jobs)
PY
```

Investigate every matching job; leave the worker offline if anything unexpected is queued.
Both jobs in the reviewed `release-candidate.yml` now require `github.ref == 'refs/heads/main'`.
Dispatch explicitly from `main`; review that workflow commit and the candidate tag separately.
The minimum-host report records workflow ref, event and `github.workflow_sha`, and the final
receipt checks their required form. Compare the recorded SHA with the actual reviewed GitHub
run; these local fields are operator attestations, not authenticated remote provenance.

A main-branch guard protects this workflow's normal dispatch path. Another workflow, or
an edited branch copy that removes the guard, can still request the runner's labels.
Queue inspection is a snapshot, not an access-control boundary. This repository-level runner
therefore trusts repository writers during the window; preserve fork approval controls,
review workflow changes, and keep the online window short. A server-enforced workflow
allowlist would require an appropriately restricted organization runner group; this setup
does not claim that restriction is installed.

Recheck Python, OS, architecture, chip/GPU and virtualization **as the dedicated account**.
Do not dispatch a placeholder candidate merely to test registration. Once readiness and tuple
review are complete, follow [the release runbook](../tools/release-all/README.md) to freeze,
sign and validate the real candidate, authorize draft transport, and dispatch
`release-candidate.yml` with its actual tag/SHA and both asset identities/digests.
Review the resulting `minimum-host.json`, log and measured runtime XML against that run.
Registration and host inspection alone cannot close exact-candidate acceptance.
