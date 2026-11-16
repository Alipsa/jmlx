# Phase 8.4 minimum-host runner setup

Readiness inspection: 2026-11-16. Setup is pending; no candidate has been executed.
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
share that account's state. The proposed new account name is `jmlxrunner`, pending owner choice.

## Provision the dedicated account

1. Have a host administrator create a **Standard** account with its own home. Do not enable
   administrator access, passwordless sudo, iCloud synchronization or shared personal folders.
   Keep the account password in the host's normal password manager, outside release evidence.
2. Log in as that account. Provision Git and Python 3.11+ on its PATH. The workflow installs
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
   as that account. Do not register or run the worker as root or the owner.

Runner registration necessarily creates the runner's own authentication files. The account
must remain free of signing keys, Sonatype credentials, SSH keys and owner credentials.
The candidate job receives only read access; draft-asset access stays in its separate hosted
transfer job. The workflow uses checkout without persisted Git credentials and removes its
job-local Gradle home. Stop the runner after verification and review residual workspace and
temporary evidence before another candidate; retain reports outside disposable build output.

## Verify registration, then execute the candidate

Check the repository runner list and record the new runner ID, labels, account and date:

```sh
gh api repos/Alipsa/jmlx/actions/runners \
  --jq '.runners[] | {id,name,status,busy,labels:[.labels[].name]}'
```

Recheck Python, OS, architecture, chip/GPU and virtualization **as the dedicated account**.
Do not dispatch a placeholder candidate merely to test registration. Once readiness and tuple
review are complete, follow [the release runbook](../tools/release-all/README.md) to freeze,
sign and validate the real candidate, authorize draft transport, and dispatch
`release-candidate.yml` with its actual tag/SHA and both asset identities/digests.
Review the resulting `minimum-host.json`, log and measured runtime XML against that run.
Registration and host inspection alone cannot close exact-candidate acceptance.
