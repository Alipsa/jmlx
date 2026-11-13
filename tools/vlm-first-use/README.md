# One-pair native VLM smoke

`.github/workflows/vlm-first-use-smoke.yml` is an opt-in macOS Apple Silicon
check of the current WP4 launcher against one old/new process pair per selected
Tier-B VLM. It runs on `self-macos-xl` (128 GiB), builds the historical
`46cb325` old-pin worktree and the selected current commit from source, and
copies the current worker source into the old worktree before building. The
launcher then checks jar freshness, pin identity, report status and measured
sample counts against real worker JVMs. This is a correctness and protocol
smoke, not a performance comparison or a third measurement session.

## Trigger

- On a same-repository PR that includes this workflow, add the
  `run-vlm-first-use-smoke` label. Remove and re-add it to rerun. This runs
  both VLMs (four worker processes total). If the label does not exist yet,
  a maintainer can create it once with
  `gh label create run-vlm-first-use-smoke --color 1d76db`.
- After the workflow reaches the default branch, use **Actions → WP4 one-pair
  native VLM smoke → Run workflow**, or run
  `gh workflow run vlm-first-use-smoke.yml --ref <branch> -f workloads=both`.
  The `workloads` choice can also select `qwen3vl` or `phi3v` to run one
  old/new pair for just that case. GitHub only exposes manual dispatch after
  the workflow file exists on the default branch.

The job downloads the digest-pinned public checkpoint into the same persistent
model cache used by the Tier-B workflow, removing unlisted files and symlinks
first. The default `both` selection may need to fetch roughly 12.6 GB if the
cache is empty. It stages each tree's own native pin, rebuilds both
`installDist` outputs with `--rerun-tasks --no-build-cache`, then runs one pair
with one warmup and two measured requests per worker. The session spec and
per-worker raw reports are uploaded as the `wp4-vlm-first-use-smoke-*`
artifact even if the run fails. The model weights and native binaries are not
uploaded.

The macOS native CI job does not run this launcher. A successful workflow run
establishes that this harness accepted real workers on the two staged pins;
it does not change the `unstable` status of the two historical WP4 cells.
