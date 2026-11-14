#!/bin/zsh
# Phase 8.2 WP4 session runner: brackets a benchmarkVlmFirstUse launcher run
# with tools/gpu-burn/hostrec.sh records (pre, 60 s poll, post). The host log
# is kept in ~/wp4-scratch/hostrec/<label>/ and copied into the session
# directory afterwards (the launcher requires out_root not to pre-exist, so
# the pre record cannot go into it directly).
set -eu
setopt pipe_fail
LABEL=${1:?usage: run-session.sh <label> <spec.json>}
SPEC=${2:?usage: run-session.sh <label> <spec.json>}
SESS=$(python3 -c "import json; print(json.load(open('$SPEC'))['out_root'])")
HREC=~/wp4-scratch/hostrec/$LABEL
HOSTREC=tools/gpu-burn/hostrec.sh
mkdir -p "$HREC"

HOSTREC_DIR="$HREC" "$HOSTREC" "pre:$LABEL"
(
  while true; do
    sleep 60
    HOSTREC_DIR="$HREC" "$HOSTREC" "poll:$LABEL"
  done
) &
POLLER=$!
trap 'kill $POLLER 2>/dev/null || true' EXIT

cd .
RC=0
./gradlew :jmlx-benchmarks:benchmarkVlmFirstUse --args="$SPEC" 2>&1 | tee "$HREC/launcher.log" || RC=$?
HOSTREC_DIR="$HREC" "$HOSTREC" "post:$LABEL"
if [ -d "$SESS" ]; then
  cp "$HREC/host.log" "$SESS/hostrec.log"
fi
echo "session exit=$RC hostlog=$HREC/host.log"
exit $RC
