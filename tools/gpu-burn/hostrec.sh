#!/bin/zsh
# Per-run host-state recorder for the Phase 8.2 timing methodology.
#
# Committed from the issue #73 degraded-state session (2026-11-02, 16:07-16:19
# UTC, M5 Max / macOS 27.2); the committed output of it is
# req/benchmarks/phase7-5-wp6-issue73-host-state.txt.
#
# Usage:
#   HOSTREC_DIR=<session dir> ./hostrec.sh <label>
# appends one line to $HOSTREC_DIR/host.log, recorded before each run.
#
# Line format (UTC):
#   HH:MM:SS | <label> | load <1m> <5m> <15m> | therm-warnings=<N> | omlx=<CPU>% | memfree=<P>%
#
# Field notes:
#   load        1/5/15-min load averages (uptime).
#   therm-warnings  `pmset -g therm | grep -c 'warning level has been
#                   recorded'`. 2 is the no-warnings state: the thermal and
#                   performance "Note: No ... level has been recorded" lines
#                   match, the CPU-power note does not. A recorded warning
#                   replaces those notes ("... warning level set to N") and
#                   drops the count below 2. This is a secondary signal —
#                   keep full `pmset -g therm` outputs (session start and
#                   end) as the primary thermal record.
#   omlx        %CPU of the resident omlx-server process, "none" if absent.
#               (omlx-server is a Python process on this host — not a JVM —
#               but it is a resident GPU user and the leading contention
#               candidate for issue #73.)
#   memfree     free-memory fraction from `memory_pressure -Q`.

HOSTREC_DIR=${HOSTREC_DIR:?set HOSTREC_DIR to the session directory}
L=$1
OLPID=$(pgrep -x omlx-server | head -1)
OLCPU=$(ps -o %cpu= -p $OLPID 2>/dev/null | tr -d ' ')
THERM=$(pmset -g therm 2>/dev/null | grep -c 'warning level has been recorded')
MEMFREE=$(memory_pressure -Q 2>/dev/null | grep -o '[0-9]*%' | head -1)
LOAD=$(uptime | grep -o 'load averages:.*' | sed 's/load averages: //')
echo "$(date -u +%H:%M:%S) | $L | load $LOAD | therm-warnings=$THERM | omlx=${OLCPU:-none}% | memfree=$MEMFREE" >> $HOSTREC_DIR/host.log
