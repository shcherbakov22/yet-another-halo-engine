#!/usr/bin/env bash
# Run a GPU command with the kernel log captured on both sides.
#
# usage: gpu_run.sh <tag> -- <command...>
#   env: YAH_GPU_LOG_DIR   where to write the log (default /home/q/yah-scratch)
#
# A bad dispatch on this target does not fail in-process: an access past an allocation hangs with no page fault.
# Then gfx_0.1.0 times out, MES stops answering msg=RESET, the GPU reset fails and the machine goes down.
# The kernel log is the only record: snapshot it before, follow it during the run (flushed as it writes), snapshot after.
# After a reboot, the previous boot's log is still in the journal:
#   doas journalctl -k -b -1 --no-pager | grep -iE 'amdgpu|timeout|reset|MES'
set -uo pipefail

LOGDIR="${YAH_GPU_LOG_DIR:-/home/q/yah-scratch}"
if [ "$#" -lt 3 ] || [ "$2" != "--" ]; then
  echo "usage: gpu_run.sh <tag> -- <command...>" >&2
  exit 2
fi
TAG="$1"; shift 2

# Refuse a driver binary older than its source: cmake --build does not build these, engine/build_hrx.sh does.
# A stale driver can launch a new HAL on an old grid; Loom drops clamps for the compiled grid, so it reads out of bounds.
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
for pair in "loom_forward_pp:engine/run/loom_forward_pp.cc" "hal_bench:engine/run/hal_bench.cc"; do
  bin="${pair%%:*}"; src="$ROOT/${pair#*:}"
  for arg in "$@"; do
    case "$arg" in
      */"$bin")
        if [ -f "$arg" ] && [ -f "$src" ] && [ "$src" -nt "$arg" ]; then
          echo "gpu_run: $arg is older than $src -- rebuild with engine/build_hrx.sh" >&2
          exit 3
        fi ;;
    esac
  done
done

# Refuse a shared GPU: another compute client, or another workload keeping it busy, skews every timing and counter.
# The desktop holds the render node and idles at a few percent busy.
KFD="$(doas fuser /dev/kfd 2>/dev/null | tr -s ' ')"
if [ -n "${KFD// /}" ]; then
  echo "gpu_run: /dev/kfd is open by pid(s)$KFD -- refusing a shared GPU" >&2
  exit 4
fi
BUSY=0
for _ in 1 2 3 4 5; do
  b="$(cat /sys/class/drm/card*/device/gpu_busy_percent 2>/dev/null | sort -n | tail -1)"
  BUSY=$((BUSY + ${b:-0}))
  sleep 0.2
done
if [ $((BUSY / 5)) -gt 20 ]; then
  echo "gpu_run: GPU is $((BUSY / 5))% busy before the run -- refusing a shared GPU" >&2
  exit 4
fi
mkdir -p "$LOGDIR"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="$LOGDIR/gpu-${TAG}-${STAMP}.dmesg.log"

if [ "$(id -u)" = "0" ]; then DMESG="dmesg"; else DMESG="doas dmesg"; fi

{
  echo "### $(date -Is) tag=$TAG"
  echo "### cmd: $*"
  echo "### uptime: $(uptime)"
  echo "### --- dmesg before ---"
} > "$LOG" 2>&1
$DMESG >> "$LOG" 2>&1

# Follower: its writes are visible in the file as they happen.
$DMESG -W > "$LOGDIR/.dmesg-w-${STAMP}.log" 2>&1 &  # -W: new messages only (-w replays the buffer)
FOLLOWER=$!

sync
"$@"
RC=$?

kill "$FOLLOWER" 2>/dev/null
wait "$FOLLOWER" 2>/dev/null

{
  echo "### --- exit code: $RC ---"
  echo "### --- dmesg follower ---"
  cat "$LOGDIR/.dmesg-w-${STAMP}.log" 2>/dev/null
  echo "### --- dmesg after ---"
} >> "$LOG" 2>&1
$DMESG >> "$LOG" 2>&1
rm -f "$LOGDIR/.dmesg-w-${STAMP}.log"

echo "gpu_run: exit=$RC log=$LOG"
# Scan only the follower section: the before/after snapshots repeat the whole boot's history, old warnings included.
DURING="$(sed -n '/^### --- dmesg follower ---/,/^### --- dmesg after ---/p' "$LOG")"
# "(no timeout)" ends every machine-check report line; it is not a GPU timeout.
FAULT="$(grep -iE 'timeout|GPU reset|MES failed|wedged|page fault|ring .* reset' <<<"$DURING" | grep -vic 'no timeout' || true)"
if [ "$FAULT" != "0" ]; then
  echo "gpu_run: *** $FAULT GPU fault line(s) logged during the run ***"
  grep -iE 'timeout|GPU reset|MES failed|wedged|page fault|ring .* reset' <<<"$DURING" | grep -vi 'no timeout' | tail -10
fi
exit $RC
