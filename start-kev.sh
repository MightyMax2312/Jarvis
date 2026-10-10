#!/bin/bash
# Starts the kev router.
#
# DEVICE CHOICE -- measured, not assumed.
#
# Default: on the GPU, exactly as it has always run. The router answers the real
# 30-verb payload in 72-81 ms warm there.
#
# KEV_ON_CPU=1 moves it to the CPU with CUDA_VISIBLE_DEVICES=-1 (a trick that
# needs no edit to the kev repo: kev.device.default_device() returns "cuda"
# whenever torch can see a GPU, so hiding the GPU is the whole mechanism). That
# was tried and measured, and it is a net loss on this box:
#
#   router latency     72-81 ms (GPU)  ->  2104-2586 ms (CPU fp32)
#                                          2257-2685 ms (CPU bf16)
#   talker residency   77% on GPU before -> 77% on GPU after (no change at all)
#   decode             25.6 -> 30.5 tok/s, but measured while thrashing
#   model load         0.65 s -> 40.7 s, prompt eval 993 -> 2032 ms
#   swap               1.3 GB used -> 2048/2048 MB, nothing free
#
# The router on CPU costs ~2.2 s per action turn, which is more than the ~1.5 s
# LLM turn the fast path exists to skip, so the fast path becomes a pessimisation
# on every command. And the VRAM it frees does not help: the talker stays at
# 2889 MB of its 3750 MB either way, because that split comes from the talker's
# own num_gpu setting, not from free VRAM. kev at fp32 also wants 2.5-2.7 GB of
# RAM, and KEV_DTYPE=bfloat16 did NOT halve it (2713 MB), so it cannot be made
# cheap there either.
#
# Kept as an option because the trade changes if the box ever gets more RAM and a
# smaller talker: `KEV_ON_CPU=1 ./start-kev.sh`.
#
# House style follows start-agent.sh and restart.sh: a PID file rather than
# pkill -f (whose pattern also matches the calling shell), and a wait for
# readiness rather than a blind sleep.
set -u
BASE="${JARVIS_BACKEND:-$(cd "$(dirname "$0")" && pwd)}"
PIDFILE=/tmp/kev.pid
LOG=/tmp/kev.log
RUN="${KEV_RUN:-$HOME/kev-0.5b}"
PORT=8000

if [ -f "$PIDFILE" ]; then
  OLD=$(cat "$PIDFILE")
  if kill -0 "$OLD" 2>/dev/null; then
    kill "$OLD" 2>/dev/null
    for _ in $(seq 1 20); do kill -0 "$OLD" 2>/dev/null || break; sleep 0.5; done
    kill -9 "$OLD" 2>/dev/null
    echo "stopped kev router $OLD"
  fi
fi

# Anything started by hand rather than by this script leaves no PID file. Match the
# full python command line, NOT just "kev.serve": the looser pattern also matches a
# shell whose command line merely mentions the string, which is how this loop once
# killed its own parent shell (the pgrep -f footgun the header warns about -- the
# job died with SIGTERM, exit status 15, and the services after it never started).
for pid in $(pgrep -f "python -m kev\.serve" 2>/dev/null); do
  [ "$pid" = "$$" ] && continue
  [ "$pid" = "$PPID" ] && continue
  kill "$pid" 2>/dev/null && echo "stopped kev router $pid (no pid file)"
done
sleep 1

cd "$BASE" || exit 1
if [ "${KEV_ON_CPU:-0}" = "1" ]; then
  # -1 is an invalid CUDA device id, so torch.cuda.is_available() is false and
  # kev lands on "cpu" (mps does not exist in WSL).
  DEVICE_ENV="CUDA_VISIBLE_DEVICES=-1"
  WHERE="cpu"
else
  DEVICE_ENV=""
  WHERE="gpu"
fi

# KEV_DTYPE is passed through if set (serve.py defaults to bfloat16 on a
# non-cpu device, which is the measured-fastest serving precision).
# shellcheck disable=SC2086
env $DEVICE_ENV setsid ./jarvis-env/bin/python -m kev.serve \
  --run "$RUN" --port "$PORT" > "$LOG" 2>&1 < /dev/null &
NEW=$!
echo "$NEW" > "$PIDFILE"
disown
echo "started kev router $NEW on $WHERE (log: $LOG)"

for _ in $(seq 1 90); do
  sleep 1
  if curl -sf -m 3 "http://127.0.0.1:$PORT/v1/models" > /dev/null 2>&1; then
    sleep 1
    DEV=$(curl -s -m 3 "http://127.0.0.1:$PORT/v1/models" | grep -o '"device":"[a-z]*"' | head -1)
    RSS=$(ps -o rss= -p "$NEW" 2>/dev/null | tr -d ' ')
    echo "kev ready after ~${SECONDS}s  $DEV  rss=$((RSS / 1024))MB"
    exit 0
  fi
  if ! kill -0 "$NEW" 2>/dev/null; then
    echo "kev died during startup:"
    tail -25 "$LOG"
    exit 1
  fi
done
echo "TIMED OUT waiting for kev"
tail -25 "$LOG"
exit 1
