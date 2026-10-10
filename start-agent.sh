#!/bin/bash
# Starts the Pipecat agent. Uses a PID file -- never pkill -f, whose pattern
# matches the calling shell and kills the parent.
set -u
BASE="${JARVIS_BACKEND:-$(cd "$(dirname "$0")" && pwd)}"
PIDFILE=/tmp/piped.pid
LOG=/tmp/piped.log

if [ -f "$PIDFILE" ]; then
  OLD=$(cat "$PIDFILE")
  if kill -0 "$OLD" 2>/dev/null; then
    kill "$OLD" 2>/dev/null
    for _ in $(seq 1 20); do kill -0 "$OLD" 2>/dev/null || break; sleep 0.5; done
    kill -9 "$OLD" 2>/dev/null
    echo "stopped agent $OLD"
  fi
fi

cd "$BASE" || exit 1
setsid ./pipecat-env/bin/python -m piped.run "$@" > "$LOG" 2>&1 < /dev/null &
NEW=$!
echo "$NEW" > "$PIDFILE"
disown
echo "started agent $NEW (log: $LOG)"

for _ in $(seq 1 60); do
  sleep 1
  if grep -q "websocket listening" "$LOG" 2>/dev/null; then
    sleep 1
    echo "agent ready after ~${SECONDS}s"
    exit 0
  fi
  if ! kill -0 "$NEW" 2>/dev/null; then
    echo "agent died during startup:"
    tail -25 "$LOG"
    exit 1
  fi
done
echo "TIMED OUT waiting for agent"
tail -25 "$LOG"
exit 1
