#!/bin/bash
# Phase 0 restart helper. Uses a PID file so we never pkill -f (which matches
# the calling shell's own command line and kills the parent).
set -u
BACKEND="${JARVIS_BACKEND:-$(cd "$(dirname "$0")" && pwd)}"
PIDFILE=/tmp/jarvis-backend.pid

if [ -f "$PIDFILE" ]; then
  OLD=$(cat "$PIDFILE")
  if kill -0 "$OLD" 2>/dev/null; then
    kill "$OLD" 2>/dev/null
    for _ in $(seq 1 20); do kill -0 "$OLD" 2>/dev/null || break; sleep 0.5; done
    kill -9 "$OLD" 2>/dev/null
    echo "stopped backend $OLD"
  fi
fi

rm -f "$BACKEND"/_kev_probe.py "$BACKEND"/_ollama_probe.py "$BACKEND"/_ollama_probe2.py \
      "$BACKEND"/_ollama_probe3.py "$BACKEND"/_ollama_probe4.py \
      "$BACKEND"/_tpl_probe.py "$BACKEND"/_phase0_verify.py \
      "$BACKEND"/_keepalive_probe.py
rm -rf "$BACKEND"/__pycache__
echo "cleaned scratch files"

cd "$BACKEND" || exit 1
setsid ./jarvis-env/bin/python server.py > /tmp/jarvis-server.log 2>&1 < /dev/null &
NEW=$!
echo "$NEW" > "$PIDFILE"
disown
echo "started backend $NEW (threshold=${JEV_CONFIDENCE_THRESHOLD:-0.70})"

for _ in $(seq 1 60); do
  sleep 2
  if curl -sf -m 5 http://127.0.0.1:8080/api/health > /dev/null 2>&1; then
    echo "healthy after ~$((SECONDS))s"
    exit 0
  fi
done
echo "TIMED OUT waiting for health"
tail -20 /tmp/jarvis-server.log
exit 1
