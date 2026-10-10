#!/bin/bash
# Bring the whole stack up, in the only order that survives this VM.
#
# WHY THE ORDER MATTERS -- measured, not guessed.
#
# The WSL VM has 7607 MB. Once up, the resident set is: kev 1.6 GB, backend 1.5 GB,
# agent ~1.3 GB, plus Ollama's host-side staging and page cache for the 3.75 GB
# talker. That fits. The peak does not: loading the talker while kev and the
# backend are already running has now OOM-killed the VM four times. The failure
# looks like this from Windows, and every process in the distro dies with it:
#
#     Catastrophic failure
#     Error code: Wsl/Service/E_UNEXPECTED
#
# Loaded FIRST, while nothing else is starting, the talker fits comfortably, and
# after that the other three come up without incident.
#
# THE DURABLE FIX is more memory for the VM. In %USERPROFILE%\.wslconfig:
#
#     [wsl2]
#     networkingMode=mirrored
#     memory=11GB
#     swap=8GB
#
# then `wsl --shutdown` once. 11 GB of a 15.35 GB host leaves ~4.3 GB for Windows;
# swap on disk means a spike pages instead of the VM being killed. Until that is
# applied, prefer this script over starting the services by hand.
#
# Backend memory note: server.py no longer imports torch or builds Kokoro at
# import time (both are needed only by the torch TTS path, and the agent asks for
# text only), which took it from ~3.1 GB to ~1.5 GB. That is 1.6 GB of the problem
# solved in code rather than with a config file.
set -u
BASE="${JARVIS_BACKEND:-$(cd "$(dirname "$0")" && pwd)}"
cd "$BASE" || exit 1
MODEL="${JARVIS_LLM_MODEL:-jarvis-llm}"

mem() { free -m | awk 'NR==2 {printf "  memory: %5s MB used, %5s MB available\n", $3, $7}'; }

echo "=== 1/4 talker into Ollama (first, while nothing else is loading) ==="
mem
curl -s -m 300 http://127.0.0.1:11434/api/chat \
     -d "{\"model\":\"$MODEL\",\"keep_alive\":-1}" -o /dev/null \
     -w "  preload: HTTP %{http_code} in %{time_total}s\n" || echo "  preload failed (Ollama down?)"
mem

echo "=== 2/4 router (kev, on the GPU) ==="
./start-kev.sh || exit 1
mem

echo "=== 3/4 backend ==="
./restart.sh || exit 1
mem

echo "=== 4/4 agent ==="
./start-agent.sh || exit 1
mem

echo "=== listening ==="
ss -ltn | grep -E ":8765|:8080|:8000|:11434"
echo "=== resident set ==="
ps -eo pid,rss,cmd | grep -E "kev.serve|server.py|piped.run" | grep -v grep \
  | awk '{printf "  pid=%-7s rss=%.2f GB  %s\n", $1, $2/1048576, $6}'
