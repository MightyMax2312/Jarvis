# Jarvis

A voice assistant that runs entirely on one machine — no cloud, no wake word. It
listens, transcribes, decides whether what it heard is a *command* or a
*conversation*, acts on the Windows desktop when asked, and speaks the answer back
in its own voice.

Everything happens across the Windows/WSL2 boundary: the WSL side owns the models,
the Windows side owns the microphone, the speakers, the screen, the mouse and the
keyboard.

```
 mic ──► STT ──► router ──┬─► fast path: run the verb, speak the result
 (Windows client)         │      (0.5B model, no LLM turn)
                          └─► 4B vision LLM ──► TTS ──► speaker
                                  │
                                  └─► tool call ──► Windows client executes it
```

## Components

| piece | role | runtime |
|---|---|---|
| `server.py` | FastAPI: Parakeet STT (`/api/stt`), health, and `/api/command` (router verdict, screen description) | `jarvis-env` |
| `piped/` | the real-time voice pipeline (pipecat): VAD → STT → fast path → LLM → TTS | `pipecat-env` |
| `piped/vocab.py` | the desktop-action vocabulary, imported by **both** runtimes | stdlib only |
| `piped/tools.py` | the LLM's only way to touch the machine: a verb allowlist and the tool schemas | `pipecat-env` |
| `start-all.sh` | brings the whole stack up in the order that survives a small VM | bash |
| Windows client | mic, speaker, screen capture, keyboard/mouse, tool endpoint on `:8766` | Windows Python |

Two models are served by Ollama: a **4B vision talker** (`jarvis-llm`, built from
`Modelfile.jarvis-llm`) and a **0.5B prefill-only router** (the `kev` project) that
answers "which of these verbs did the user mean?" in ~80 ms. The router is what
makes commands feel instant: when it is confident about a verb that needs no
argument, the turn skips the 4B model entirely.

## Why it is built this way (all measured on the reference machine)

Reference hardware: RTX 3050 Laptop (4096 MiB VRAM), Ryzen 7 5800H, 15 GB host RAM,
WSL2 with 11 GB of it.

| decision | measurement |
|---|---|
| Router on the **GPU** | 72–84 ms warm. Moving it to the CPU to free VRAM measured 2104–2685 ms and freed nothing the talker claimed, so it moved back |
| `TextAggregationMode.TOKEN` | with punctuation preserved, pipecat still regroups tokens into sentences; `SENTENCE` would only add latency |
| Punctuation must reach the TTS | the old filter deleted it (`'…as JARVIS.'` → `'…as JARVIS'`), which stripped sentence-final intonation *and* stopped pipecat finding sentence boundaries, so first audio came late |
| VAD `stop_secs=0.4` | at 0.2 the VAD closed the utterance on an ordinary pause, producing two transcripts ~1 s apart; the model answered the first half before hearing the rest. Fixing it cut time-to-first-audio from 2.55 s to 1.60 s |
| Talker preloaded before the socket opens | first-turn load measured 18.5 s cold against 0.33 s warm |
| `torch`/Kokoro lazily loaded in `server.py` | backend RSS 3.1 GB → 1.49 GB, and it never loads them at all when the caller asks for text only |

Turn latency on the reference machine: LLM inference p50 **0.8 s**, speech synthesis
p50 **0.86 s**, STT p50 **0.27 s**, turn detection p50 **0.015 s**.

## Setup

1. **Two virtualenvs** (they have incompatible dependency sets on purpose — the
   pipeline has no torch, the backend has no pipecat):

   ```bash
   python3 -m venv jarvis-env && ./jarvis-env/bin/pip install \
     fastapi uvicorn requests numpy scipy soundfile sherpa-onnx kokoro torch
   python3 -m venv pipecat-env && ./pipecat-env/bin/pip install \
     "pipecat-ai[kokoro,ollama,websocket,silero,smart-turn]"
   ```

2. **STT model** — Sherpa-ONNX Parakeet TDT 0.6B v3 (int8) into
   `models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/`.

3. **Talker** — see `piped/make_modelfile.py`, which writes `Modelfile.jarvis-llm`;
   then `ollama create jarvis-llm -f Modelfile.jarvis-llm`.

4. **Memory** — the stack peaks near 9 GB. In `%USERPROFILE%\.wslconfig`:

   ```ini
   [wsl2]
   networkingMode=mirrored
   memory=11GB
   swap=8GB
   ```

   `networkingMode=mirrored` is what lets the Windows client reach `127.0.0.1`
   inside WSL. Without enough memory the VM is OOM-killed while the models load.

5. **Run it:**

   ```bash
   ./start-all.sh
   ```

   Order matters: the talker is loaded *first*, then the router, the backend and
   the agent. Loading the talker while the others are already resident is what
   OOM-killed the VM repeatedly.

## Configuration

| variable | default | meaning |
|---|---|---|
| `JARVIS_BACKEND` | this directory | where the scripts work |
| `JARVIS_LLM_MODEL` | `jarvis-llm` | Ollama model name |
| `JARVIS_STT_URL` | `http://127.0.0.1:8080/api/stt` | STT endpoint |
| `JARVIS_OLLAMA_BASE` | `http://127.0.0.1:11434/v1` | LLM endpoint |
| `JARVIS_CLIENT_HOST` | `127.0.0.1:8766` | desktop client `host:port` |
| `JARVIS_GATEWAY_FALLBACK` | *(unset)* | set when WSL is in NAT mode rather than mirrored |
| `JARVIS_IDLE_TIMEOUT` | `0` (off) | seconds of silence before the agent shuts itself down |
| `JARVIS_FASTPATH` | `1` | `0` routes every turn through the LLM, for A/B comparison |
| `JARVIS_SPY_LOG` | *(unset)* | also write the session log to this file |

## Security posture

The agent can type, click, lock the workstation and read the screen. Every service
therefore binds `127.0.0.1` only, and the tool layer validates every verb against a
fixed allowlist, so a hallucinated action becomes a refusal rather than a stray
keystroke. `OPEN_URL` accepts `http`/`https` only, and the clipboard and media
verbs are deliberately narrow.
