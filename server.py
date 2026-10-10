import time
import asyncio
import threading
import requests
import numpy as np
import io
import re
import os
import base64
import soundfile as sf
import sherpa_onnx
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
# torch and kokoro are NOT imported here on purpose: both are needed only by the
# torch TTS path, and importing torch alone costs a few hundred MB of RSS. They
# are imported inside tts_pipeline() the first time audio is actually asked for.
# Measured on this box: the backend sat at ~3.1 GB RSS, and with kev (1.1 GB), the
# agent (~1 GB) and Ollama staging the 3.75 GB talker, the 7.6 GB WSL VM was
# OOM-killed three times while the services came up.

# One definition of the action vocabulary, shared with the agent. piped/ sits
# next to this file and vocab.py imports nothing, so this resolves under
# jarvis-env -- which has no pipecat installed and therefore cannot import the
# rest of piped. It is what stops this endpoint and the agent's fast path from
# disagreeing about which verbs exist, or about which of them may be dispatched
# without an argument.
from piped.vocab import ACTION_CHOICES, ACTION_KEYWORDS, SELF_CONTAINED_ROUTER

JEV_ENDPOINT = "http://127.0.0.1:8000/v1/systemone"
OLLAMA_ENDPOINT = "http://127.0.0.1:11434/api/chat"
OLLAMA_MODEL = "jarvis-llm"

# --- tuning knobs (phase 0) -------------------------------------------------
# Jev is a ~53 ms prefill on CUDA once warm, but the first call pays ~1.2 s of
# lazy CUDA/cuDNN init, so the warmup below exists and the timeout is generous.
JEV_TIMEOUT = float(os.environ.get("JEV_TIMEOUT", "15"))
# Ollama needs to page the 4B in on a cold request: measured 9.3 s cold,
# 0.66-0.90 s warm. 4 s was far too tight and produced the "Task timed out."
# responses. 60 s covers a cold start without hanging the caller.
OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", "60"))
OLLAMA_VISION_TIMEOUT = float(os.environ.get("OLLAMA_VISION_TIMEOUT", "90"))

# kev-0.5b scores 9/10 on the 18-verb set, and its one miss ("hello there how are
# you" -> OPEN_BROWSER) came back at confidence 0.49-0.50, while every correct
# verb scored >= 0.77. 0.70 is a clean separator: take the fast path only when
# the router is genuinely sure, otherwise let the LLM decide.
JEV_CONFIDENCE_THRESHOLD = float(os.environ.get("JEV_CONFIDENCE_THRESHOLD", "0.70"))

# Pin the talker in VRAM. Ollama unloads a model after 5 minutes by default and
# paging this 3.4 GB one back in costs 9-16 s -- measured 16.3 s on an un-pinned
# request, and the worst outlier in the whole latency log was a 39.3 s LLM
# inference against a 0.81 s median. Set on every request below rather than via
# OLLAMA_KEEP_ALIVE on the service, because that needs sudo. Override with
# OLLAMA_KEEP_ALIVE to hand control back to the server default.
OLLAMA_KEEP_ALIVE = int(os.environ.get("OLLAMA_KEEP_ALIVE", "-1"))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARAKEET_MODEL_DIR = os.path.join(
    BASE_DIR, "models", "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
)
STT_SAMPLE_RATE = 16000
# Left at 4 on purpose. Raising it to 8 of the 16 hardware threads this box has
# (Ryzen 7 5800H, 8c/16t) measured *slower* on a 2.75 s utterance -- 0.34-0.41 s
# against 0.28-0.42 s -- because thread synchronisation costs more than the extra
# parallelism buys at this model size. Decoding also happens off the audio path:
# pipecat only calls this after VAD has closed an utterance, so it is not the
# stage in front of a turn anyway. Override with STT_NUM_THREADS if a different
# workload disagrees.
STT_NUM_THREADS = int(os.environ.get("STT_NUM_THREADS", "4"))

stt_recognizer = None
stt_lock = threading.Lock()


def load_stt_recognizer():
    """Loads the Parakeet TDT 0.6B v3 (int8 ONNX) offline recognizer.

    Runs on CPU by default: it is already ~12x realtime there, and the GPU is
    contended between Ollama and Kokoro.
    """
    global stt_recognizer
    if stt_recognizer is not None:
        return stt_recognizer

    required = ["encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"]
    missing = [f for f in required if not os.path.isfile(os.path.join(PARAKEET_MODEL_DIR, f))]
    if missing:
        raise FileNotFoundError(
            f"Parakeet model incomplete at {PARAKEET_MODEL_DIR}; missing: {missing}"
        )

    t0 = time.time()
    stt_recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=os.path.join(PARAKEET_MODEL_DIR, "encoder.int8.onnx"),
        decoder=os.path.join(PARAKEET_MODEL_DIR, "decoder.int8.onnx"),
        joiner=os.path.join(PARAKEET_MODEL_DIR, "joiner.int8.onnx"),
        tokens=os.path.join(PARAKEET_MODEL_DIR, "tokens.txt"),
        num_threads=STT_NUM_THREADS,
        model_type="",
        provider="cpu",
        debug=False,
    )
    print(f"[WSL Server] Parakeet TDT v3 int8 STT ready on CPU in {time.time() - t0:.2f}s")
    return stt_recognizer


def _resample_to_16k(samples: np.ndarray, sr: int) -> np.ndarray:
    if sr == STT_SAMPLE_RATE:
        return samples.astype(np.float32, copy=False)
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(int(sr), STT_SAMPLE_RATE)
    return resample_poly(samples, STT_SAMPLE_RATE // g, int(sr) // g).astype(np.float32)


def transcribe_pcm(samples: np.ndarray) -> str:
    """Blocking transcribe of mono float32 PCM. Caller should offload to a thread."""
    recognizer = load_stt_recognizer()
    with stt_lock:
        stream = recognizer.create_stream()
        stream.accept_waveform(STT_SAMPLE_RATE, samples)
        recognizer.decode_stream(stream)
        return (stream.result.text or "").strip()

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await asyncio.to_thread(load_stt_recognizer)
    except Exception as e:
        print(f"[WSL Server] STT warmup warning: {e}")

    try:
        await asyncio.to_thread(warm_jev)
    except Exception as e:
        print(f"[WSL Server] Jev warmup warning: {e}")

    yield

app = FastAPI(title="Jarvis WSL Backend", lifespan=lifespan)

# Kokoro is built on first use, never at import.
#
# It used to be constructed here. That cost 5-10 s of startup and held torch plus
# the 82M weights resident for the whole process lifetime -- memory this box does
# not have to spare alongside kev, the agent and Ollama's staging of the 3.75 GB
# talker. The agent's screen-description path does not need it at all any more:
# its client asks for text only (want_audio false) and the agent speaks the reply
# with its own ONNX Kokoro, which is also 3-4x faster than this torch path.
#
# Pinned to CPU on purpose. Kokoro is only 82M params and runs ONNX-only under
# Pipecat, so holding ~731 MiB of VRAM for it starves the two models that
# actually need the GPU: kev.serve (bf16 router) and the 4B talker. With
# Kokoro on CUDA the card sat at 3762/4096 MiB and Ollama thrashed, producing
# 36 s conversational turns. On CPU the talker gets the whole card.
_tts_pipeline = None


def tts_pipeline():
    """The Kokoro pipeline, built on first use. See the note above."""
    global _tts_pipeline
    if _tts_pipeline is None:
        from kokoro import KPipeline

        print("[WSL Server] Loading Kokoro-82M TTS on CPU (first use)...")
        _tts_pipeline = KPipeline(lang_code='a', device='cpu', repo_id='hexgrad/Kokoro-82M')
    return _tts_pipeline

class CommandRequest(BaseModel):
    user_text: str
    screen_context: str = "Desktop environment active."
    image_b64: str = ""
    # The agent's desktop client posts here to get a screen description or input
    # grounding, and the agent then speaks the answer itself with its own ONNX
    # Kokoro. Synthesising a WAV here as well cost 5-6 s of blocked time per call
    # for audio the caller threw away: measured 6.19 s and 5.07 s of torch Kokoro
    # for a sentence the agent synthesises in under 2 s. Callers that really do
    # want the WAV (the original client.py plays it) keep asking for it, so this
    # defaults to on and the agent's client opts out.
    want_audio: bool = True


@app.post("/api/stt")
async def handle_stt(request: Request):
    """Transcribe audio with Parakeet TDT 0.6B v3.

    Body is either:
      * raw mono float32 PCM, rate taken from X-Sample-Rate (default 16000), or
      * any WAV/FLAC/OGG soundfile can decode (sniffed via RIFF/etc. magic).
    """
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="Empty audio body")

    header_sr = request.headers.get("x-sample-rate")
    try:
        declared_sr = int(header_sr) if header_sr else STT_SAMPLE_RATE
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Bad X-Sample-Rate: {header_sr}")

    if len(body) >= 4 and body[:4] in (b"RIFF", b"FORM", b"OggS", b"fLaC"):
        try:
            samples, sr = sf.read(io.BytesIO(body), dtype="float32", always_2d=False)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Undecodable audio: {e}")
        samples = samples.reshape(-1)
    else:
        if len(body) % 4 != 0:
            raise HTTPException(
                status_code=400,
                detail=f"Raw float32 PCM length must be a multiple of 4, got {len(body)}",
            )
        samples = np.frombuffer(body, dtype=np.float32)
        sr = declared_sr

    if samples.size == 0:
        raise HTTPException(status_code=400, detail="No audio samples")
    if samples.size > STT_SAMPLE_RATE * 60:
        raise HTTPException(status_code=413, detail="Utterance longer than 60s")

    try:
        samples = _resample_to_16k(samples, sr)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Resample failed: {e}")

    duration = samples.size / STT_SAMPLE_RATE
    t0 = time.time()
    try:
        text = await asyncio.to_thread(transcribe_pcm, samples)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        print(f"[STT Error]: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")

    elapsed = time.time() - t0
    print(f"\u270d\ufe0f [STT] {duration:.2f}s audio -> \"{text}\" ({elapsed:.2f}s)")
    return {
        "text": text,
        "duration": round(duration, 3),
        "decode_time": round(elapsed, 3),
        "model": "parakeet-tdt-0.6b-v3-int8",
    }


@app.get("/api/stt/health")
async def stt_health():
    return {
        "ready": stt_recognizer is not None,
        "model_dir": PARAKEET_MODEL_DIR,
        "sample_rate": STT_SAMPLE_RATE,
        "num_threads": STT_NUM_THREADS,
    }


def jev_health() -> dict:
    try:
        res = requests.get("http://127.0.0.1:8000/v1/models", timeout=3.0)
        res.raise_for_status()
        card = (res.json().get("models") or [{}])[0]
        return {
            "up": True,
            "device": card.get("device"),
            "dtype": card.get("dtype"),
            "base": card.get("base"),
        }
    except Exception as e:
        return {"up": False, "error": str(e)}


def warm_jev():
    """Triggers CUDA/cuDNN init so the first real utterance is not 1.2 s slow."""
    warm = {
        "state": "User intent: open youtube. Context: warmup",
        "questions": {"action": {"type": "choice", "criteria": {"NAVIGATE_YOUTUBE": "Open YouTube", "NONE": "No action"}}},
    }
    try:
        requests.post(JEV_ENDPOINT, json=warm, timeout=JEV_TIMEOUT)
        print("[WSL Server] Jev router ready.")
    except Exception as e:
        print(f"[WSL Server] Jev router unavailable: {e}")


def query_jev_action(user_text: str, context: str, choices: dict):
    payload = {
        "state": f"User intent: {user_text}. Context: {context}",
        "questions": {"action": {"type": "choice", "criteria": choices}}
    }
    t0 = time.time()
    try:
        res = requests.post(JEV_ENDPOINT, json=payload, timeout=JEV_TIMEOUT)
        res.raise_for_status()
        data = res.json()
        action = data["answers"]["action"]["choice"]
        confidence = data["answers"]["action"]["confidence"]
        print(f"[Jev] {time.time()-t0:.3f}s -> {action} conf={confidence:.3f}")
        return action, confidence
    except Exception as e:
        print(f"[Jev Error]: {e}")
        return None, 0.0

def query_vlm_visual_grounding(prompt: str, image_b64: str) -> dict:
    sys_prompt = "Locate visual target in screenshot. Return JSON strictly: {\"x\": <0-1000>, \"y\": <0-1000>}."
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"Target: {prompt}"},
        ],
        "images": [image_b64] if image_b64 else [],
        "stream": False,
        # This model reports 'thinking' in its capabilities; without this it
        # burns the whole timeout on <think> tokens and returns empty content.
        "think": False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {"num_predict": 25, "temperature": 0.0}
    }
    try:
        res = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=OLLAMA_VISION_TIMEOUT)
        res.raise_for_status()
        resp_text = res.json().get("message", {}).get("content", "").strip()
        match = re.search(r'"x"\s*:\s*(\d+),\s*"y"\s*:\s*(\d+)', resp_text)
        if match:
            return {"norm_x": int(match.group(1)), "norm_y": int(match.group(2))}
        print(f"[VLM Grounding] no coords in {resp_text[:120]!r}")
    except Exception as e:
        print(f"[VLM Grounding Error/Timeout]: {e}")
    return {"norm_x": None, "norm_y": None}

def query_vlm_describe(prompt: str, image_b64: str) -> str:
    """Ask the vision model what is on the screen, in words.

    Separate from query_vlm_visual_grounding, which returns coordinates for a
    click target. The grounding prompt is useless for a question like "what do
    you see" -- it wants {"x":..,"y":..} and will invent one.

    Routed to the conversation branch rather than the action branch on purpose:
    this is an answer, not something to execute, and the old behaviour returned
    the literal string "Executing describe screen." without the model ever
    seeing the screenshot.
    """
    if not image_b64:
        return "I could not capture the screen."
    sys_prompt = (
        "You are describing a screenshot of the user's screen for a voice "
        "assistant. Answer in ONE or two short spoken sentences. Name the "
        "applications you can see and what they appear to be showing. Do not "
        "use markdown, lists, or code blocks. Do not mention that you received "
        "an image."
    )
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt or "What is on this screen?"},
        ],
        "images": [image_b64],
        "stream": False,
        "think": False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {"num_predict": 90, "temperature": 0.2},
    }
    try:
        res = requests.post(OLLAMA_ENDPOINT, json=payload,
                            timeout=OLLAMA_VISION_TIMEOUT)
        res.raise_for_status()
        text = res.json().get("message", {}).get("content", "").strip()
        return text or "I could not make sense of the screen."
    except Exception as e:
        print(f"[VLM Describe Error]: {e}")
        return f"I could not look at the screen right now: {e}"


def query_system_2(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": "You are JARVIS, a concise voice assistant. "
                                          "Answer in at most one or two short sentences. "
                                          "Never use markdown, lists, or code blocks. "
                                          "Speak plainly, as if out loud."},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        # Required: jarvis-llm is a reasoning model. With thinking left on it
        # spent 34 s on <think> tokens and hit the old 4 s timeout.
        "think": False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {"num_predict": 120}
    }
    try:
        res = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=OLLAMA_TIMEOUT)
        res.raise_for_status()
        return res.json().get("message", {}).get("content", "").strip()
    except requests.exceptions.Timeout:
        return "Task timed out."
    except Exception as e:
        return f"System 2 core unreachable: {e}"

def synthesize_tts(text: str) -> bytes:
    try:
        generator = tts_pipeline()(text, voice='af_heart', speed=1.0)
        # Anything tensor-like has .cpu(); plain arrays do not. Previously this
        # asked isinstance(audio, torch.Tensor), which is why torch was imported
        # at module scope for the sake of one check.
        audio_chunks = [audio.cpu().numpy() if hasattr(audio, "cpu") else audio
                        for _, _, audio in generator]
        if not audio_chunks:
            return b""
        full_audio = np.concatenate(audio_chunks, axis=0).astype(np.float32)
        byte_io = io.BytesIO()
        sf.write(byte_io, full_audio, 24000, format='WAV')
        return byte_io.getvalue()
    except Exception as e:
        print(f"[TTS Error]: {e}")
        return b""


async def _audio_b64(text: str, want_audio: bool) -> str:
    """The spoken reply as base64 WAV, or empty when the caller does not want it.

    Skipping the synthesis rather than synthesising and discarding it is the whole
    point: on this box one torch-Kokoro pass over a sentence is 5-6 s, and it sits
    between the vision call and the response.
    """
    if not want_audio or not text:
        return ""
    audio = await asyncio.to_thread(synthesize_tts, text)
    return base64.b64encode(audio).decode("ascii") if audio else ""

# The option set and the pre-filter come from piped/vocab.py; see the import at
# the top of this file. This module used to carry byte-identical copies of both.

@app.post("/api/command")
async def handle_command(req: CommandRequest):
    user_text = req.user_text.strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="Empty input")

    print(f"\n👤 [User Query]: \"{user_text}\"")

    if any(kw in user_text.lower() for kw in ACTION_KEYWORDS):
        # Every call in here is blocking (requests + Kokoro ONNX). Running them
        # inline froze the whole event loop, which also serves /api/stt for the
        # Pipecat agent -- one slow Ollama turn stalled transcription for every
        # other client. to_thread keeps the loop free.
        action, conf = await asyncio.to_thread(
            query_jev_action, user_text, req.screen_context, ACTION_CHOICES
        )

        if action and action != "NONE" and conf >= JEV_CONFIDENCE_THRESHOLD:
            if action not in SELF_CONTAINED_ROUTER:
                # The same rule the agent's fast path applies, from the same set.
                # A verb that needs an argument cannot be dispatched from a router
                # verdict alone: the router never sees "what to type" or "which
                # button", so executing here would announce an action that had
                # nothing to act on. Let the LLM, which reads the whole sentence,
                # extract the target instead.
                print(f"[Router] {action} at conf={conf:.3f} needs an argument "
                      f"-> deferring to LLM")
            elif action == "DESCRIBE_SCREEN":
                # This IS the live route for the agent's describe_screen tool, and
                # the sequence is worth spelling out because it crosses the process
                # boundary twice: the fast path sends the verb to the desktop
                # client, the client captures the screen and posts it straight back
                # here as image_b64, and the vision model runs on this side
                # (query_vlm_describe). The client never talks to Ollama itself.
                description = await asyncio.to_thread(
                    query_vlm_describe, user_text, req.image_b64
                )
                return {
                    "type": "conversation",
                    "text": description,
                    "audio_wav_base64": await _audio_b64(description, req.want_audio),
                }
            else:
                friendly_name = action.lower().replace('_', ' ')
                response_text = f"Executing {friendly_name}."

                payload_res = {
                    "type": "action",
                    "action": action,
                    "confidence": conf,
                    "text": response_text,
                    # base64, matching the key name. This was hex-encoded before,
                    # so any client doing base64 decode got garbage.
                    "audio_wav_base64": await _audio_b64(response_text, req.want_audio),
                }

                if action == "VISUAL_CLICK" and req.image_b64:
                    payload_res.update(
                        await asyncio.to_thread(
                            query_vlm_visual_grounding, user_text, req.image_b64
                        )
                    )

                return payload_res

        if action and action != "NONE" and conf < JEV_CONFIDENCE_THRESHOLD:
            print(f"[Router] {action} at conf={conf:.3f} below "
                  f"{JEV_CONFIDENCE_THRESHOLD} -> deferring to LLM")

    # Fallback to System 2 conversation
    response_text = await asyncio.to_thread(query_system_2, user_text)
    return {
        "type": "conversation",
        "text": response_text,
        "audio_wav_base64": await _audio_b64(response_text, req.want_audio),
    }

@app.get("/api/health")
async def health():
    """Single place to see what is actually up and where the VRAM went."""
    return {
        "stt": {
            "ready": stt_recognizer is not None,
            "model": "parakeet-tdt-0.6b-v3-int8",
            "device": "cpu",
        },
        # to_thread: jev_health is a blocking requests.get with a 3 s timeout.
        # Called inline it would stall the event loop, so a slow router would
        # also delay /api/stt for the agent.
        "router": await asyncio.to_thread(jev_health),
        "confidence_threshold": JEV_CONFIDENCE_THRESHOLD,
        "llm": {"model": OLLAMA_MODEL, "think": False},
    }

if __name__ == "__main__":
    import uvicorn
    # Loopback by default. This port serves /api/command, which can make the
    # machine act, and /api/stt, with no authentication of any kind; binding
    # 0.0.0.0 put both on the Wi-Fi LAN. The box is in WSL mirrored mode, so
    # Windows -- and the desktop client -- reach a 127.0.0.1 listener here
    # directly. Set JARVIS_HOST=0.0.0.0 to expose it deliberately.
    uvicorn.run(app, host=os.environ.get("JARVIS_HOST", "127.0.0.1"), port=8080)
