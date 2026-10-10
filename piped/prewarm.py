"""Warm the two things a cold first turn would otherwise pay for.

THE TALKER. jarvis-llm is a 3.75 GB model. Every request the agent sends carries
keep_alive: -1, which is what keeps it resident, but resident is not the same as
guaranteed: an Ollama restart, `ollama run` on the same card, or a second model
being loaded (OLLAMA_MAX_LOADED_MODELS defaults to 3 x GPUs) can all evict it.
The cost of that is measured and it is enormous compared with the ~0.8 s median
inference: 0.65 s load when the file is in page cache, 9-16 s after a normal
eviction, and 31.5 s under memory pressure (the swap-thrashing case measured
while investigating the VRAM split -- see the header of start-kev.sh). A turn
that lands in that window is 10-30 s of silence, which reads as a broken agent
rather than a slow one.

Ollama's documented remedy is to preload: send the model an empty request and it
is loaded and kept in memory. This module does exactly that, twice in the places
that matter -- once when the agent starts, and again when a client connects,
because a connection is the moment somebody is about to speak.

THE VOICE. pipecat builds the kokoro-onnx session in KokoroTTSService.__init__
(services/kokoro/tts.py:212), so the weights are already in memory by the time
build_pipeline returns. The first inferences are still slower than the rest:
measured 1.8 s / 1.5 s / 2.7 s for the first three utterances of a run against
0.63-0.89 s afterwards, which is onnxruntime's arena and thread pool warming up
plus espeak-ng's first call. One throwaway synthesis at boot moves that off the
user's first sentence instead of onto it.

Everything here is best-effort by design: a failed warm-up must never stop the
agent from starting or serving, so each call logs and returns.
"""

import asyncio
import os
import time

import aiohttp
from loguru import logger
from pipecat.utils.types import assert_given, require_given

OLLAMA_BASE = os.environ.get("JARVIS_OLLAMA_BASE", "http://127.0.0.1:11434/v1")
OLLAMA_MODEL = os.environ.get("JARVIS_LLM_MODEL", "jarvis-llm")

# The agent's own endpoint is the OpenAI-compatible /v1 one, but the documented
# preload endpoint is the native API, so the base is derived by trimming.
OLLAMA_NATIVE_BASE = OLLAMA_BASE.removesuffix("/v1")

# A client can connect, drop, and reconnect in quick succession; there is no point
# paying for a second preload inside this window, since the first one either
# succeeded (the model is up) or failed loudly in the log.
PREWARM_COOLDOWN_SECS = float(os.environ.get("JARVIS_PREWARM_COOLDOWN", "10"))

_lock = asyncio.Lock()
_last_talker_prewarm = 0.0
_tts_warmed = False


async def prewarm_talker(reason: str) -> None:
    """Ask Ollama to load the talker and keep it loaded.

    An empty request is the documented trigger; keep_alive -1 pins it. The reply
    is discarded except for load_duration, which is the number worth having in
    the log: near zero means it was already resident, thousands of ms means this
    call just saved the next speaker that much silence.
    """
    payload = {"model": OLLAMA_MODEL, "keep_alive": -1}
    t0 = time.perf_counter()
    try:
        # Long enough for a genuine cold load (measured up to 31 s), short enough
        # that a hung Ollama cannot hold startup hostage.
        timeout = aiohttp.ClientTimeout(total=90)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(f"{OLLAMA_NATIVE_BASE}/api/chat", json=payload) as resp:
                body = await resp.json()
                if resp.status >= 400:
                    raise RuntimeError(f"ollama returned {resp.status}: {str(body)[:200]}")
        # Wall time is the honest number. For a preload request Ollama reports
        # load_duration as 0 even when it has just spent 25 s paging the model in
        # (observed), so the reported figure is only worth printing when it is not
        # zero -- otherwise the log line reads as "nothing happened" after the
        # one event worth knowing about.
        load_ms = (body.get("load_duration") or 0) / 1e6
        logger.info(
            "[prewarm] talker ready in {:.2f}s{} ({})",
            time.perf_counter() - t0,
            f", ollama load {load_ms:.0f}ms" if load_ms >= 1 else "",
            reason,
        )
    except Exception as e:
        logger.warning("[prewarm] talker preload failed ({}): {}", reason, e)
        return

    # The empty request above only LOADS the model: Ollama answers it with
    # done_reason "load" and generates nothing. The first real turn then pays for
    # the first full inference against the agent's system prompt and 40-verb tool
    # schema -- measured 9.3 s of LLM inference on the first turn after a restart,
    # against 0.56-1.02 s on the turns after it. So warm the generation path as
    # well, through the same OpenAI-compatible endpoint (and the same reasoning
    # override) that the agent uses, with a one-token budget.
    t1 = time.perf_counter()
    try:
        warm_body = {
            "model": OLLAMA_MODEL,
            "messages": [{"role": "user", "content": "ok"}],
            "max_tokens": 1,
            "stream": False,
            "keep_alive": -1,
            "reasoning": {"effort": "none"},
        }
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(f"{OLLAMA_BASE}/chat/completions", json=warm_body) as resp:
                detail = await resp.json()
                if resp.status >= 400:
                    raise RuntimeError(f"ollama returned {resp.status}: {str(detail)[:160]}")
        logger.info(
            "[prewarm] first inference warm in {:.2f}s ({})",
            time.perf_counter() - t1,
            reason,
        )
    except Exception as e:
        logger.warning("[prewarm] inference warm-up failed ({}): {}", reason, e)


async def prewarm_tts(service) -> None:
    """Run one throwaway synthesis so the first real sentence is not the warm-up.

    Reaches into the service for its session and settings on purpose: the point
    is to exercise the same onnxruntime session and espeak path that run_tts
    uses, without pipecat's metrics, resampling or frame plumbing around it. The
    settings are read with the same helpers tts.py uses, so a missing voice or
    language fails the same way it would at synthesis time.
    """
    global _tts_warmed
    if _tts_warmed or service is None:
        return
    try:
        voice = require_given(service._settings.voice, "Kokoro TTS voice")
        lang = require_given(service._settings.language, "Kokoro TTS language")
        speed = assert_given(service._settings.speed)
        t0 = time.perf_counter()
        stream = service._kokoro.create_stream("Warming up.", voice=voice, lang=lang, speed=speed)
        async for _samples, _rate in stream:
            break  # one chunk is enough to pay the first-call cost
        _tts_warmed = True
        logger.info("[prewarm] kokoro first chunk in {:.2f}s", time.perf_counter() - t0)
    except Exception as e:
        logger.warning("[prewarm] kokoro warm-up failed: {}", e)


async def prewarm_all(reason: str, tts_service=None) -> None:
    """Both warm-ups, cheap to call: repeats inside the cooldown are skipped."""
    global _last_talker_prewarm
    async with _lock:
        now = time.monotonic()
        if now - _last_talker_prewarm >= PREWARM_COOLDOWN_SECS:
            _last_talker_prewarm = now
            await prewarm_talker(reason)
        else:
            logger.debug("[prewarm] skipped talker preload ({}; inside cooldown)", reason)
        await prewarm_tts(tts_service)
