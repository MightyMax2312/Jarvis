"""Parakeet STT over the existing WSL backend's /api/stt endpoint.

The model is a batch (offline) ASR, so this is a SegmentedSTTService: VAD
delimits an utterance, we transcribe it once, off the audio path. Measured
~0.30 s for a 3.8 s utterance on CPU.
"""

import asyncio
from collections.abc import AsyncGenerator

import aiohttp
import numpy as np
from loguru import logger

from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.utils.time import time_now_iso8601


class ParakeetHTTPSTTService(SegmentedSTTService):
    """Posts raw mono float32 PCM to a local FastAPI Parakeet endpoint."""

    def __init__(
        self,
        *,
        url: str,
        sample_rate: int = 16000,
        request_timeout: float = 30.0,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self._url = url
        self._request_timeout = request_timeout
        self._session: aiohttp.ClientSession | None = None

    @property
    def wants_wav_segments(self) -> bool:
        # The endpoint takes raw float32 PCM; a WAV header would be read as audio.
        return False

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self._request_timeout)
            )
        return self._session

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        if not audio:
            return

        samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0

        try:
            session = await self._ensure_session()
            async with session.post(
                self._url,
                data=samples.tobytes(),
                headers={
                    "Content-Type": "application/octet-stream",
                    "X-Sample-Rate": str(self.sample_rate),
                },
            ) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    yield ErrorFrame(
                        error=f"{self}: STT {resp.status} from {self._url}: {body[:200]}"
                    )
                    return
                payload = await resp.json()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            yield ErrorFrame(error=f"{self}: STT request failed: {e}")
            return

        text = (payload.get("text") or "").strip()
        if not text:
            return

        logger.info(
            "[STT] {:.2f}s audio decoded in {:.2f}s: {}",
            payload.get("duration", 0.0),
            payload.get("decode_time", 0.0),
            text,
        )
        yield TranscriptionFrame(text, self._user_id, time_now_iso8601(), "")
