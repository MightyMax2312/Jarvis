"""Raw PCM serializer for the WebSocket transport.

Pipecat's bundled serializers are all provider-specific (Twilio, Plivo, Genesys,
...) and the websocket transport drops every message when no serializer is set
(server.py: `if not self._params.serializer: continue`). This one is the
simplest possible framing: audio in, audio out, no headers.

Wire format, both directions: headerless mono signed 16-bit little-endian PCM.
"""

from loguru import logger

from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    TTSAudioRawFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer


class RawPCMSerializer(FrameSerializer):
    def __init__(self, *, input_sample_rate: int = 16000, input_channels: int = 1,
                 **kwargs):
        super().__init__(**kwargs)
        self._input_sample_rate = input_sample_rate
        self._input_channels = input_channels
        self._saw_audio = False

    async def setup(self, setup):
        await super().setup(setup)
        self._input_sample_rate = setup.audio_in_sample_rate or self._input_sample_rate

    async def deserialize(self, data: str | bytes) -> Frame | None:
        if isinstance(data, str):
            # Control channel, kept JSON so the client can send small commands
            # (e.g. a barge-in signal) without touching the audio path.
            import json

            try:
                msg = json.loads(data)
            except json.JSONDecodeError:
                logger.warning(f"[serializer] unparseable control frame: {data[:120]!r}")
                return None
            if msg.get("type") == "interrupt":
                return InterruptionFrame()
            logger.info(f"[serializer] control frame: {msg}")
            return None

        if not data:
            return None

        self._saw_audio = True
        return InputAudioRawFrame(
            audio=data,
            sample_rate=self._input_sample_rate,
            num_channels=self._input_channels,
        )

    async def serialize(self, frame: Frame) -> str | bytes | None:
        # The output transport re-wraps TTS audio as OutputAudioRawFrame before
        # calling the serializer (websocket/server.py:456), so handling only
        # TTSAudioRawFrame silently drops every chunk. Accept both.
        if isinstance(frame, (OutputAudioRawFrame, TTSAudioRawFrame)):
            return frame.audio
        return None
