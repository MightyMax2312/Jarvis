"""Live session log: what the user said, what the agent said, and what it did.

The default agent log is scattered. Transcripts appear only when the STT
request finishes, tool calls appear without their results, and the agent's
actual spoken words are buried in a DEBUG line inside Kokoro
("Generating TTS [...]"). None of it is in one place, and none of it lines up
with a turn boundary, so you cannot tell what the agent is doing right now.

This adds two FrameProcessors to the pipeline -- one after the STT to catch
what came in, one at the tail to catch what goes out -- both writing to a
single stream:

    11:04:12.331  YOU     "what windows do i currently have open"
    11:04:12.640  ROUTER  defer NONE conf=0.334 (184ms) -> LLM
    11:04:13.980  TOOL    list_windows
    11:04:14.770  TOOL    ok  "B&O Audio Control; about:blank - Google Chrome"
    11:04:15.104  SAY     "You have these windows open: B and O Audio Control"
    11:04:16.512  SPEAK   done 1.41s of audio
    11:04:16.512  TURN    total 4.36s

Writes to stderr (alongside the loguru stream) and, when JARVIS_SPY_LOG is set,
to that file as well. Pass-through only: this observes, it never alters, drops
or reorders a frame.
"""

import os
import time
from datetime import datetime

from loguru import logger
from pipecat.frames.frames import (
    Frame,
    LLMContextAssistantTurnFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

SPY_LOG = os.environ.get("JARVIS_SPY_LOG", "")

# SPY_DEBUG dumps every text-bearing frame reaching the tail, with its class and
# will_be_spoken flag. Off by default; used to identify which frame actually
# carries the spoken text, since that differs between streaming (TOKEN) and
# non-streaming TTS aggregation.
SPY_DEBUG = os.environ.get("JARVIS_SPY_DEBUG", "0") == "1"

# Per-turn state, reset when speech stops.
_turn_started: float | None = None


def _stamp() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def _elapsed() -> str:
    if _turn_started is None:
        return ""
    return f"  (+{time.monotonic() - _turn_started:.2f}s)"


def _emit(stage: str, message: str) -> None:
    line = f"{_stamp()}  {stage:<7} {message}{_elapsed()}"
    logger.info(line)
    if SPY_LOG:
        try:
            with open(SPY_LOG, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError:
            pass


class InputSpy(FrameProcessor):
    """Sits after the STT service. Logs what the user actually said.

    Stateless on purpose: it observes and passes through. There used to be a
    speaking flag on this class that nothing ever read.
    """

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            _emit("VOICE", "user started speaking")
        elif isinstance(frame, UserStoppedSpeakingFrame):
            _emit("VOICE", "user stopped speaking")
        elif isinstance(frame, TranscriptionFrame) and frame.text.strip():
            _emit("YOU", f'"{frame.text.strip()}"')

        await self.push_frame(frame, direction)


class OutputSpy(FrameProcessor):
    """Sits at the tail. Logs what the agent is saying.

    The spoken text is read from LLMContextAssistantTurnFrame, not from
    TTSTextFrame. That is not a shortcut, it is the only option in TOKEN
    aggregation mode: tts_service.py gates both of its text-frame pushes on
    `not self._is_streaming_tokens` (lines 1339 and 1435), so a streaming
    pipeline emits no TTS text frames downstream at all. Verified by dumping
    every text-bearing frame reaching this position -- the only one that arrives
    is LLMContextAssistantTurnFrame. The turn frame's text is the same string
    that was handed to Kokoro, so it is what the user actually heard.

    Ordering caveat: the assistant aggregator emits that frame when the reply is
    complete, which is *after* TTSStoppedFrame. So the SAY line lands near the
    end of the reply, not its start. There is no earlier signal for the text in
    TOKEN mode; TTSStartedFrame still marks when speech actually began, and the
    two together bracket the utterance.

    TTSStartedFrame/TTSStoppedFrame delimit the turn reliably in both modes and
    are what the elapsed-time column is measured against.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        global _turn_started

        if SPY_DEBUG and isinstance(frame, Frame) and hasattr(frame, "text"):
            _emit("FRAME", f"{type(frame).__name__} "
                           f"will_be_spoken={getattr(frame, 'will_be_spoken', '?')} "
                           f"{str(getattr(frame, 'text', ''))[:60]!r}")

        if isinstance(frame, TTSStartedFrame):
            _turn_started = _turn_started or time.monotonic()
            _emit("SPEAK", "agent started talking")
        elif isinstance(frame, LLMContextAssistantTurnFrame):
            text = (frame.text or "").strip()
            if text:
                _emit("SAY", f'"{text}"')
        elif isinstance(frame, TTSStoppedFrame):
            if _turn_started is not None:
                _emit("TURN", "done")
                _turn_started = None

        await self.push_frame(frame, direction)
