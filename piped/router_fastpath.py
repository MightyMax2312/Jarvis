"""kev router fast path: skip the 4B turn when the intent is unambiguous.

kev-0.5b is a prefill-only pointer model -- ~53 ms warm on CUDA against ~1.5 s
for the first Ollama turn. When it is confident about a self-contained desktop
verb there is nothing for the LLM to decide, so this processor runs the action
itself and speaks the outcome, saving a full model round trip per command.

Placement is load-bearing. It sits between the user aggregator and the LLM
because ``LLMContextFrame`` is what triggers inference (llm_service.py:771) --
the aggregator's ``push_context_frame()`` emits it once a turn closes. Swallowing
that frame is what suppresses the LLM. Forwarding it and then acting would start
a 4B turn that races the action and then talks over the result.

Speaking the result uses ``TTSSpeakFrame``, Pipecat's own mechanism for text
that is not LLM output (tts_service.py:950). It opens its own turn context and,
because no ``LLMFullResponseStartFrame`` was emitted on this path, also routes
the text into the assistant aggregator -- so the conversation history stays
coherent and the next LLM turn sees what was actually said.
"""

from loguru import logger
from pipecat.frames.frames import Frame, LLMContextFrame, TTSSpeakFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from .router import KevRouter
from .tools import perform_action


class RouterFastPath(FrameProcessor):
    """Answers confident router verdicts without involving the LLM."""

    def __init__(self, router: KevRouter | None = None, threshold: float = 0.70,
                 enabled: bool = True):
        super().__init__()
        self._router = router or KevRouter(threshold=threshold)
        self._enabled = enabled
        self._hits = 0
        self._misses = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled
        logger.info(f"[fastpath] {'enabled' if enabled else 'disabled'}")

    async def cleanup(self):
        await super().cleanup()
        await self._router.close()
        logger.info(f"[fastpath] {self._hits} hit(s), {self._misses} deferred to LLM")

    @staticmethod
    def _last_user_text(context) -> str:
        """Newest user turn, flattened to plain text.

        Content is a str for a plain turn but a list of typed parts once an
        image or audio part is attached, so both shapes are handled.
        """
        for message in reversed(context.get_messages()):
            if message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                return " ".join(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                ).strip()
            return ""
        return ""

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if not isinstance(frame, LLMContextFrame) or not self._enabled:
            await self.push_frame(frame, direction)
            return

        text = self._last_user_text(frame.context)
        if not text:
            await self.push_frame(frame, direction)
            return

        verdict = await self._router.route(text)
        if verdict.reason != "fast path":
            self._misses += 1
            await self.push_frame(frame, direction)
            return

        self._hits += 1
        logger.info(
            f"[fastpath] {verdict.action} conf={verdict.confidence:.3f} "
            f"in {verdict.latency * 1000:.0f}ms -- no LLM turn"
        )

        # Drop the context frame: forwarding it is what starts the LLM.
        # DESCRIBE_SCREEN takes the user's own sentence as the question, which is
        # why it can be self-contained: there is no argument to extract, the
        # words are already the question.
        goal = text if verdict.action == "DESCRIBE_SCREEN" else ""
        reply = await perform_action(verdict.action, goal)
        await self.push_frame(TTSSpeakFrame(reply, append_to_context=True), direction)
