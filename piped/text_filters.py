"""Hold one text frame so punctuation and spacing still reach the synthesiser.

Kokoro phonemises through espeak-ng and refuses a fragment with nothing to say
("Nothing to synthesize, '  \\n' produced no phonemes"), so something has to keep
phoneme-less fragments away from it. The original version of this filter did that
by DROPPING them -- and what it dropped was the punctuation.

Measured on a real streamed reply:

    model text : 'I am functioning perfectly as JARVIS.'
    kept       : 'I am functioning perfectly as JARVIS'
    dropped    : ['.']

The model punctuates perfectly well; the filter deleted the full stop before
Kokoro ever saw it. Three consequences, all of them audible:

  * no sentence-final intonation, so replies were delivered as flat run-ons;
  * pipecat's streaming path regroups tokens into whole sentences before
    synthesis and finds those boundaries in the punctuation (tts_service.py: "the
    sequencer regroups tokens into a sentence"). With no punctuation to find, the
    text sits in the aggregator until a length or timeout rule fires -- so the
    first audio is late and the synthesis chunks are long, which is the other half
    of why replies felt slow;
  * the assistant aggregator sits DOWNSTREAM of this filter, so the transcript
    (the SAY lines in the log) and the conversation history were punctuation-free
    as well, and the model went on to imitate its own unpunctuated replies.

So this holds instead of dropping. A phoneme-less frame -- punctuation, a space, a
newline -- is appended to the frame currently being held, and a held frame is
released the moment the next frame of any kind arrives. Punctuation therefore
rides along with the word it follows (" JARVIS.") and spacing is left exactly as
it was; the only cost is one frame of delay. Non-text frames flush the held text
first, so frame order is untouched.

The one shape this cannot fix is a reply that begins with punctuation, which would
still release a phoneme-less fragment on its own; nothing attaches to it because
nothing precedes it.
"""

import unicodedata

from pipecat.frames.frames import Frame, TextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


def pronounceable(text: str) -> bool:
    """True if espeak-ng would find a phoneme in this fragment.

    Stripping alone is not enough. The model emits standalone punctuation as its
    own tokens -- a bare '†' or '-' passes .strip() and still reaches Kokoro as
    "Nothing to synthesize, '†' produced no phonemes", which shows up as an
    ErrorFrame in the log every time the model starts a list.

    The test is therefore whether the fragment contains any letter, digit, or
    ideograph -- categories a phonemiser can actually pronounce. Punctuation,
    symbols and whitespace are all rejected, while scripts without case (CJK)
    still pass via the "L" and "N" categories.
    """
    for ch in text:
        if unicodedata.category(ch)[0] in ("L", "N"):
            return True
    return False


class HoldPunctuationFilter(FrameProcessor):
    """Keeps punctuation attached to the word it follows, whatever it costs."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._held: TextFrame | None = None

    def _release(self) -> TextFrame | None:
        held, self._held = self._held, None
        return held

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame):
            if self._held is None:
                self._held = frame
            elif pronounceable(frame.text):
                await self.push_frame(self._release(), direction)
                self._held = frame
            else:
                # Mutated in place rather than replaced: dataclasses.replace() hands
                # back a fresh frame id, and the sequencer pairs frames by id.
                self._held.text = self._held.text + frame.text
            return

        # Any other frame has to wait for the held text, or the order changes.
        held = self._release()
        if held is not None:
            await self.push_frame(held, direction)
        await self.push_frame(frame, direction)
