"""Pipecat pipeline for the JARVIS voice agent.

Latency-relevant choices, all measured on this box:

  Silero VAD stop_secs   1.0s -> 0.2s. A pure-VAD endpoint is the eot-bench
                         baseline at ~1600 ms with 55.6% false cutoffs. Pipecat
                         warns if stop_secs != 0.2 when a turn analyzer is
                         attached, because the p99 STT budget is built on it.
  Smart Turn v3          8 MB ONNX, ~13 ms on CPU, 94.31% English accuracy.
                         Bundled in the pipecat-ai wheel; onnxruntime is a core
                         dependency, so no extra install.
  Parakeet STT           ~0.30 s for a 3.8 s utterance, CPU, off the audio path.
  Ollama 4B              ~460 ms TTFT, 38.5 tok/s. MUST send think=false.
  Kokoro                 CPU-only ONNX. Never load it on CUDA: 731 MiB there
                         starves the router and the talker and causes 36 s turns.
"""

import asyncio
import os

from loguru import logger

from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.kokoro.tts import KokoroTTSService, KokoroTTSSettings
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.services.settings import STTSettings
from pipecat.services.tts_service import TextAggregationMode
from pipecat.transports.websocket.server import (
    SingleClientWebsocketServerParams,
    SingleClientWebsocketServerTransport,
)
from pipecat.turns.user_start import (
    TranscriptionUserTurnStartStrategy,
    VADUserTurnStartStrategy,
)
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from .parakeet_stt import ParakeetHTTPSTTService
from .prewarm import prewarm_all
from .raw_pcm import RawPCMSerializer
from .router import KevRouter
from .router_fastpath import RouterFastPath
from .spy import InputSpy, OutputSpy
from .text_filters import HoldPunctuationFilter
from .tools import build_tools

STT_URL = os.environ.get("JARVIS_STT_URL", "http://127.0.0.1:8080/api/stt")
OLLAMA_BASE = os.environ.get("JARVIS_OLLAMA_BASE", "http://127.0.0.1:11434/v1")
OLLAMA_MODEL = os.environ.get("JARVIS_LLM_MODEL", "jarvis-llm")

# Set JARVIS_FASTPATH=0 to turn the router fast path off and route every turn
# through the LLM, which is the honest way to A/B the latency it buys.
FASTPATH_ENABLED = os.environ.get("JARVIS_FASTPATH", "1") != "0"
FASTPATH_THRESHOLD = float(os.environ.get("JEV_CONFIDENCE_THRESHOLD", "0.70"))

# 0 disables the idle self-shutdown entirely, which is what a long-lived desk
# agent wants. Set JARVIS_IDLE_TIMEOUT to a number to get the default teardown
# back for testing.
IDLE_TIMEOUT_SECS = float(os.environ.get("JARVIS_IDLE_TIMEOUT", "0")) or None

SYSTEM_PROMPT = """You are JARVIS, a voice assistant running on the user's own computer.

You are speaking out loud through a speech synthesiser. Therefore:
- Reply with plain spoken words only. ONE or TWO short sentences, nothing more.
- NEVER output code, Python, markdown, bullet points, code blocks, emoji, or URLs.
- When the user asks you to control the computer, call the matching tool. Do not
  merely say you will do it.
- After a tool returns, tell the user in one short sentence what ACTUALLY
  happened, naming the app if the tool mentioned one. For example if a tool says
  it opened a tab in Edge, say "Opened a new tab in Edge." Never reply with just
  "Done" -- say what was done.
- If a tool returns an error, tell the user plainly what went wrong. Do not
  claim the action succeeded.
- If the request is ambiguous, ask one short question instead of guessing.
- Never claim to have done something you have not actually done with a tool.

Choosing a tool:
- run_action takes an "action" from a fixed list and a "target".
- Put whatever the action operates on in "target": the element to click, the app
  to switch to, the exact text to type, or the key to press. For actions that
  need nothing (open a tab, scroll down, mute), leave target empty.
- Changing tabs is NEXT_TAB or PREVIOUS_TAB. Going back or forward in history is
  GO_BACK or GO_FORWARD, not the tab actions.
- Switching to another program is FOCUS_APP with the app name as target. To find
  out what is open, call list_windows first so you use the real name.
- To answer a question about what is on screen, call describe_screen.
- To answer what is open, call run_action with LIST_APPS for applications or
  LIST_BROWSERS for browsers and their tabs; list_windows gives every title.
- Media and volume: MEDIA_PLAY_PAUSE, MEDIA_NEXT, MEDIA_PREV and MEDIA_STOP take
  no target. CLIPBOARD_READ reads the clipboard out; CLIPBOARD_WRITE puts the
  text in target onto it.
- To open a web address, OPEN_URL with the address as target. To save an image of
  the screen, SAVE_SCREENSHOT, with a short label in target if the user gave one.
- SNAP_LEFT and SNAP_RIGHT move a window to half the screen; target names the app
  and may be left empty for whatever is in front.
- To press enter after typing, call run_action with action PRESS_KEY and target
  "enter"."""

LATENCY_LOG = os.environ.get("JARVIS_LATENCY_LOG", "/tmp/jarvis-latency.log")


def build_pipeline(transport: SingleClientWebsocketServerTransport):
    stt = ParakeetHTTPSTTService(
        url=STT_URL,
        sample_rate=16000,
        # Measured ~0.30 s decode. The default is 1.0 s, which the turn-stop
        # strategy uses to size its fallback timer, so declaring the real number
        # lets it stop waiting sooner.
        ttfs_p99_latency=0.6,
        # A trailing pad so Parakeet does not clip the final word.
        trailing_silence_secs=0.4,
        # STTService requires every settings field to be initialised; None means
        # "this service does not use it".
        settings=STTSettings(
            model="parakeet-tdt-0.6b-v3-int8",
            language="en",
        ),
    )

    tts = KokoroTTSService(
        settings=KokoroTTSSettings(voice="af_heart", language="en", speed=1.05),
        # SENTENCE mode (the default) measured 0.694 s of pure waiting before
        # Kokoro was even called. TOKEN mode hands each token straight through,
        # which is safe here because Kokoro streams incrementally anyway.
        text_aggregation_mode=TextAggregationMode.TOKEN,
    )

    llm = OLLamaLLMService(
        base_url=OLLAMA_BASE,
        settings=OLLamaLLMService.Settings(
            model=OLLAMA_MODEL,
            temperature=0.7,
            max_tokens=120,
            system_instruction=SYSTEM_PROMPT,
            # jarvis-llm is a reasoning model. Pipecat's OLLamaLLMService talks to
            # /v1/chat/completions, which IGNORES Ollama's "think": false -- only
            # /api/chat honours it. Measured on this endpoint:
            #   no flag / think=false / chat_template_kwargs  -> content='' after
            #     3.7 s, all 120 tokens spent on <think>, finish_reason=length
            #   reasoning.effort = "none"                     -> real content in 0.6 s
            # It must be wrapped in extra_body: the OpenAI SDK rejects unknown
            # kwargs outright ("unexpected keyword argument 'reasoning'"), and
            # `extra` is merged into the request params by
            # BaseOpenAILLMService.build_chat_completion_params (base_llm.py:395).
            #
            # keep_alive: -1 pins the model in VRAM permanently. Ollama unloads
            # after 5 minutes by default, and paging a 3.4 GB model back in costs
            # 9-16 s -- measured 16.3 s for an un-pinned request, and it was the
            # single worst outlier in the latency log (LLM inference, max 39.3 s
            # against a 0.81 s median). The proper fix is OLLAMA_KEEP_ALIVE=-1 on
            # the ollama service, which needs sudo; this achieves the same thing
            # per request for the agent's own traffic.
            extra={"extra_body": {
                "reasoning": {"effort": "none"},
                "keep_alive": -1,
            }},
        ),
    )

    # The tool schemas must be advertised on the CONTEXT, and the handler must
    # live on the schema itself. Two mistakes produce a chat-only agent with
    # no error:
    #   * llm.register_function() alone -- the model never sees the schemas,
    #     because build_chat_completion_params takes tools from the context. The
    #     agent then answers "Opening a new tab now" without calling anything.
    #   * raw callables in LLMContext(tools=...) -- Pipecat treats those as
    #     *direct functions* (first arg must be named `params`), a
    #     Pipecat-specific protocol that Ollama's OpenAI-compatible endpoint
    #     does not speak.
    # A FunctionSchema carrying its own handler needs no register_function:
    # Pipecat warns the manual step is redundant.
    tool_schemas = build_tools()

    # No system message here: the prompt is passed as the service's
    # system_instruction, which is the non-deprecated form since 1.9.0.
    context = LLMContext(tools=tool_schemas)

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(
                # stop_secs 0.2 -> 0.4 after reading the live log: at 0.2 the VAD
                # was closing the utterance on the ordinary pause inside a
                # sentence, so the STT ran twice and produced two transcripts
                # ("Tell me in one short sentence" / "How you are doing today")
                # about a second apart. The conversation still worked -- the
                # aggregator holds the turn -- but the reply had to wait for the
                # second fragment, and the model then answered the first half
                # before it had heard the rest. 0.4 s still sits far below a
                # human's end-of-turn pause, and SmartTurn remains the thing that
                # decides a turn is over.
                params=VADParams(confidence=0.7, start_secs=0.2, stop_secs=0.4, min_volume=0.6)
            ),
            user_turn_strategies=UserTurnStrategies(
                start=[
                    VADUserTurnStartStrategy(enable_interruptions=True),
                    TranscriptionUserTurnStartStrategy(),
                ],
                stop=[
                    TurnAnalyzerUserTurnStopStrategy(
                        turn_analyzer=LocalSmartTurnAnalyzerV3(
                            cpu_count=2,
                            params=SmartTurnParams(
                                # Ceiling for an "incomplete" verdict. 3.0 s is the
                                # default; 1.5 s bounds worst-case dead air.
                                stop_secs=1.5,
                            ),
                        ),
                    ),
                ],
            ),
        ),
    )

    return Pipeline(
        [
            transport.input(),
            stt,
            # After the STT, so it sees the transcript rather than raw audio.
            InputSpy(),
            user_aggregator,
            # Must sit between the aggregator and the LLM: it swallows the
            # LLMContextFrame that would otherwise trigger inference when the
            # router is confident. After the aggregator, because it needs the
            # finished transcript; before the LLM, because forwarding that
            # frame is what starts the turn.
            RouterFastPath(
                router=KevRouter(threshold=FASTPATH_THRESHOLD),
                enabled=FASTPATH_ENABLED,
            ),
            llm,
            # Must sit between llm and tts. It holds punctuation-bearing fragments
            # so they arrive attached to the word they follow rather than being
            # dropped before Kokoro can use them for intonation; the measurement
            # that motivated the change is in text_filters.py.
            HoldPunctuationFilter(),
            tts,
            assistant_aggregator,
            # Last, so it sees every frame the transport is about to send:
            # the agent's spoken text, and where each turn starts and stops.
            OutputSpy(),
            transport.output(),
        ]
    )


def make_observer() -> UserBotLatencyObserver:
    """Per-stage latency attribution, so regressions are visible not guessed."""
    observer = UserBotLatencyObserver(min_contribution_secs=0.005)

    @observer.event_handler("on_latency_breakdown")
    async def _on_breakdown(observer, breakdown):
        lines = breakdown.turn_contribution_lines(by_cost=True)
        total = breakdown.total_secs
        logger.info("[latency] total {:.3f}s", total)
        for line in lines:
            logger.info("[latency]   {}", line.strip())
        with open(LATENCY_LOG, "a", encoding="utf-8") as fh:
            fh.write(f"total={total:.3f}\n")
            for line in lines:
                fh.write(f"  {line.strip()}\n")
            fh.write("\n")

    return observer


# Loopback by default: this websocket takes raw audio in and can make the
# machine act, with no authentication of any kind. This box is in WSL mirrored
# mode, so Windows reaches a 127.0.0.1 listener here directly (verified) and the
# desktop client keeps working. Pass --host 0.0.0.0 to piped.run deliberately if
# you ever want the agent reachable from the LAN.
async def run_bot(host: str = "127.0.0.1", port: int = 8765):
    transport = SingleClientWebsocketServerTransport(
        params=SingleClientWebsocketServerParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=24000,
            add_wav_header=False,
            # Mandatory: the transport drops every incoming message when no
            # serializer is set. There is no bundled raw-PCM serializer, so we
            # bring our own.
            serializer=RawPCMSerializer(input_sample_rate=16000),
        ),
        host=host,
        port=port,
    )

    # A connection is the moment somebody is about to speak, so that is when the
    # talker needs to be in memory. Prewarming on connect is what removes the dead
    # 10-30 s of silence on the first turn after the model was evicted; the
    # measurements behind that range are in piped/prewarm.py.
    @transport.event_handler("on_client_connected")
    async def _on_client_connected(transport, websocket):
        asyncio.create_task(prewarm_all("client connect", tts))

    pipeline = build_pipeline(transport)
    # Kokoro's ONNX session is built inside build_pipeline (services/kokoro/tts.py:212),
    # but its first inference is still the slow one, so the warm-up gets the service.
    tts = next((p for p in pipeline.processors if isinstance(p, KokoroTTSService)), None)
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,  # required for LatencyBreakdown
            audio_in_sample_rate=16000,
            audio_out_sample_rate=24000,
        ),
        # idle_timeout_secs is a PipelineWorker argument, NOT a PipelineParams
        # field. This matters because PipelineParams is a pydantic model, so the
        # mistake is silent: the unknown key is dropped, the worker keeps its own
        # 300 s default, and the agent still cancels itself five minutes after
        # the last thing anyone said. The log signature is
        #   WARNING | Idle timeout detected.
        #   WARNING | Idle pipeline detected, cancelling pipeline worker...
        # followed by the runner exiting -- which is what used to kill this agent
        # between uses. For a desk-side agent, "no traffic for five minutes" is
        # the normal state, not a fault.
        # IDLE_TIMEOUT_SECS is None unless JARVIS_IDLE_TIMEOUT sets a number, and
        # None disables the idle monitor entirely (worker.py: `if
        # self._idle_timeout_secs:`). A nonzero value restores the teardown, for
        # testing.
        idle_timeout_secs=IDLE_TIMEOUT_SECS,
        observers=[make_observer()],
        name="jarvis",
    )

    # Awaited rather than fired and forgotten: the websocket must not start
    # accepting connections before the talker can answer, or the "ready" line in
    # start-agent.sh means "listening, but 25 s away from being useful". When the
    # model is already resident this costs ~0.36 s (measured); after an eviction it
    # moves that load into startup, where start-agent.sh waits for it on purpose,
    # instead of onto the first thing somebody says. Failures are logged inside
    # prewarm_all and swallowed, so a broken Ollama cannot stop the agent from
    # starting -- it only makes the first turn slow, exactly as before.
    await prewarm_all("startup", tts)

    runner = PipelineRunner(name="jarvis", handle_sigint=True)
    logger.info(f"[bot] websocket listening on ws://{host}:{port}")
    try:
        await runner.run(worker)
    finally:
        pass
