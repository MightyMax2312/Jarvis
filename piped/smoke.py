"""Smoke test: every service initialises before we involve a microphone.

The real end-to-end test (piped.e2e) drives a live pipeline; this only checks
that bundled weights resolve and the tool layer is wired.
"""
import asyncio
import time

import aiohttp

from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.transports.websocket.server import (
    SingleClientWebsocketServerParams,
    SingleClientWebsocketServerTransport,
)

from piped.bot import build_pipeline


async def main():
    print("=== bundled ONNX weights ===")
    t0 = time.time()
    st = LocalSmartTurnAnalyzerV3(cpu_count=2, params=SmartTurnParams(stop_secs=1.5))
    print(f"  SmartTurn v3 : {time.time()-t0:.3f}s")
    t0 = time.time()
    vad = SileroVADAnalyzer(params=VADParams(confidence=0.7, start_secs=0.2, stop_secs=0.2))
    print(f"  Silero VAD   : {time.time()-t0:.3f}s  frames_required={vad.num_frames_required()}")

    print("\n=== pipeline ===")
    transport = SingleClientWebsocketServerTransport(
        params=SingleClientWebsocketServerParams(
            audio_in_enabled=True, audio_out_enabled=True,
            audio_in_sample_rate=16000, audio_out_sample_rate=24000,
            add_wav_header=False,
        ),
        host="0.0.0.0", port=8765,
    )
    session = aiohttp.ClientSession()
    t0 = time.time()
    pipeline = build_pipeline(transport)
    print(f"  built in {time.time()-t0:.2f}s")
    print(f"  chain: {' -> '.join(type(p).__name__ for p in pipeline.processors)}")

    stt = next(p for p in pipeline.processors
               if type(p).__name__ == "ParakeetHTTPSTTService")
    print(f"\n  STT init sample_rate={stt._init_sample_rate} "
          f"(resolved to live rate in setup()), wants_wav={stt.wants_wav_segments}")

    print("\n=== tool validation (no client needed) ===")
    from pipecat.adapters.schemas.function_schema import FunctionSchema
    from pipecat.services.llm_service import FunctionCallParams
    from piped.tools import build_tools

    schemas = build_tools()
    print(f"  tools: {[s.name for s in schemas]}")
    for s in schemas:
        print(f"    {s.name}: required={s.required} "
              f"props={sorted(s.properties)} handler={s.handler.__name__}")
    print(f"  {schemas[0].description[:100]}...")

    await session.close()
    print("\nsmoke OK")


asyncio.run(main())
