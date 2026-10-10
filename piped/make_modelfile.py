"""Generate the jarvis-llm Modelfile.

Thinking is NOT disabled here. A custom TEMPLATE block was tried -- rewriting
the `enable_thinking` gate to `true` -- but Ollama's template engine rejects the
round-tripped macro definitions with `function "content" not defined`, even
with the template text byte-identical. The working fix is per-request instead:
Pipecat passes {"reasoning": {"effort": "none"}} via the LLM service `extra`
field, which Ollama honours on /v1/chat/completions.
"""
import os

import requests

BASE_MODEL = "hf.co/HauhauCS/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive:Q4_K_M"
# Written beside this package's parent directory rather than into a hardcoded home
# path, so the published copy carries nobody's username.
OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Modelfile.jarvis-llm"
)

modelfile = f"""FROM {BASE_MODEL}

# ---------------------------------------------------------------------------
# Measured: RTX 3050 Laptop, 4096 MiB VRAM, Ollama 0.30.10
#
# VRAM budget:
#   kev.serve (bf16, resident)     ~  977 MiB
#   jarvis-llm (Q4_K_M, 32 blocks) ~ 2400 MiB at ctx 4096
# Kokoro runs CPU-only ONNX and holds nothing here -- putting it on CUDA cost
# 731 MiB and produced 36 s conversational turns.
#
# num_gpu 32 vs 20: all blocks on GPU measured 38.5 tok/s and 459 ms TTFT;
# 20 blocks (rest on CPU) measured 13.7 tok/s and 745 ms TTFT.

# 262144 native context needs ~1.07 GiB of KV cache. 4096 halves that and is
# ample for a voice turn: system prompt + tool schemas + a few exchanges.
PARAMETER num_ctx 4096
PARAMETER num_gpu 32

PARAMETER temperature 0.7
PARAMETER top_p 0.8
PARAMETER top_k 20
PARAMETER num_predict 120

# NOTE: this is a reasoning model. Ollama ignores "think": false on
# /v1/chat/completions (the endpoint Pipecat uses); without an explicit
# reasoning override every turn returns content=''. See piped/bot.py.
"""

with open(OUT, "w", encoding="utf-8") as fh:
    fh.write(modelfile)

print(f"wrote {OUT} ({len(modelfile)} bytes)")
