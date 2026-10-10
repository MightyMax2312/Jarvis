"""Generate spoken test prompts with Kokoro, for feeding the agent's own loop."""
import sys

import numpy as np
import soundfile as sf
import torch
from kokoro import KPipeline

LINES = {
    "open_tab": "Please open a new browser tab for me.",
    "next_tab": "Switch to the next tab please.",
    "back": "Go back a page.",
    "windows": "What windows do I currently have open?",
    "apps": "What apps do I have open?",
    "focus_app": "Switch to Windows Terminal please.",
    "mute": "Mute the volume please.",
    "scroll": "Scroll down please.",
    "see": "What do you see on my screen?",
    "thanks": "Thanks, that is all I needed.",
    "capital": "What is the capital of France?",
}

name = sys.argv[1] if len(sys.argv) > 1 else "open_tab"
out = sys.argv[2] if len(sys.argv) > 2 else f"/tmp/prompt_{name}.wav"

text = LINES.get(name)
if text is None:
    raise SystemExit(f"unknown prompt {name!r}; have {sorted(LINES)}")

pipe = KPipeline(lang_code="a", device="cpu", repo_id="hexgrad/Kokoro-82M")
chunks = [a.cpu().numpy() for _, _, a in pipe(text, voice="af_heart", speed=1.0)]
audio = np.concatenate(chunks).astype(np.float32)
sf.write(out, audio, 24000, subtype="PCM_16")
print(f"wrote {out}  {len(audio)/24000:.2f}s  {text!r}")
