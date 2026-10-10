"""Verify the agent's TTS output is real speech by transcribing it back."""
import sys

import numpy as np
import requests
import soundfile as sf

path = sys.argv[1] if len(sys.argv) > 1 else "replay_out.wav"
data, sr = sf.read(path, dtype="float32")
data = data.reshape(-1)
print(f"input: {path}  {len(data)/sr:.2f}s @ {sr} Hz  peak={np.abs(data).max():.3f} "
      f"rms={np.sqrt((data**2).mean()):.4f}")

# The model was asked: "what is the capital of France?" -- so "Paris" should
# appear. /api/stt resamples internally, so hand it whatever rate we have.
for label, chunk in (("full", data), ("first 2s", data[: int(sr * 2)]),
                     ("last 2s", data[-int(sr * 2):])):
    r = requests.post("http://127.0.0.1:8080/api/stt", data=chunk.tobytes(),
                      headers={"Content-Type": "application/octet-stream",
                               "X-Sample-Rate": str(sr)}, timeout=120)
    print(f"  {label:10} -> {r.json()['text']!r}")
