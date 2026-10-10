#!/bin/bash
cd "$(dirname "$0")" || exit 1
for n in open_tab windows thanks; do
  ./jarvis-env/bin/python make_prompt.py "$n" 2>&1 | grep wrote
done
ls -la /tmp/prompt_*.wav
