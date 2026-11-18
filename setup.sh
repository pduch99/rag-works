#!/usr/bin/env bash
# one-time setup on mac / linux. from this folder:  bash setup.sh
# makes .venv, installs everything, makes .env from .env.example. then: bash run.sh
set -e
command -v python3 >/dev/null || { echo "python3 not found. install python 3.11+"; exit 1; }
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt
[ -f .env ] || cp .env.example .env
echo
echo "done. next:"
echo "  1. open .env and paste a free GEMINI_API_KEY or GROQ_API_KEY (or set RAG_PROVIDER=ollama, see README)"
echo "     or skip it and paste a key in the app"
echo "  2. bash run.sh"
