@echo off
rem one-time setup on windows. double-click it, or run it from this folder:  setup.bat
rem makes .venv, installs everything, makes .env from .env.example. then: run.bat
where python >nul 2>nul || (echo python not found. install python 3.11+ from python.org and tick "add to PATH" & pause & exit /b 1)
if not exist .venv python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip -q
pip install -r requirements.txt
if not exist .env copy .env.example .env >nul
echo.
echo done. next:
echo   1. open .env and paste a free GEMINI_API_KEY or GROQ_API_KEY (or set RAG_PROVIDER=ollama, see README)
echo      or skip it and paste a key in the app
echo   2. run.bat
pause
