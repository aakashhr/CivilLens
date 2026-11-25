@echo off
title CivilLens - Groq key setup
cd /d "%~dp0"
echo.
echo Paste your Groq API key (starts with gsk_) and press Enter.
echo Get one free at https://console.groq.com/keys
echo.
set /p GROQKEY=Groq API key: 
if "%GROQKEY%"=="" (
  echo No key entered. Nothing was changed.
  pause
  exit /b 1
)
(
  echo LLM_PROVIDER=groq
  echo GROQ_API_KEY=%GROQKEY%
  echo GROQ_MODEL=llama-3.3-70b-versatile
  echo TOXICITY_BACKEND=auto
  echo EMBEDDING_BACKEND=auto
) > .env
echo.
echo Saved to .env. Now double-click run_windows.bat (or restart the app if it is already running).
pause
