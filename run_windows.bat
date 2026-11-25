@echo off
title CivilLens
cd /d "%~dp0"
where python >/dev/null 2>nul
if errorlevel 1 (
  echo Python was not found. Install Python 3.10 or newer from python.org and tick "Add Python to PATH".
  pause
  exit /b 1
)
if not exist .venv (
  echo Creating virtual environment...
  python -m venv .venv
)
call .venv\Scripts\activate.bat
echo Installing packages (first run takes a few minutes)...
python -m pip install --upgrade pip >nul
pip install -r requirements.txt
if errorlevel 1 (
  echo Package install failed. Check your internet connection and try again.
  pause
  exit /b 1
)
if not exist models\toxicity_model.joblib python train.py
echo Starting CivilLens at http://localhost:8501 ...
streamlit run app.py
pause
