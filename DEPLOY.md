# Deploying CivilLens

## 1. Local (laptop)
```
pip install -r requirements.txt
python train.py
streamlit run app.py
```
Windows: double-click `run_windows.bat`.

## 2. Docker
```
docker build -t civillens .
docker run -p 8501:8501 -e LLM_PROVIDER=gemini -e GEMINI_API_KEY=<your key> civillens
```
Open http://localhost:8501. The image trains the toxicity model at build time and has a health check.

## 3. Streamlit Community Cloud (free public URL)
1. Push this folder to a GitHub repository (the `.gitignore` keeps `.env` and keys out).
2. Go to share.streamlit.io, click **Create app**, pick the repo and `app.py`.
3. In **App settings -> Secrets**, paste the contents of `.streamlit/secrets.toml.example` with your real key.
4. Deploy. The model is trained automatically on first start (about 5 seconds).

## Safety checklist
- API keys only in `.env`, Docker `-e` variables or Streamlit Secrets, never in code.
- Toxic words are masked in the UI and before any LLM call.
- If the LLM fails (quota, retired model, no internet) the app falls back to offline models automatically.
