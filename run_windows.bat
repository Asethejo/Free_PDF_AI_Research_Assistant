@echo off
cd /d %~dp0
if not exist .venv (
    python -m venv .venv
)
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
if not exist .streamlit\secrets.toml (
    copy .streamlit\secrets.toml.example .streamlit\secrets.toml
    echo Please edit .streamlit\secrets.toml and add your Groq API key, then rerun this file.
    pause
    exit /b
)
streamlit run app.py
pause
