# Free PDF AI Research Assistant (Groq + Streamlit)

A lightweight RAG-style PDF Q&A app. Visitors upload PDFs, the app extracts text and retrieves relevant passages using TF-IDF, and Groq generates an answer with page citations. No Ollama installation is required for visitors.

## Features
- Upload one or multiple PDFs
- Page-level text extraction
- Overlapping chunks
- TF-IDF retrieval (no embedding model download)
- Groq chat completion
- Page/source citations in generated answers
- View retrieved passages
- Local run and Streamlit Community Cloud deployment

## Requirements
- Python 3.10–3.12 recommended
- A Groq API key (the API's free tier is subject to provider limits and availability)

## Run on Windows
1. Extract the ZIP.
2. Get an API key from https://console.groq.com/keys
3. Open `.streamlit/secrets.toml.example`, copy its contents into a new `.streamlit/secrets.toml` file.
4. Replace the placeholder with your API key.
5. Double-click `run_windows.bat`, or manually run these commands in the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501.

## Deploy to Streamlit Community Cloud
1. Create a new public GitHub repository.
2. Upload the project files to the repository. Do NOT upload `.streamlit/secrets.toml`; it is ignored by git.
3. Go to https://share.streamlit.io and create an app from your repository.
4. Select `app.py` as the entrypoint.
5. Open Advanced settings / Secrets and add:

```toml
GROQ_API_KEY = "your-key-here"
```

6. Deploy. The public link lets visitors upload PDFs and ask questions without installing Ollama or entering an API key.

## Limitations
- PDF extraction only supports selectable text; scanned PDFs need OCR.
- TF-IDF retrieval is lexical rather than deep semantic retrieval. It is lightweight and avoids downloading a large embedding model.
- Uploaded documents and the index are held in app memory during the session; the app does not persist them as a database.
- Groq usage is subject to free-tier quotas, model availability, and rate limits.
- This is a demo. Avoid uploading confidential documents to a public app.

## Security
Never commit API keys. Keep `.streamlit/secrets.toml` private. Do not expose your Groq key to users.
