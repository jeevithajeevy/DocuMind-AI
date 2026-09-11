# DocuMind AI V2

A Streamlit AI document assistant with:

- Multiple PDF upload
- PDF text extraction
- Page-aware chunking
- Semantic embeddings
- Local persistent vector index
- Semantic search
- RAG-style question answering
- Gemini AI integration
- Source/page citations
- Retrieval confidence indicator
- Follow-up chat memory
- Document library
- Document analysis
- Suggested questions
- Page Explorer
- Markdown/PDF report export
- Conversation export

## Install

Activate your existing virtual environment, then:

```powershell
pip install -r requirements.txt
```

## Gemini API key

The app works in retrieval-only mode without a Gemini key.

To enable generated AI answers, set the key in PowerShell for the current terminal:

```powershell
$env:GEMINI_API_KEY="YOUR_API_KEY"
```

Then run:

```powershell
streamlit run app.py
```

For a persistent Streamlit secret, create:

`.streamlit/secrets.toml`

with:

```toml
GEMINI_API_KEY = "YOUR_API_KEY"
```

Do not upload or commit that file publicly.

## Run

```powershell
streamlit run app.py
```

The first run downloads the embedding model `all-MiniLM-L6-v2`.

## Project structure

```text
DocuMind_AI_V2/
├── app.py
├── requirements.txt
├── README.md
└── data/
```

The `data/` folder stores the local document index.
