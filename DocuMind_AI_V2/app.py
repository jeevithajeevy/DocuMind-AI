import os
import re
import hashlib
import pickle
from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit as st
import fitz  # PyMuPDF
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

try:
    from google import genai
except ImportError:
    genai = None

APP_NAME = "DocuMind AI"
DATA_DIR = Path("data")
INDEX_FILE = DATA_DIR / "document_index.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-3.8-flash"
GEMINI_FALLBACK_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
]
TOP_K = 5
CHUNK_WORDS = 180
OVERLAP_WORDS = 35

DATA_DIR.mkdir(exist_ok=True)

st.set_page_config(
    page_title=APP_NAME,
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: #080b12; }
    #MainMenu, footer { visibility: hidden; }
    header { background: transparent; }

    section[data-testid="stSidebar"] {
        background: #0d111a;
        border-right: 1px solid #1b2230;
    }

    .brand {
        font-size: 23px;
        font-weight: 750;
        color: #f8fafc;
        margin-bottom: 2px;
    }

    .muted { color: #7f899a; font-size: 12px; }

    .eyebrow {
        color: #71809a;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.6px;
        text-transform: uppercase;
    }

    .title {
        color: #f8fafc;
        font-size: 42px;
        font-weight: 750;
        line-height: 1.15;
        margin: 8px 0;
    }

    .subtitle {
        color: #8d97a8;
        font-size: 15px;
        margin-bottom: 25px;
    }

    .card {
        background: #0e141e;
        border: 1px solid #202a39;
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 14px;
    }

    .source-card {
        background: #0c121b;
        border: 1px solid #202a39;
        border-radius: 12px;
        padding: 12px 14px;
        margin: 7px 0;
    }

    .metric {
        background: #0e141e;
        border: 1px solid #202a39;
        border-radius: 13px;
        padding: 14px;
        text-align: center;
    }

    .metric-value {
        color: #f8fafc;
        font-size: 24px;
        font-weight: 750;
    }

    .metric-label {
        color: #7f899a;
        font-size: 11px;
    }

    .stButton > button {
        border-radius: 10px;
        border: 1px solid #222c3c;
        background: #111722;
        color: #d7deea;
    }

    .stButton > button:hover {
        border-color: #4b5b73;
        background: #151d2a;
    }

    [data-testid="stChatMessage"] {
        background: #101621;
        border: 1px solid #1d2737;
        border-radius: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(MODEL_NAME)


def empty_index():
    return {
        "documents": {},
        "chunks": [],
        "embeddings": np.empty((0, 384), dtype=np.float32),
    }


def load_index():
    if INDEX_FILE.exists():
        try:
            with INDEX_FILE.open("rb") as f:
                value = pickle.load(f)

            if (
                isinstance(value, dict)
                and "documents" in value
                and "chunks" in value
                and "embeddings" in value
                and getattr(value["embeddings"], "ndim", 0) == 2
                and (
                    value["embeddings"].shape[0] == len(value["chunks"])
                    or value["embeddings"].shape[0] == 0
                )
            ):
                return value
        except Exception:
            pass

    return empty_index()


def save_index(value):
    with INDEX_FILE.open("wb") as f:
        pickle.dump(value, f)


index = load_index()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def extract_pages(pdf_bytes):
    """Extract text robustly from uploaded PDFs.

    PyMuPDF is used first because it is more tolerant of PDFs that contain
    unusual/truncated stream structures. pypdf is kept as a fallback.
    """
    pages = []

    # Primary parser: PyMuPDF
    try:
        document = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text") or ""
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n{3,}", "\n\n", text).strip()
            pages.append({"page": page_number, "text": text})
        document.close()
        return pages
    except Exception:
        pages = []

    # Fallback parser: pypdf, with strict=False for slightly malformed PDFs
    try:
        reader = PdfReader(BytesIO(pdf_bytes), strict=False)
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n{3,}", "\n\n", text).strip()
            pages.append({"page": page_number, "text": text})
        return pages
    except Exception as exc:
        raise ValueError(
            "This PDF could not be read. The file may be damaged or incomplete. "
            f"PDF parser details: {exc}"
        ) from exc


def chunk_pages(pages, filename):
    chunks = []

    for page in pages:
        words = page["text"].split()

        if not words:
            continue

        start = 0
        chunk_number = 0

        while start < len(words):
            end = min(start + CHUNK_WORDS, len(words))
            text = " ".join(words[start:end]).strip()

            if text:
                chunks.append({
                    "id": f"{filename}-{page['page']}-{chunk_number}",
                    "document": filename,
                    "page": page["page"],
                    "text": text,
                })

            if end >= len(words):
                break

            start = max(end - OVERLAP_WORDS, start + 1)
            chunk_number += 1

    return chunks


def add_document(filename, pdf_bytes):
    file_hash = sha256_bytes(pdf_bytes)

    if file_hash in index["documents"]:
        return False, "This PDF is already in the workspace."

    pages = extract_pages(pdf_bytes)
    chunks = chunk_pages(pages, filename)

    if not chunks:
        return False, (
            "No selectable text was found. This PDF may be scanned/image-only. "
            "OCR can be added as a later upgrade."
        )

    model = load_embedding_model()

    embeddings = model.encode(
        [chunk["text"] for chunk in chunks],
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)

    index["documents"][file_hash] = {
        "hash": file_hash,
        "filename": filename,
        "pages": len(pages),
        "chunks": len(chunks),
        "size_mb": len(pdf_bytes) / (1024 * 1024),
    }

    index["chunks"].extend(chunks)

    if index["embeddings"].shape[0] == 0:
        index["embeddings"] = embeddings
    else:
        index["embeddings"] = np.vstack(
            [index["embeddings"], embeddings]
        )

    save_index(index)

    return True, {
        "pages": len(pages),
        "chunks": len(chunks),
    }


def retrieve(query, top_k=TOP_K, document_filter=None):
    if not index["chunks"] or index["embeddings"].shape[0] == 0:
        return []

    model = load_embedding_model()

    query_vector = model.encode(
        [query],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0].astype(np.float32)

    scores = index["embeddings"] @ query_vector

    candidates = []

    for i, chunk in enumerate(index["chunks"]):
        if document_filter and chunk["document"] != document_filter:
            continue

        candidates.append((i, float(scores[i])))

    candidates.sort(key=lambda item: item[1], reverse=True)

    results = []

    for i, score in candidates[:top_k]:
        item = dict(index["chunks"][i])
        item["score"] = score
        results.append(item)

    return results


def get_api_key():
    try:
        key = st.secrets.get("GEMINI_API_KEY")
        if key:
            return key
    except Exception:
        pass

    return os.getenv("GEMINI_API_KEY")


def ai_available():
    return bool(get_api_key()) and genai is not None


def generate_with_gemini(prompt):
    """Generate an answer with automatic Gemini model fallback.

    A 503/429/5xx error can be temporary. Try each supported stable
    Flash model before giving up so the app can continue working even
    when one model is overloaded.
    """
    if not ai_available():
        return None

    client = genai.Client(api_key=get_api_key())
    last_error = None

    for model_name in GEMINI_FALLBACK_MODELS:
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )

                text = (response.text or "").strip()
                if text:
                    return text

            except Exception as exc:
                last_error = exc
                error_text = str(exc).lower()

                # Retry only transient service/rate-limit errors.
                transient = any(
                    code in error_text
                    for code in ["503", "429", "500", "502", "504", "unavailable", "overloaded"]
                )

                if not transient:
                    break

                # Small backoff before the second attempt on the same model.
                if attempt == 0:
                    import time
                    time.sleep(1.5)

    # Returning None is important: the caller will use the working
    # retrieval-based fallback instead of displaying a raw API error.
    return None


def answer_question(question, sources, history):
    if not sources:
        return "I couldn't find relevant information in the selected document."

    context = "\n\n".join(
        f"[SOURCE {i} | {source['document']} | PAGE {source['page']}]\n"
        f"{source['text']}"
        for i, source in enumerate(sources, start=1)
    )

    recent_history = "\n".join(
        f"{item['role'].upper()}: {item['content']}"
        for item in history[-6:]
    )

    prompt = f"""
You are DocuMind AI, a professional document assistant.

Answer ONLY from the supplied document context.

Rules:
- Do not invent facts.
- Answer the user's question directly.
- Use previous conversation context for follow-up questions.
- Cite factual statements with [Page X] or [Document Name, Page X].
- If several pages support an answer, cite all relevant pages.
- If the answer is not present in the context, say so.
- Use clean professional formatting.
- Prefer bullets when they improve readability.

CONVERSATION:
{recent_history}

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}
"""

    return generate_with_gemini(prompt)


def fallback_answer(question, sources):
    if not sources:
        return "I couldn't find relevant information."

    output = [
        "**Relevant information found:**",
        "",
    ]

    for source in sources[:3]:
        output.append(
            f"- **{source['document']} — Page {source['page']}**: "
            f"{source['text'][:500]}..."
        )

    output.extend([
        "",
        "_Semantic retrieval is working. Add a GEMINI_API_KEY to enable "
        "natural-language AI answers._",
    ])

    return "\n".join(output)


def analyze_document(filename):
    chunks = [
        chunk
        for chunk in index["chunks"]
        if chunk["document"] == filename
    ]

    if not chunks:
        return "No indexed content was found."

    text = "\n\n".join(chunk["text"] for chunk in chunks)
    text = text[:30000]

    prompt = f"""
Analyze this document professionally.

Return:
## Executive Summary
## Main Objective
## Key Concepts
## Methodology / Approach
## Important Findings
## Advantages
## Limitations
## Future Scope
## Conclusion

Use ONLY the supplied document. Do not invent information.

DOCUMENT:
{text}
"""

    result = generate_with_gemini(prompt)

    if result:
        return result

    return (
        "### Document indexed successfully\n\n"
        f"**Document:** {filename}\n\n"
        f"**Indexed chunks:** {len(chunks)}\n\n"
        "Add `GEMINI_API_KEY` to generate the full AI analysis."
    )


def build_markdown_report(filename):
    analysis = analyze_document(filename)

    return f"""# DocuMind AI — Document Analysis

**Document:** {filename}

{analysis}
"""


def create_pdf_report(title, text):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
    )
    from reportlab.lib.enums import TA_LEFT

    output = BytesIO()
    document = SimpleDocTemplate(
        output,
        pagesize=A4,
        rightMargin=45,
        leftMargin=45,
        topMargin=45,
        bottomMargin=45,
    )

    styles = getSampleStyleSheet()
    story = [
        Paragraph(title, styles["Title"]),
        Spacer(1, 12),
    ]

    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue

        safe = (
            block.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

        safe = safe.replace("\n", "<br/>")

        story.append(Paragraph(safe, styles["BodyText"]))
        story.append(Spacer(1, 8))

    document.build(story)

    return output.getvalue()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        '<div class="brand">📄 DocuMind AI</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="muted">Intelligent Document Workspace</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### 📥 Upload PDFs")

    uploads = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploads:
        for uploaded in uploads:
            if st.button(
                f"Process • {uploaded.name}",
                key=f"process_{uploaded.name}",
                use_container_width=True,
            ):
                with st.spinner(
                    f"Extracting, chunking and indexing {uploaded.name}..."
                ):
                    success, result = add_document(
                        uploaded.name,
                        uploaded.getvalue(),
                    )

                if success:
                    st.success(
                        f"Indexed {result['pages']} pages and "
                        f"{result['chunks']} chunks."
                    )
                    st.rerun()
                else:
                    st.warning(result)

    st.markdown("### 📚 Document Library")

    documents = list(index["documents"].values())

    if documents:
        for document in documents:
            st.markdown(
                f"**📄 {document['filename']}**  \n"
                f"<span class='muted'>"
                f"{document['pages']} pages • "
                f"{document['chunks']} chunks"
                f"</span>",
                unsafe_allow_html=True,
            )
    else:
        st.caption("No documents indexed yet.")

    if st.button(
        "🗑️ Clear document library",
        use_container_width=True,
    ):
        index = empty_index()
        save_index(index)
        st.session_state.pop("messages", None)
        st.rerun()

    st.markdown("### ⚙️ System")

    st.caption(
        "AI generation: "
        + ("Connected" if ai_available() else "Not configured")
    )
    st.caption(f"Embeddings: {MODEL_NAME}")
    st.caption(f"LLM: {GEMINI_MODEL} + automatic fallback")

# ============================================================
# MAIN
# ============================================================

st.markdown(
    '<div class="eyebrow">AI DOCUMENT WORKSPACE</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="title">AI Document Assistant</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Understand documents faster. Ask questions, discover insights, "
    "and trace answers back to source pages."
    "</div>",
    unsafe_allow_html=True,
)

documents = list(index["documents"].values())

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(
        f'<div class="metric"><div class="metric-value">'
        f'{len(documents)}</div><div class="metric-label">'
        "DOCUMENTS</div></div>",
        unsafe_allow_html=True,
    )

with m2:
    st.markdown(
        f'<div class="metric"><div class="metric-value">'
        f'{len(index["chunks"])}</div><div class="metric-label">'
        "INDEXED CHUNKS</div></div>",
        unsafe_allow_html=True,
    )

with m3:
    st.markdown(
        f'<div class="metric"><div class="metric-value">'
        f'{len(index["embeddings"])}</div><div class="metric-label">'
        "VECTORS</div></div>",
        unsafe_allow_html=True,
    )

with m4:
    st.markdown(
        f'<div class="metric"><div class="metric-value">'
        f'{"ON" if ai_available() else "OFF"}</div>'
        '<div class="metric-label">AI GENERATION</div></div>',
        unsafe_allow_html=True,
    )

st.write("")

if not documents:
    st.markdown(
        """
        <div class="card">
            <h3>👋 Welcome to DocuMind AI</h3>
            <p class="muted">
                Upload a PDF from the sidebar. DocuMind AI will extract the
                text, split it into meaningful chunks, create semantic
                embeddings and make the content searchable.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            "### 📄 Upload & Analyze\n"
            "Extract and index PDF content."
        )

    with c2:
        st.markdown(
            "### 🧠 Semantic Search\n"
            "Find relevant passages by meaning."
        )

    with c3:
        st.markdown(
            "### 🔎 Source Citations\n"
            "Trace answers back to pages."
        )

else:
    selected_document = st.selectbox(
        "Document scope",
        ["All documents"] + [
            document["filename"] for document in documents
        ],
    )

    document_filter = (
        None
        if selected_document == "All documents"
        else selected_document
    )

    tab_chat, tab_analyze, tab_search, tab_page = st.tabs(
        [
            "💬 Ask Questions",
            "✨ Analyze",
            "🔎 Search",
            "📖 Page Explorer",
        ]
    )

    # --------------------------------------------------------
    # CHAT
    # --------------------------------------------------------

    with tab_chat:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        st.markdown("**Suggested questions**")

        suggestions = [
            "What is the main objective?",
            "Summarize the important points.",
            "What methodology is used?",
            "What are the key findings?",
        ]

        suggestion_columns = st.columns(4)

        for i, suggestion in enumerate(suggestions):
            with suggestion_columns[i]:
                if st.button(
                    suggestion,
                    key=f"suggestion_{i}",
                    use_container_width=True,
                ):
                    st.session_state["pending_question"] = suggestion
                    st.rerun()

        pending_question = st.session_state.pop(
            "pending_question",
            None,
        )

        question = st.chat_input(
            "Ask something about your document..."
        )

        question = question or pending_question

        if question:
            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": question,
                }
            )

            sources = retrieve(
                question,
                TOP_K,
                document_filter,
            )

            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                if not sources:
                    answer = (
                        "I couldn't find relevant information in "
                        "the selected document."
                    )

                else:
                    answer = answer_question(
                        question,
                        sources,
                        st.session_state.messages[:-1],
                    )

                    if answer is None:
                        answer = fallback_answer(
                            question,
                            sources,
                        )

                st.markdown(answer)

                if sources:
                    best_score = max(
                        source["score"]
                        for source in sources
                    )

                    retrieval_confidence = int(
                        np.clip(
                            (best_score + 1) * 50,
                            0,
                            100,
                        )
                    )

                    st.caption(
                        f"Retrieval confidence: "
                        f"{retrieval_confidence}% "
                        "(semantic similarity indicator, "
                        "not an AI probability)"
                    )

                    st.markdown("**📚 Sources**")

                    for source in sources:
                        st.markdown(
                            f"""
                            <div class="source-card">
                                <strong>📄 {source['document']}</strong>
                                — Page <strong>{source['page']}</strong>
                                <br><br>
                                <span class="muted">
                                {source['text'][:700]}
                                </span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                }
            )

    # --------------------------------------------------------
    # ANALYSIS
    # --------------------------------------------------------

    with tab_analyze:
        if document_filter is None:
            st.info(
                "Select one specific document from "
                "Document scope above."
            )
        else:
            st.markdown(
                f"### ✨ Full Analysis — {document_filter}"
            )

            if st.button(
                "Generate professional document analysis",
                use_container_width=True,
            ):
                with st.spinner("Analyzing document..."):
                    report = build_markdown_report(
                        document_filter
                    )

                st.markdown(report)

                st.download_button(
                    "⬇️ Download Markdown Report",
                    data=report,
                    file_name=(
                        f"{Path(document_filter).stem}"
                        "_analysis.md"
                    ),
                    mime="text/markdown",
                    use_container_width=True,
                )

                try:
                    pdf_bytes = create_pdf_report(
                        f"DocuMind AI — {document_filter}",
                        report,
                    )

                    st.download_button(
                        "📄 Download PDF Report",
                        data=pdf_bytes,
                        file_name=(
                            f"{Path(document_filter).stem}"
                            "_analysis.pdf"
                        ),
                        mime="application/pdf",
                        use_container_width=True,
                    )
                except Exception as exc:
                    st.warning(
                        f"PDF export unavailable: {exc}"
                    )

    # --------------------------------------------------------
    # SEARCH
    # --------------------------------------------------------

    with tab_search:
        search_query = st.text_input(
            "Search the knowledge base",
            placeholder=(
                "Example: deepfake detection methodology"
            ),
        )

        if search_query:
            results = retrieve(
                search_query,
                10,
                document_filter,
            )

            for result in results:
                st.markdown(
                    f"""
                    <div class="source-card">
                        <strong>{result['document']}</strong>
                        — Page {result['page']}
                        — Similarity {result['score']:.2f}
                        <br><br>
                        {result['text']}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # --------------------------------------------------------
    # PAGE EXPLORER
    # --------------------------------------------------------

    with tab_page:
        if document_filter is None:
            st.info(
                "Select one document to explore its extracted pages."
            )
        else:
            page_numbers = sorted({
                chunk["page"]
                for chunk in index["chunks"]
                if chunk["document"] == document_filter
            })

            if page_numbers:
                page = st.selectbox(
                    "Select page",
                    page_numbers,
                )

                page_chunks = [
                    chunk["text"]
                    for chunk in index["chunks"]
                    if (
                        chunk["document"] == document_filter
                        and chunk["page"] == page
                    )
                ]

                st.markdown(
                    f"### 📖 {document_filter} — Page {page}"
                )

                st.text_area(
                    "Extracted page text",
                    value="\n\n".join(page_chunks),
                    height=500,
                )

# ============================================================
# EXPORT CHAT
# ============================================================

if st.session_state.get("messages"):
    chat_text = "\n\n".join(
        f"{message['role'].upper()}: "
        f"{message['content']}"
        for message in st.session_state.messages
    )

    st.download_button(
        "⬇️ Export Conversation",
        data=chat_text,
        file_name="documind_conversation.txt",
        mime="text/plain",
    )
