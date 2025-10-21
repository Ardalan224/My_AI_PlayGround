# app.py — Personal Diary Chatbot (cleaned)

import re
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# ---------- App & paths ----------
APP_NAME = "ArdaBrain"
APP_VERSION = "v0.1"

st.set_page_config(
    page_title=f"{APP_NAME} {APP_VERSION}", page_icon="🧠", layout="wide"
)


BASE_DIR = Path(__file__).parent.resolve()
DIARY_DIR = BASE_DIR / "diary"
VECTOR_DIR = BASE_DIR / ".vectordb"  # Chroma index stored here
DIARY_DIR.mkdir(exist_ok=True)
VECTOR_DIR.mkdir(exist_ok=True)

# ---------- Header ----------
st.title(f"🧠 {APP_NAME} {APP_VERSION}")
st.caption("Local, private, RAG-powered chat over your daily notes.")
with st.expander("How to use", expanded=False):
    st.markdown(
        """
1. Put your daily notes as `.txt` files in the `diary/` folder.
   **Naming**: `DD.MM.YYYY.txt` (e.g., `19.10.2025.txt`).
2. Ask questions like:
   - *What did I do on 03.02.2025?*
   - *When did I take a bus?*
   - *Tell me a fun memory.*
3. Everything runs locally via your chosen LLM model. No external APIs.
        """
    )

# ---------- Sidebar controls ----------
with st.sidebar:
    st.header("Settings")
    model_tag = st.text_input("LLM model tag", value="llama3.1:8b")
    model_temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    embedding_model_name = st.text_input(
        "Embedding model tag", value="nomic-embed-text"
    )
    st.markdown("### Date filter")
    date_filter_mode = st.selectbox(
        "Limit search to:",
        ["No filter", "Last 7 days", "Last 30 days", "Custom range"],
        index=0,
    )
    custom_start = custom_end = None
    if date_filter_mode == "Custom range":
        custom_start = st.text_input("Start date (DD.MM.YYYY)", value="")
        custom_end = st.text_input("End date (DD.MM.YYYY)", value="")


# ---------- Model factories (cached) ----------
@st.cache_resource(show_spinner=False)
def get_llm(model_name: str, temperature: float = 0.2):
    """Create and cache a local chat model via Ollama."""
    return ChatOllama(model=model_name, temperature=temperature)


@st.cache_resource(show_spinner=False)
def get_embeddings(model_name: str):
    """Create and cache a local embedding model via Ollama."""
    return OllamaEmbeddings(model=model_name)


llm = get_llm(model_tag, model_temperature)
embeddings = get_embeddings(embedding_model_name)


# ---------- Diary loading ----------
@st.cache_data(show_spinner=False)
def load_diary_files(diary_dir: Path):
    """Return [(filename, text), ...] for all non-empty .txt files."""
    items = []
    for fp in sorted(diary_dir.glob("*.txt")):
        try:
            txt = fp.read_text(encoding="utf-8").strip()
            if txt:
                items.append((fp.name, txt))
        except Exception as e:
            st.warning(f"Could not read {fp.name}: {e}")
    return items


diary_pairs = load_diary_files(DIARY_DIR)


# ---------- Parse date from filename (DD.MM.YYYY) ----------
def parse_date_from_filename(fname: str):
    """Extract DD.MM.YYYY from filenames like 'DD.MM.YYYY.txt' / 'DD_MM_YYYY.txt'."""
    m = re.search(r"(\d{2})[._-](\d{2})[._-](\d{4})", fname)
    if not m:
        return None
    d, mo, y = m.groups()
    try:
        return date(int(y), int(mo), int(d)).strftime("%d.%m.%Y")
    except ValueError:
        return None


@st.cache_data(show_spinner=False)
def to_docs_with_metadata(diary_pairs):
    """[{content, date (DD.MM.YYYY)|None, source}]"""
    return [
        {"content": text, "date": parse_date_from_filename(fname), "source": fname}
        for fname, text in diary_pairs
    ]


docs = to_docs_with_metadata(diary_pairs)

# ---------- Chunking (fixed settings) ----------
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


@st.cache_data(show_spinner=False)
def chunk_docs(docs):
    """Split each doc into overlapping chunks; keep metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    out = []
    for d in docs:
        parts = splitter.split_text(d["content"])
        for i, part in enumerate(parts):
            out.append(
                {
                    "content": part,
                    "date": d.get("date"),
                    "source": d.get("source"),
                    "chunk_id": i,
                }
            )
    return out


chunked = chunk_docs(docs) if docs else []

# ---------- Convert chunks -> LangChain Documents ----------
try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.docstore.document import Document  # type: ignore


@st.cache_data(show_spinner=False)
def to_lc_documents(chunks):
    """LangChain Documents with display, ISO, and ordinal (int) dates in metadata."""

    def to_iso(ddmmyyyy):
        if not ddmmyyyy:
            return None
        try:
            return datetime.strptime(ddmmyyyy, "%d.%m.%Y").date().isoformat()
        except Exception:
            return None

    def to_ord(ddmmyyyy):
        """DD.MM.YYYY -> int YYYYMMDD (e.g., 20251015), or None."""
        if not ddmmyyyy:
            return None
        try:
            return int(datetime.strptime(ddmmyyyy, "%d.%m.%Y").strftime("%Y%m%d"))
        except Exception:
            return None

    lc_docs = []
    for c in chunks:
        dd = c.get("date")  # DD.MM.YYYY
        lc_docs.append(
            Document(
                page_content=c["content"],
                metadata={
                    "date": dd,  # e.g., 03.02.2025 (for display)
                    "date_iso": to_iso(dd),  # e.g., 2025-02-03 (string)
                    "date_ord": to_ord(dd),  # e.g., 20250203 (INT for filtering)
                    "source": c.get("source"),
                    "chunk_id": c.get("chunk_id"),
                },
            )
        )
    return lc_docs


lc_docs = to_lc_documents(chunked)


# ---------- Build / load vector index (once per session) ----------
def build_vector_index(documents, embedding_fn, persist_dir: Path):
    """Fresh build of Chroma from documents; persists to disk and returns the store."""
    vs = Chroma.from_documents(
        documents=documents,
        embedding=embedding_fn,
        persist_directory=str(persist_dir),
    )
    vs.persist()
    return vs


@st.cache_resource(show_spinner=False)
def open_vectorstore(persist_dir: Path, embedding_fn):
    """Open an existing Chroma index from disk."""
    return Chroma(persist_directory=str(persist_dir), embedding_function=embedding_fn)


# Build the index only once per session; reuse thereafter
if "vectorstore" not in st.session_state:
    with st.spinner("Indexing diary into Chroma…"):
        vs = build_vector_index(lc_docs, embeddings, VECTOR_DIR)
        st.session_state["vectorstore"] = vs
        st.success(f"Indexed {len(lc_docs)} chunks.")

# Use the session-scoped instance everywhere below
vectorstore = st.session_state["vectorstore"]


# ---------- Date utilities & filter ----------
def today_berlin():
    return datetime.now(ZoneInfo("Europe/Berlin")).date()


def last_n_days_range(n: int):
    end_d = today_berlin()
    start_d = end_d - timedelta(days=n - 1)
    return (start_d, end_d)


def parse_ddmmyyyy(s: str):
    try:
        return datetime.strptime(s, "%d.%m.%Y").date()
    except Exception:
        return None


def build_chroma_date_filter(
    mode: str, custom_start: str | None, custom_end: str | None
):
    """Return a Chroma filter dict over 'date_ord' (int), or None."""
    if mode == "No filter":
        return None

    if mode == "Last 7 days":
        start_d, end_d = last_n_days_range(7)
    elif mode == "Last 30 days":
        start_d, end_d = last_n_days_range(30)
    elif mode == "Custom range":
        sd = parse_ddmmyyyy(custom_start or "")
        ed = parse_ddmmyyyy(custom_end or "")
        if not sd or not ed:
            return None
        if sd > ed:
            sd, ed = ed, sd
        start_d, end_d = sd, ed
    else:
        return None

    start_ord = int(start_d.strftime("%Y%m%d"))
    end_ord = int(end_d.strftime("%Y%m%d"))

    # One operator per clause; combine with $and
    return {
        "$and": [
            {"date_ord": {"$gte": start_ord}},
            {"date_ord": {"$lte": end_ord}},
        ]
    }


# ---------- Prompt building ----------
def format_context(docs):
    """Label snippets; expose DATE_STR for verbatim copying."""
    lines = []
    for i, d in enumerate(docs, start=1):
        meta = d.metadata or {}
        date_str = meta.get("date", "N/A")
        src = meta.get("source", "unknown")
        snippet = d.page_content.strip().replace("\n", " ")
        lines.append(f"[{i}] DATE_STR={date_str} | SOURCE={src} | TEXT={snippet}")
    return "\n".join(lines)


def build_prompt(user_q, docs):
    """First-person assistant; copy DATE_STR exactly (DD.MM.YYYY)."""
    context_block = format_context(docs)
    return (
        "You are my private memory assistant. Speak in first person as if these are your own memories.\n"
        "Use ONLY the provided diary snippets as your memory. Do not invent details.\n"
        "Guidelines:\n"
        " - Answer in 'I' voice (e.g., 'I went...', 'I did...').\n"
        " - If you mention a date, COPY the exact DATE_STR value verbatim (format: DD.MM.YYYY). Do NOT convert to month names.\n"
        " - If the answer isn't in the snippets, say: \"I don't remember this.\"\n"
        " - Prefer mentioning specific DATE_STR values when relevant.\n"
        " - Keep answers brief and factual.\n"
        " - Avoid phrases like 'according to the snippets'—just answer as me.\n\n"
        f"MY DIARY SNIPPETS:\n{context_block}\n\n"
        f"QUESTION: {user_q}\n"
        "FIRST-PERSON ANSWER:"
    )


def push_welcome():
    welcome = (
        "Hi. I’m Ardalan — only the cyber version 🤖️ What do you want to know about me?"
    )
    st.session_state.chat.append({"role": "assistant", "content": welcome})


# ---------- Chat UI ----------
if "chat" not in st.session_state:
    st.session_state.chat = []
    push_welcome()  # show greeting at the very start of the session


with st.sidebar:
    if st.button("🧹 Clear chat history"):
        st.session_state["chat"] = []
        push_welcome()  # greet right after clearing
        st.success("Chat history cleared.")


# Render history
for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Input
user_msg = st.chat_input("Ask about your notes (e.g., “What did I do on 03.02.2025?”)")
RAG_TOP_K = 3

if user_msg:
    st.session_state.chat.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.write(user_msg)

    with st.chat_message("assistant"):
        with st.spinner("Thinking with your diary…"):
            try:
                date_filter = build_chroma_date_filter(
                    date_filter_mode, custom_start, custom_end
                )
                retrieved = vectorstore.max_marginal_relevance_search(
                    user_msg,
                    k=RAG_TOP_K,  # final number of results you want
                    fetch_k=32,  # candidate pool to diversify from
                    lambda_mult=0.5,  # 0 = more diversity, 1 = more similarity
                    filter=date_filter,
                )

                prompt_text = build_prompt(user_msg, retrieved)
                resp = llm.invoke(prompt_text)
                answer = resp.content
                st.write(answer)

                # Show sources for transparency
                if retrieved:
                    st.markdown("**Sources**")
                    for i, d in enumerate(retrieved, start=1):
                        meta = d.metadata or {}
                        st.markdown(
                            f"- **[{i}]** `{meta.get('source', 'unknown')}` — `{meta.get('date', 'N/A')}`"
                        )

                st.session_state.chat.append({"role": "assistant", "content": answer})

            except Exception as e:
                err = f"RAG chat failed: {e}"
                st.error(err)
                st.session_state.chat.append({"role": "assistant", "content": err})
