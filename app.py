import streamlit as st
import os
from rag_pipeline import RAGPipeline

# ─────────────────────────────
# CONFIG
# ─────────────────────────────
st.set_page_config(page_title="IT Policy", layout="wide")
st.title("📄IT Policy Assistant")

# ─────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ─────────────────────────────
# LOAD PIPELINE (CACHED)
# ─────────────────────────────
@st.cache_resource
def load_pipeline():
    return RAGPipeline()

rag = load_pipeline()

# ─────────────────────────────
# SIDEBAR
# ─────────────────────────────
st.sidebar.header("⚙️ Controls")

debug_mode = st.sidebar.toggle("🧪 Debug Mode", value=False)
clear_db   = st.sidebar.checkbox("🗑️ Clear DB before ingest")

uploaded_file = st.sidebar.file_uploader("📂 Upload PDF", type=["pdf"])

# ── Ingest ──
if st.sidebar.button("🚀 Ingest PDF"):
    if uploaded_file:
        file_path = f"temp_{uploaded_file.name}"
        try:
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())

            with st.spinner("⏳ Ingesting..."):
                rag.ingest(file_path, clear_existing=clear_db)
                count = rag.store.verify()
                st.sidebar.success(f"✅ Ingested | Vectors: {count}")

        except Exception as e:
            st.sidebar.error(f"❌ Ingest failed: {e}")
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
    else:
        st.sidebar.warning("⚠️ Upload a PDF first.")

# ── DB Stats ──
if st.sidebar.button("📊 DB Stats"):
    try:
        count = rag.store.verify()
        st.sidebar.info(f"Total vectors: {count}")
    except Exception as e:
        st.sidebar.error(f"❌ Could not fetch stats: {e}")

# ── Reset Chat ──
if st.sidebar.button("🔄 Reset Chat"):
    st.session_state.messages = []
    st.rerun()

# ─────────────────────────────
# DISPLAY CHAT HISTORY
# ─────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ─────────────────────────────
# CHAT INPUT
# ─────────────────────────────
prompt = st.chat_input("Ask something about your PDF...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    answer = None

    with st.chat_message("assistant"):
        with st.spinner("🤖 Thinking..."):
            try:
                # Build history from session (exclude current user message)
                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]
                ]

                # FIX: Use pipeline's query_with_history — no duplicated
                # embed/retrieve logic in app.py; chunks returned for citations
                answer, chunks = rag.query_with_history(prompt, history=history)

                if not chunks:
                    st.warning(answer)
                else:
                    st.markdown(answer)

                    # ── Source Citations ──
                    with st.expander("📚 Sources"):
                        for i, c in enumerate(chunks, 1):
                            score = round(float(c.get("score", 0)), 4)
                            page  = c.get("page", "N/A")
                            src   = c.get("source", "unknown")
                            st.markdown(f"**{i}. {src} | Page {page} | Score: {score}**")

                    # ── Debug Mode ──
                    if debug_mode:
                        with st.expander("🧪 Debug: Retrieved Chunks"):
                            for i, c in enumerate(chunks, 1):
                                st.markdown(f"### Chunk {i}")
                                st.markdown(f"**Source:** {c.get('source', 'N/A')}")
                                st.markdown(f"**Page:** {c.get('page', 'N/A')}")
                                st.markdown(f"**Score:** {c.get('score', 'N/A')}")
                                st.text(c.get("text", "")[:1000])

            except Exception as e:
                answer = f"❌ Unexpected error: {e}"
                st.error(answer)

    if answer is not None:
        st.session_state.messages.append({"role": "assistant", "content": answer})