import streamlit as st
import re
from rag_pipeline import RAGPipeline

# ─────────────────────────────
# PAGE CONFIG
# ─────────────────────────────
st.set_page_config(
    page_title="IT Policy Assistant",
    page_icon="🛡️",
    layout="centered",
)

# ─────────────────────────────
# SECURITY — sanitize LLM output
# ─────────────────────────────
_CREDENTIAL_VALUE = re.compile(
    r"""
    (?:
        (?:password|passwd|pwd|secret|api[_-]?key|token|auth|credential
           |private[_-]?key|access[_-]?key|client[_-]?secret|bearer
           |passphrase|pin|ssn|cvv)
        \s*[=:]\s*\S+
    )
    |(?:sk-|gsk_|pcsk_|Bearer\s|ghp_|xox[baprs]-)[A-Za-z0-9_\-]{8,}
    |(?:Bearer|Basic)\s+\S+
    """,
    re.IGNORECASE | re.VERBOSE,
)

_INLINE_CREDENTIAL = re.compile(
    r"""
    (?:
        (?:default\s+)?(?:password|passwd|pwd)\s*(?:is|:|\=)\s*\S+
        |(?:username|user)\s*(?:is|:|\=)\s*\S+\s+(?:and\s+)?(?:password|passwd)\s*(?:is|:|\=)\s*\S+
        |api\s*key\s*(?:is|:|\=)\s*\S+
        |token\s*(?:is|:|\=)\s*\S+
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

def sanitize_content(text: str) -> str:
    text = _CREDENTIAL_VALUE.sub("[REDACTED]", text)
    text = _INLINE_CREDENTIAL.sub("[REDACTED]", text)
    return text


# ─────────────────────────────
# LOAD PIPELINE (CACHED)
# ─────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline():
    return RAGPipeline()

rag = load_pipeline()

# ─────────────────────────────
# SESSION STATE
# ─────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ─────────────────────────────
# SIDEBAR — minimal
# ─────────────────────────────
with st.sidebar:
    st.markdown("### 🛡️ IT Policy Assistant")
    st.markdown("Ask questions about your organization's IT policies.")
    st.divider()

    # DB stats — read only, no config
    try:
        count   = rag.store.verify()
        sources = rag.store.list_sources()
        st.metric("Indexed Documents", len(sources))
        st.metric("Total Vectors", count)
        if sources:
            st.markdown("**Documents:**")
            for s in sources:
                st.markdown(f"- 📄 {s}")
    except Exception:
        pass

    st.divider()
    if st.button("🔄 Reset Chat"):
        st.session_state.messages = []
        st.rerun()


# ─────────────────────────────
# HEADER
# ─────────────────────────────
st.markdown("## 🛡️ IT Policy Assistant")
st.markdown("Ask me anything about your organization's IT policies.")
st.divider()

# ─────────────────────────────
# DISPLAY CHAT HISTORY
# ─────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ─────────────────────────────
# CHAT INPUT
# ─────────────────────────────
prompt = st.chat_input("Ask about IT policies...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    answer = None

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]
                ]

                answer, chunks = rag.query_with_history(prompt, history=history)
                answer = sanitize_content(answer)

                if not chunks:
                    st.warning(answer)
                else:
                    st.markdown(answer)
                    with st.expander("📚 Sources"):
                        for i, c in enumerate(chunks, 1):
                            score = round(float(c.get("score", 0)), 4)
                            page  = c.get("page", "N/A")
                            src   = c.get("source", "unknown")
                            st.markdown(f"**{i}. {src} | Page {page} | Score: {score}**")

            except Exception as e:
                answer = "❌ Something went wrong. Please try again."
                st.error(answer)

    if answer is not None:
        st.session_state.messages.append({"role": "assistant", "content": answer})
