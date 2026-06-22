import os
import glob
import re
import math
import streamlit as st
from groq import Groq

try:
    import streamlit as _st
    GROQ_API_KEY = _st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
except Exception:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
CORPUS_DIR = "./corpus"
TOP_K = 4

groq_client = Groq(api_key=GROQ_API_KEY)

STOP = {"the","a","an","and","or","in","of","to","is","was","for","with","on","at","by","from",
        "that","this","it","as","are","were","be","been","has","have","had","but","not","they",
        "their","which","who","what","how","when","where","its","he","she","we","you","i","my",
        "our","his","her"}

def tokenize(t):
    return [w.lower() for w in re.findall(r'\b[a-z]{3,}\b', t.lower()) if w.lower() not in STOP]

@st.cache_resource
def load_index():
    chunks = []
    files = sorted(glob.glob(os.path.join(CORPUS_DIR, "*.md")))
    for filepath in files:
        filename = os.path.basename(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        sections = re.split(r'\n## ', content)
        for i, section in enumerate(sections):
            section = section.strip()
            if not section:
                continue
            if i == 0 and section.startswith("# "):
                lines = section.split("\n")
                title = lines[0].replace("# ", "").strip()
                body = "\n".join(lines[1:]).strip()
                if body:
                    chunks.append({"text": f"# {title}\n\n{body}", "source": filename, "header": title})
            else:
                lines = section.split("\n")
                header = lines[0].strip()
                body = "\n".join(lines[1:]).strip()
                if len(body) > 100:
                    chunks.append({"text": f"## {header}\n\n{body}", "source": filename, "header": header})

    # Build TF-IDF vectors
    vocab = {}
    for c in chunks:
        for word in tokenize(c["text"]):
            if word not in vocab:
                vocab[word] = len(vocab)

    vectors = []
    for c in chunks:
        vec = [0.0] * len(vocab)
        for word in tokenize(c["text"]):
            if word in vocab:
                vec[vocab[word]] += 1.0
        norm = math.sqrt(sum(v*v for v in vec)) or 1.0
        vectors.append([v/norm for v in vec])

    return chunks, vectors, vocab

def retrieve(query, chunks, vectors, vocab):
    words = tokenize(query)
    q_vec = [0.0] * len(vocab)
    for word in words:
        if word in vocab:
            q_vec[vocab[word]] += 1.0
    norm = math.sqrt(sum(v*v for v in q_vec)) or 1.0
    q_vec = [v/norm for v in q_vec]
    scores = [(sum(x*y for x,y in zip(q_vec, v)), i) for i, v in enumerate(vectors)]
    scores.sort(reverse=True)
    return [chunks[i] for _, i in scores[:TOP_K]]

def generate(query, chunks):
    context = "\n\n---\n\n".join([f"Source: {c['header']}\n\n{c['text']}" for c in chunks])
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a scholarly expert on Tamil and Tamil Nadu history. Answer based on provided context. Be thorough and cite specific names, dates, and events."},
            {"role": "user", "content": f"Context:\n{context}\n\n---\n\nQuestion: {query}\n\nProvide a comprehensive scholarly answer."}
        ],
        max_tokens=1024, temperature=0.3
    )
    return response.choices[0].message.content

# --- UI ---
st.set_page_config(page_title="Tamil History RAG", page_icon="🏛️", layout="centered")
st.title("🏛️ Tamil History — RAG Knowledge Engine")
st.caption("Scholarly Q&A over 5000+ years of Tamil and Tamil Nadu history · Built by [Azariah Onyx](https://github.com/AzariahOnyx)")

with st.spinner("Loading corpus and building index..."):
    chunks, vectors, vocab = load_index()

st.success(f"Index ready — {len(chunks)} chunks from 10 scholarly chapters")

if "history" not in st.session_state:
    st.session_state.history = []

# Display chat history
for q, a in st.session_state.history:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.write(a)

# Examples
with st.expander("Example questions"):
    examples = [
        "What is the significance of the Keezhadi excavation?",
        "Who were the Sangam Age poets and what did they write about?",
        "Describe the naval expeditions of Rajendra Chola I",
        "What was the Dravidian movement and who founded it?",
        "Who was Periyar and what was the Self-Respect Movement?",
    ]
    for ex in examples:
        if st.button(ex, key=ex):
            st.session_state.pending_query = ex
            st.rerun()

query = st.chat_input("Ask anything about Tamil history...")
if not query and "pending_query" in st.session_state:
    query = st.session_state.pop("pending_query")

if query:
    with st.chat_message("user"):
        st.write(query)
    with st.chat_message("assistant"):
        with st.spinner("Searching corpus and generating answer..."):
            top_chunks = retrieve(query, chunks, vectors, vocab)
            sources = list({c["header"] for c in top_chunks})
            answer = generate(query, top_chunks)
            src_note = " · ".join(sources[:3])
            full = answer + f"\n\n---\n📚 **Sources:** {src_note}"
            st.write(full)
    st.session_state.history.append((query, full))
