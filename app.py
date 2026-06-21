import os
import glob
import re
import gradio as gr
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq

# ── CONFIG ────────────────────────────────────────────────────────────────
GROQ_API_KEY = "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"
CORPUS_DIR = "./corpus"
COLLECTION_NAME = "tamil_history"
TOP_K = 4

groq_client = Groq(api_key=GROQ_API_KEY)

# ── CHUNKING ──────────────────────────────────────────────────────────────
def load_and_chunk_corpus(corpus_dir):
    chunks = []
    files = sorted(glob.glob(os.path.join(corpus_dir, "*.md")))
    for filepath in files:
        filename = os.path.basename(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        # Split by H2 headers — each Q&A section is a chunk
        sections = re.split(r'\n## ', content)
        for i, section in enumerate(sections):
            section = section.strip()
            if not section:
                continue
            if i == 0 and section.startswith("# "):
                # File title — skip or treat as intro
                lines = section.split("\n")
                title = lines[0].replace("# ", "").strip()
                body = "\n".join(lines[1:]).strip()
                if body:
                    chunks.append({
                        "id": f"{filename}_intro",
                        "text": f"# {title}\n\n{body}",
                        "source": filename,
                        "header": title
                    })
            else:
                # Regular H2 section
                lines = section.split("\n")
                header = lines[0].strip()
                body = "\n".join(lines[1:]).strip()
                if len(body) > 100:
                    chunks.append({
                        "id": f"{filename}_{i}",
                        "text": f"## {header}\n\n{body}",
                        "source": filename,
                        "header": header
                    })
    return chunks

# ── VECTOR DB ─────────────────────────────────────────────────────────────
def build_index(chunks):
    client = chromadb.Client()
    ef = embedding_functions.DefaultEmbeddingFunction()
    # Delete if exists
    try:
        client.delete_collection(COLLECTION_NAME)
    except:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef
    )
    # Add in batches
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        collection.add(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[{"source": c["source"], "header": c["header"]} for c in batch]
        )
    return collection

# ── RETRIEVAL ─────────────────────────────────────────────────────────────
def retrieve(query, collection, top_k=TOP_K):
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    return docs, metas

# ── GENERATION ────────────────────────────────────────────────────────────
def generate(query, docs, metas):
    context = "\n\n---\n\n".join([
        f"Source: {m['header']}\n\n{d}"
        for d, m in zip(docs, metas)
    ])
    system_prompt = """You are a scholarly expert on Tamil and Tamil Nadu history with deep knowledge spanning prehistoric times to the modern era. 

Answer questions based on the provided context. Be thorough, accurate, and scholarly in your responses. 
- Cite specific names, dates, and events when available in the context
- If the context doesn't fully answer the question, say so clearly
- Use a scholarly but accessible tone
- Structure longer answers with clear paragraphs"""

    user_prompt = f"""Context from Tamil History knowledge base:

{context}

---

Question: {query}

Please provide a comprehensive, scholarly answer based on the context above."""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=1024,
        temperature=0.3
    )
    return response.choices[0].message.content

# ── INITIALISE ────────────────────────────────────────────────────────────
print("Loading corpus...")
chunks = load_and_chunk_corpus(CORPUS_DIR)
print(f"Loaded {len(chunks)} chunks from {CORPUS_DIR}")

print("Building vector index...")
collection = build_index(chunks)
print(f"Index built with {collection.count()} vectors")

# ── GRADIO UI ─────────────────────────────────────────────────────────────
EXAMPLES = [
    "What is the significance of the Keezhadi excavation?",
    "Who were the Sangam Age poets and what did they write about?",
    "Describe the naval expeditions of Rajendra Chola I",
    "What was the Dravidian movement and who founded it?",
    "How old is the Tamil language and what makes it unique?",
    "What were the Pallava dynasty's contributions to temple architecture?",
    "Who was Periyar and what was the Self-Respect Movement?",
    "Describe the trade connections between ancient Tamil Nadu and Rome",
    "What happened during the Anti-Hindi agitations in Tamil Nadu?",
    "Who were the Pandyas and what were the pearl fisheries?",
]

def answer(query, history):
    if not query.strip():
        return "", history
    docs, metas = retrieve(query, collection)
    sources = list({m['header'] for m in metas})
    response = generate(query, docs, metas)
    source_note = f"\n\n---\n📚 **Sources retrieved:** {' · '.join(sources[:3])}"
    full_response = response + source_note
    history.append((query, full_response))
    return "", history

CSS = """
.gradio-container { max-width: 900px !important; margin: 0 auto; }
footer { display: none !important; }
"""

with gr.Blocks(css=CSS, title="Tamil History RAG") as demo:
    gr.HTML("""
    <div style="text-align:center;padding:1.5rem 0 1rem">
      <h1 style="font-size:1.75rem;font-weight:700;margin-bottom:0.35rem">
        🏛️ Tamil History — RAG Knowledge Engine
      </h1>
      <p style="color:#666;font-size:0.9rem;margin:0">
        Scholarly Q&A over 5000+ years of Tamil and Tamil Nadu history · 
        Built by <a href="https://github.com/AzariahOnyx" target="_blank">Azariah Onyx</a>
      </p>
    </div>
    """)

    chatbot = gr.Chatbot(
        label="",
        height=480,
        show_label=False,
        bubble_full_width=False,
    )

    with gr.Row():
        query_input = gr.Textbox(
            placeholder="Ask anything about Tamil history — from prehistoric times to modern Tamil Nadu...",
            show_label=False,
            scale=5,
            container=False
        )
        submit_btn = gr.Button("Ask", variant="primary", scale=1)

    gr.Examples(
        examples=EXAMPLES,
        inputs=query_input,
        label="Example questions"
    )

    gr.HTML("""
    <div style="text-align:center;padding:1rem 0 0;color:#888;font-size:0.8rem">
      Corpus: 10 scholarly chapters · Prehistoric → Sangam → Pallava → Chola → Pandya → Nayak → Colonial → Dravidian → Modern
    </div>
    """)

    submit_btn.click(answer, [query_input, chatbot], [query_input, chatbot])
    query_input.submit(answer, [query_input, chatbot], [query_input, chatbot])

demo.launch(server_name="0.0.0.0", server_port=7860)
