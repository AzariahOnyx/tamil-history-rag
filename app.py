import os
import glob
import re
import gradio as gr
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
CORPUS_DIR = "./corpus"
COLLECTION_NAME = "tamil_history"
TOP_K = 4

groq_client = Groq(api_key=GROQ_API_KEY)


def load_and_chunk_corpus(corpus_dir):
    chunks = []
    files = sorted(glob.glob(os.path.join(corpus_dir, "*.md")))
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
                    chunks.append({
                        "id": f"{filename}_intro",
                        "text": f"# {title}\n\n{body}",
                        "source": filename,
                        "header": title
                    })
            else:
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


def build_index(chunks):
    client = chromadb.Client()
    ef = embedding_functions.DefaultEmbeddingFunction()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef
    )
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        collection.add(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[{"source": c["source"], "header": c["header"]} for c in batch]
        )
    return collection


def retrieve(query, collection, top_k=TOP_K):
    results = collection.query(query_texts=[query], n_results=top_k)
    return results["documents"][0], results["metadatas"][0]


def generate(query, docs, metas):
    context = "\n\n---\n\n".join([
        f"Source: {m['header']}\n\n{d}"
        for d, m in zip(docs, metas)
    ])
    system_prompt = (
        "You are a scholarly expert on Tamil and Tamil Nadu history. "
        "Answer questions based on the provided context. "
        "Be thorough, accurate, and scholarly. "
        "Cite specific names, dates, and events."
    )
    user_prompt = (
        f"Context:\n{context}\n\n---\n\n"
        f"Question: {query}\n\n"
        f"Provide a comprehensive scholarly answer."
    )
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


print("Loading corpus...")
chunks = load_and_chunk_corpus(CORPUS_DIR)
print(f"Loaded {len(chunks)} chunks")
print("Building index...")
collection = build_index(chunks)
print(f"Index ready with {collection.count()} vectors")

EXAMPLES = [
    "What is the significance of the Keezhadi excavation?",
    "Who were the Sangam Age poets and what did they write about?",
    "Describe the naval expeditions of Rajendra Chola I",
    "What was the Dravidian movement and who founded it?",
    "How old is the Tamil language and what makes it unique?",
    "Who was Periyar and what was the Self-Respect Movement?",
    "Describe the trade connections between ancient Tamil Nadu and Rome",
    "What happened during the Anti-Hindi agitations in Tamil Nadu?",
    "Who were the Pandyas and what were the pearl fisheries?",
]


def answer(query, history):
    if not query.strip():
        return "", history
    docs, metas = retrieve(query, collection)
    sources = list({m["header"] for m in metas})
    response = generate(query, docs, metas)
    src_note = " · ".join(sources[:3])
    full_response = response + f"\n\n---\n📚 **Sources:** {src_note}"
    history = history + [{"role": "user", "content": query}, {"role": "assistant", "content": full_response}]
    return "", history


with gr.Blocks(title="Tamil History RAG") as demo:
    gr.HTML(
        '<div style="text-align:center;padding:1.5rem 0 1rem">'
        '<h1>🏛️ Tamil History — RAG Knowledge Engine</h1>'
        '<p>Scholarly Q&A over 5000+ years of Tamil and Tamil Nadu history · '
        'Built by <a href="https://github.com/AzariahOnyx" target="_blank">Azariah Onyx</a></p>'
        '</div>'
    )
    chatbot = gr.Chatbot(label="", height=480, type="messages")
    with gr.Row():
        query_input = gr.Textbox(
            placeholder="Ask anything about Tamil history...",
            show_label=False,
            scale=5,
            container=False
        )
        submit_btn = gr.Button("Ask", variant="primary", scale=1)
    gr.Examples(examples=EXAMPLES, inputs=query_input, label="Example questions")
    gr.HTML(
        '<div style="text-align:center;padding:1rem 0 0;color:#888;font-size:0.8rem">'
        'Corpus: 10 scholarly chapters · '
        'Prehistoric → Sangam → Pallava → Chola → Pandya → Nayak → Colonial → Dravidian → Modern'
        '</div>'
    )
    submit_btn.click(answer, [query_input, chatbot], [query_input, chatbot])
    query_input.submit(answer, [query_input, chatbot], [query_input, chatbot])

demo.launch(server_name="0.0.0.0", server_port=7860)
