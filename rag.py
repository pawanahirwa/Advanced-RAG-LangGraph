r"""
Advanced Local RAG — LangGraph + Qdrant + BGE Embeddings + BGE Reranker + Ollama
--------------------------------------------------------------------------------
Pipeline (as a stateful graph):
    retrieve  ->  rerank  ->  grade  ->  generate
                                   \-> (if weak)  "not found"

Metadata filtering: each chunk stores {page, source}; queries can filter by page.
Optimized for low-RAM machines. Swap model names below to scale up.
"""

import gradio as gr
import ollama
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition, Range,
)
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END

# ── CONFIG (swap these to scale up on better hardware) ──────────────────────
EMBED_MODEL   = "BAAI/bge-small-en-v1.5"     # 384-dim, ~130 MB
RERANK_MODEL  = "BAAI/bge-reranker-base"     # ~1.1 GB  (heaviest piece)
LLM_MODEL     = "qwen2.5:0.5b"               # ~400 MB; try qwen2.5:1.5b if RAM allows
COLLECTION    = "pdf_docs"
EMBED_DIM     = 384
FETCH_K       = 10    # how many to retrieve before reranking
TOP_K         = 3     # how many to keep after reranking
SCORE_FLOOR   = 0.0   # min rerank score to consider a chunk "relevant"
# BGE query instruction improves retrieval for short queries
QUERY_PREFIX  = "Represent this sentence for searching relevant passages: "

# ── Load models once ────────────────────────────────────────────────────────
print("Loading embedding model…")
embedder = SentenceTransformer(EMBED_MODEL, device="cpu")
print("Loading reranker…")
reranker = CrossEncoder(RERANK_MODEL, device="cpu")
print("Connecting to Qdrant (embedded)…")
qdrant = QdrantClient(path="./qdrant_data")   # local, persists to disk, no Docker


# ── Indexing ─────────────────────────────────────────────────────────────────
def index_pdf(pdf_path: str) -> int:
    """Read a PDF, chunk it, embed, and upsert into Qdrant with metadata."""
    reader = PdfReader(pdf_path)
    source = pdf_path.split("/")[-1].split("\\")[-1]

    # recreate collection fresh for each upload
    qdrant.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )

    chunks, metas = [], []
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        # simple chunking: ~700 chars with overlap
        size, overlap = 700, 100
        start = 0
        while start < len(text):
            piece = text[start:start + size].strip()
            if len(piece) > 40:
                chunks.append(piece)
                metas.append({"page": page_num + 1, "source": source})
            start += size - overlap

    if not chunks:
        return 0

    # batch embed to keep memory low
    vectors = embedder.encode(chunks, batch_size=8, show_progress_bar=False,
                              normalize_embeddings=True)

    points = [
        PointStruct(id=i, vector=vectors[i].tolist(),
                    payload={"text": chunks[i], **metas[i]})
        for i in range(len(chunks))
    ]
    qdrant.upsert(collection_name=COLLECTION, points=points)
    return len(chunks)


# ── LangGraph state ──────────────────────────────────────────────────────────
class RAGState(TypedDict):
    question: str
    page_filter: Optional[int]      # metadata filter (None = no filter)
    retrieved: List[dict]
    reranked: List[dict]
    answer: str


# ── Nodes ────────────────────────────────────────────────────────────────────
def retrieve_node(state: RAGState) -> RAGState:
    q_vec = embedder.encode([QUERY_PREFIX + state["question"]],
                            normalize_embeddings=True)[0].tolist()

    # metadata filtering: optionally restrict to one page
    q_filter = None
    if state.get("page_filter"):
        q_filter = Filter(must=[FieldCondition(
            key="page",
            range=Range(gte=state["page_filter"], lte=state["page_filter"]),
        )])

    response = qdrant.query_points(
        collection_name=COLLECTION,
        query=q_vec,
        query_filter=q_filter,
        limit=FETCH_K,
    )
    hits = response.points
    state["retrieved"] = [
        {"text": h.payload["text"], "page": h.payload["page"], "score": h.score}
        for h in hits
    ]
    return state


def rerank_node(state: RAGState) -> RAGState:
    docs = state["retrieved"]
    if not docs:
        state["reranked"] = []
        return state

    pairs = [(state["question"], d["text"]) for d in docs]
    scores = reranker.predict(pairs)          # cross-encoder relevance scores
    for d, s in zip(docs, scores):
        d["rerank_score"] = float(s)

    ranked = sorted(docs, key=lambda d: d["rerank_score"], reverse=True)
    state["reranked"] = ranked[:TOP_K]
    return state


def grade_edge(state: RAGState) -> str:
    """Conditional edge: if best chunk is too weak, skip generation."""
    top = state["reranked"]
    if not top or top[0]["rerank_score"] < SCORE_FLOOR:
        return "weak"
    return "ok"


def not_found_node(state: RAGState) -> RAGState:
    state["answer"] = "I couldn't find this information in the document."
    return state


def generate_node(state: RAGState) -> RAGState:
    context = "\n\n".join(
        f"[Page {d['page']}] {d['text']}" for d in state["reranked"]
    )
    prompt = (
        "Use the context to answer the question. Be concise. "
        "If the answer is not in the context, say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {state['question']}\nAnswer:"
    )
    resp = ollama.generate(model=LLM_MODEL, prompt=prompt,
                           options={"num_ctx": 2048, "temperature": 0.2})
    state["answer"] = resp["response"].strip()
    return state


# ── Build the graph ──────────────────────────────────────────────────────────
def build_graph():
    g = StateGraph(RAGState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("rerank", rerank_node)
    g.add_node("generate", generate_node)
    g.add_node("not_found", not_found_node)

    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "rerank")
    g.add_conditional_edges("rerank", grade_edge,
                            {"ok": "generate", "weak": "not_found"})
    g.add_edge("generate", END)
    g.add_edge("not_found", END)
    return g.compile()

rag_graph = build_graph()


# ── Gradio UI ────────────────────────────────────────────────────────────────
indexed = {"ready": False}

def on_upload(pdf):
    if pdf is None:
        return "⚠️ No file selected.", gr.update(interactive=False)
    try:
        n = index_pdf(pdf.name)
        if n == 0:
            return "❌ No extractable text found in PDF.", gr.update(interactive=False)
        indexed["ready"] = True
        return f"✅ Ready! Indexed {n} chunks.", gr.update(interactive=True)
    except Exception as e:
        return f"❌ Error: {e}", gr.update(interactive=False)


def on_ask(question, page, history):
    if not question.strip():
        return history, ""
    if not indexed["ready"]:
        history = history + [{"role": "user", "content": question},
                             {"role": "assistant", "content": "⚠️ Upload a PDF first."}]
        return history, ""
    try:
        page_filter = int(page) if str(page).strip().isdigit() else None
        result = rag_graph.invoke({
            "question": question, "page_filter": page_filter,
            "retrieved": [], "reranked": [], "answer": "",
        })
        # show which pages were used
        pages = sorted({d["page"] for d in result["reranked"]})
        tag = f"\n\n*(sources: page {', '.join(map(str, pages))})*" if pages else ""
        history = history + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": result["answer"] + tag},
        ]
        return history, ""
    except Exception as e:
        history = history + [{"role": "user", "content": question},
                             {"role": "assistant", "content": f"❌ Error: {e}"}]
        return history, ""


with gr.Blocks(title="LangGraph RAG") as demo:
    gr.Markdown("## 🧠 Advanced RAG — LangGraph + Qdrant + BGE Reranker")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📂 Document")
            pdf_in = gr.File(label="Upload PDF", file_types=[".pdf"])
            up_btn = gr.Button("⚡ Process PDF", variant="primary")
            status = gr.Textbox(label="Status", value="No document loaded.",
                                interactive=False)
            page_in = gr.Textbox(label="Filter by page (optional)",
                                 placeholder="e.g. 3")

        with gr.Column(scale=2):
            gr.Markdown("### 💬 Chat")
            chat = gr.Chatbot(height=440, show_label=False)
            with gr.Row():
                q_in = gr.Textbox(placeholder="Ask about your document…",
                                  show_label=False, scale=5, interactive=False)
                send = gr.Button("Send", variant="primary", scale=1)

    up_btn.click(on_upload, inputs=pdf_in, outputs=[status, q_in])
    send.click(on_ask, inputs=[q_in, page_in, chat], outputs=[chat, q_in])
    q_in.submit(on_ask, inputs=[q_in, page_in, chat], outputs=[chat, q_in])

demo.launch()