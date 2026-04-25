import boto3
import json
import io
import os
import faiss
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

bedrock   = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_DEFAULT_REGION"))
s3_client = boto3.client("s3")
BUCKET    = os.getenv("S3_BUCKET_NAME")

# ── Helpers ───────────────────────────────────────────────────
def embed_text(text: str) -> np.ndarray:
    body = json.dumps({"inputText": text})
    response = bedrock.invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        body=body,
        contentType="application/json",
        accept="application/json"
    )
    result = json.loads(response["body"].read())
    return np.array(result["embedding"], dtype="float32")

def process_pdf(pdf_bytes: bytes, filename: str) -> list[Document]:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages  = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages.append(Document(
                page_content=text,
                metadata={"page": i + 1, "source": filename}
            ))
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(pages)

def build_index(all_chunks: list[Document], label: str = "") -> faiss.IndexFlatL2:
    embeddings = []
    bar = st.progress(0, text=f"Embedding {label}...")
    for i, chunk in enumerate(all_chunks):
        embeddings.append(embed_text(chunk.page_content))
        bar.progress((i + 1) / len(all_chunks),
                     text=f"Embedding chunk {i+1}/{len(all_chunks)}...")
    bar.empty()
    matrix = np.array(embeddings, dtype="float32")
    idx = faiss.IndexFlatL2(matrix.shape[1])
    idx.add(matrix)
    return idx

def retrieve(question: str, index, chunks: list[Document], top_k: int = 4):
    q_vec = embed_text(question).reshape(1, -1)
    _, indices = index.search(q_vec, top_k)
    return [chunks[i] for i in indices[0] if i != -1]

def build_prompt(question: str,
                 context_chunks: list[Document],
                 all_sources: list[str],
                 history: list[dict]) -> str:
    """Build a Llama 3 prompt with RAG context + conversation memory."""

    context = "\n\n".join([
        f"[Document: {c.metadata['source']} | Page {c.metadata['page']}]\n{c.page_content}"
        for c in context_chunks
    ])
    sources_list = "\n".join([f"- {s}" for s in all_sources])

    # System block
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant analyzing documents. The user has loaded {len(all_sources)} document(s):
{sources_list}

Answer questions using ONLY the retrieved context provided.
When comparing documents, refer to them by their filename.
You also have access to the recent conversation history — use it to understand follow-up questions.
If the answer is not in the context, say "I don't have enough information."
<|eot_id|>"""

    # Inject last 6 exchanges as memory (3 user + 3 assistant turns)
    recent = history[-6:] if len(history) > 6 else history
    for msg in recent:
        role = "user" if msg["role"] == "user" else "assistant"
        prompt += f"<|start_header_id|>{role}<|end_header_id|>\n{msg['content']}<|eot_id|>"

    # Current turn with RAG context
    prompt += f"""<|start_header_id|>user<|end_header_id|>
Context:
{context}

Question: {question}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    return prompt

def stream_answer(prompt: str):
    """Stream tokens from Llama 3 on Bedrock using invoke_model_with_response_stream."""
    response = bedrock.invoke_model_with_response_stream(
        modelId="meta.llama3-8b-instruct-v1:0",
        body=json.dumps({
            "prompt": prompt,
            "max_gen_len": 512,
            "temperature": 0.2,
            "top_p": 0.9
        }),
        contentType="application/json",
        accept="application/json"
    )
    for event in response["body"]:
        chunk = json.loads(event["chunk"]["bytes"])
        token = chunk.get("generation", "")
        if token:
            yield token

# ── Streamlit UI ──────────────────────────────────────────────
st.set_page_config(page_title="PDF Q&A — AWS RAG", page_icon="📄")
st.title("📄 PDF Q&A with AWS Bedrock")
st.caption("Powered by S3 · Titan Embeddings V2 · Llama 3 · FAISS · Streaming + Memory")

for key, default in {
    "index": None, "chunks": [], "history": [], "loaded_files": []
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Upload PDFs")
    st.caption("Upload multiple PDFs and query across all of them.")

    uploaded_files = st.file_uploader("Choose PDF(s)", type="pdf", accept_multiple_files=True)

    if uploaded_files and st.button("🔄 Process PDFs"):
        all_chunks = []
        progress   = st.progress(0, text="Starting...")

        for file_idx, uploaded in enumerate(uploaded_files):
            filename = uploaded.name
            progress.progress(
                file_idx / len(uploaded_files),
                text=f"Processing {filename}..."
            )
            pdf_bytes = uploaded.read()
            s3_client.upload_fileobj(io.BytesIO(pdf_bytes), BUCKET, f"uploads/{filename}")
            all_chunks.extend(process_pdf(pdf_bytes, filename))

        progress.empty()

        st.session_state.index        = build_index(all_chunks, label=f"{len(uploaded_files)} PDF(s)")
        st.session_state.chunks       = all_chunks
        st.session_state.history      = []
        st.session_state.loaded_files = sorted(set(c.metadata["source"] for c in all_chunks))

        st.success(f"✅ {len(all_chunks)} chunks indexed from {len(st.session_state.loaded_files)} PDF(s)")

    if st.session_state.loaded_files:
        st.divider()
        st.markdown(f"**📚 Loaded files ({len(st.session_state.loaded_files)}):**")
        for fname in st.session_state.loaded_files:
            chunk_count = sum(1 for c in st.session_state.chunks if c.metadata["source"] == fname)
            st.markdown(f"- {fname} `{chunk_count} chunks`")

    if st.session_state.index is not None:
        st.divider()
        # Memory indicator
        turn_count = len([m for m in st.session_state.history if m["role"] == "user"])
        st.caption(f"🧠 Memory: {min(turn_count, 3)} of last 3 turns active")

        if st.button("🗑️ Clear & Start Over"):
            for k in ["index", "chunks", "history", "loaded_files"]:
                st.session_state[k] = None if k == "index" else []
            st.rerun()

# ── Main chat ─────────────────────────────────────────────────
if st.session_state.index is not None:
    st.divider()

    # Render existing history
    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg:
                st.caption(f"📄 {msg['sources']}")

    question = st.chat_input("Ask anything about your PDF(s)...")
    if question:
        # Add user message
        st.session_state.history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            # Retrieve relevant chunks
            hits = retrieve(question, st.session_state.index, st.session_state.chunks)

            # Build prompt with memory
            prompt = build_prompt(
                question,
                hits,
                st.session_state.loaded_files,
                # Pass history excluding the current user turn
                st.session_state.history[:-1]
            )

            # Stream response token by token
            full_answer = st.write_stream(stream_answer(prompt))

            sources = ", ".join(sorted(set(
                f"{c.metadata['source']} p.{c.metadata['page']}" for c in hits
            )))
            st.caption(f"📄 Sources: {sources}")

            st.session_state.history.append({
                "role": "assistant",
                "content": full_answer,
                "sources": f"Sources: {sources}"
            })

    with st.expander("🔍 Debug — retrieved chunks"):
        if st.session_state.history:
            last_q = next(
                (m["content"] for m in reversed(st.session_state.history) if m["role"] == "user"), None
            )
            if last_q:
                hits = retrieve(last_q, st.session_state.index, st.session_state.chunks)
                for i, c in enumerate(hits):
                    st.markdown(f"**Chunk {i+1} — {c.metadata['source']} | Page {c.metadata['page']}**")
                    st.text(c.page_content[:300])
else:
    st.info("👈 Upload one or more PDFs from the sidebar to get started.")
