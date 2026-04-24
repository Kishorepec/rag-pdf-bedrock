import boto3
import json
import io
import os
import faiss
import pickle
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

def build_index(chunks):
    embeddings = []
    bar = st.progress(0, text="Embedding chunks...")
    for i, chunk in enumerate(chunks):
        embeddings.append(embed_text(chunk.page_content))
        bar.progress((i + 1) / len(chunks), text=f"Embedding chunk {i+1}/{len(chunks)}...")
    bar.empty()
    matrix = np.array(embeddings, dtype="float32")
    idx = faiss.IndexFlatL2(matrix.shape[1])
    idx.add(matrix)
    return idx

def retrieve(question, index, chunks, top_k=4):
    q_vec = embed_text(question).reshape(1, -1)
    distances, indices = index.search(q_vec, top_k)
    return [chunks[i] for i in indices[0] if i != -1]

def generate_answer(question, context_chunks):
    context = "\n\n".join([
        f"[Page {c.metadata['page']}]\n{c.page_content}"
        for c in context_chunks
    ])

    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant. Answer the question using ONLY the context provided.
If the answer is not in the context, say "I don't have enough information."
<|eot_id|><|start_header_id|>user<|end_header_id|>
Context:
{context}

Question: {question}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    response = bedrock.invoke_model(
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
    result = json.loads(response["body"].read())
    return result["generation"].strip()

# ── Streamlit UI ──────────────────────────────────────────────
st.set_page_config(page_title="PDF Q&A — AWS RAG", page_icon="📄")
st.title("📄 PDF Q&A with AWS Bedrock")
st.caption("Powered by S3 · Titan Embeddings · Llama 3 · FAISS")

if "index"   not in st.session_state: st.session_state.index   = None
if "chunks"  not in st.session_state: st.session_state.chunks  = None
if "history" not in st.session_state: st.session_state.history = []

with st.sidebar:
    st.header("📂 Upload PDF")
    uploaded = st.file_uploader("Choose a PDF", type="pdf")

    if uploaded and st.button("🔄 Process PDF"):
        # Read all bytes once — reuse for both S3 and PDF parsing
        pdf_bytes = uploaded.read()

        with st.spinner("Uploading to S3..."):
            s3_key = f"uploads/{uploaded.name}"
            s3_client.upload_fileobj(io.BytesIO(pdf_bytes), BUCKET, s3_key)

        with st.spinner("Extracting text..."):
            reader = PdfReader(io.BytesIO(pdf_bytes))
            pages = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    pages.append(Document(
                        page_content=text,
                        metadata={"page": i+1, "source": uploaded.name}
                    ))

        with st.spinner("Chunking..."):
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_documents(pages)

        st.session_state.index  = build_index(chunks)
        st.session_state.chunks = chunks
        st.session_state.history = []

        st.success(f"✅ Ready! {len(chunks)} chunks indexed.")
        st.info(f"📁 Saved to S3: {s3_key}")

if st.session_state.index:
    st.divider()
    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg:
                st.caption(f"📄 Sources: {msg['sources']}")

    question = st.chat_input("Ask anything about your PDF...")
    if question:
        st.session_state.history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                hits    = retrieve(question, st.session_state.index, st.session_state.chunks)
                answer  = generate_answer(question, hits)
                sources = ", ".join(sorted(set([f"Page {c.metadata['page']}" for c in hits])))
            st.write(answer)
            st.caption(f"📄 Sources: {sources}")
            st.session_state.history.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })

    with st.expander("🔍 Debug — retrieved chunks"):
        if st.session_state.history:
            last_q = next((m["content"] for m in reversed(st.session_state.history) if m["role"] == "user"), None)
            if last_q:
                hits = retrieve(last_q, st.session_state.index, st.session_state.chunks)
                for i, c in enumerate(hits):
                    st.markdown(f"**Chunk {i+1} — Page {c.metadata['page']}**")
                    st.text(c.page_content[:300])
else:
    st.info("👈 Upload a PDF from the sidebar to get started.")
