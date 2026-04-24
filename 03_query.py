import boto3
import json
import os
import faiss
import pickle
import numpy as np
from dotenv import load_dotenv

load_dotenv()

bedrock = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_DEFAULT_REGION"))

# ── 1. Load FAISS index + chunks ─────────────────────────────
print("📂 Loading FAISS index and chunk metadata...")
index  = faiss.read_index("faiss_index.bin")
with open("chunks_metadata.pkl", "rb") as f:
    chunks = pickle.load(f)
print(f"✅ Loaded {index.ntotal} vectors, {len(chunks)} chunks")

# ── 2. Embed the user question ────────────────────────────────
def embed_text(text: str) -> np.ndarray:
    body = json.dumps({"inputText": text})
    response = bedrock.invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        body=body,
        contentType="application/json",
        accept="application/json"
    )
    result = json.loads(response["body"].read())
    return np.array(result["embedding"], dtype="float32").reshape(1, -1)

# ── 3. Retrieve top-k relevant chunks ────────────────────────
def retrieve(question: str, top_k: int = 4):
    q_vec = embed_text(question)
    distances, indices = index.search(q_vec, top_k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx != -1:
            results.append({
                "chunk": chunks[idx],
                "score": float(dist)
            })
    return results

# ── 4. Generate answer using Meta Llama 3 on Bedrock ─────────
def answer(question: str):
    hits = retrieve(question)

    context = "\n\n".join([
        f"[Page {h['chunk'].metadata['page']}]\n{h['chunk'].page_content}"
        for h in hits
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
    answer_text = result["generation"].strip()
    sources = list(set([f"Page {h['chunk'].metadata['page']}" for h in hits]))
    return answer_text, sources

# ── 5. Test it ────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🤖 RAG Query Test — Llama 3 on Bedrock")
    print("=" * 50)
    question = input("Enter your question: ")
    print("\n🔍 Retrieving relevant chunks...")
    ans, sources = answer(question)
    print(f"\n💬 Answer:\n{ans}")
    print(f"\n📄 Sources: {', '.join(sources)}")
