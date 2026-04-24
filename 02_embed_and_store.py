import boto3
import io
import os
import json
import faiss
import pickle
import numpy as np
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

BUCKET = os.getenv("S3_BUCKET_NAME")
S3_KEY = "Kishore_new_resume].pdf"   # <- your filename in S3

# ── 1. Reload chunks ─────────────────────────────────────────
print("📥 Downloading PDF from S3...")
s3 = boto3.client("s3")
obj = s3.get_object(Bucket=BUCKET, Key=S3_KEY)
pdf_bytes = io.BytesIO(obj["Body"].read())

reader = PdfReader(pdf_bytes)
pages = []
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if text and text.strip():
        pages.append(Document(page_content=text, metadata={"page": i+1, "source": S3_KEY}))

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(pages)
print(f"✅ {len(chunks)} chunks ready for embedding")

# ── 2. Embed using Bedrock Titan ─────────────────────────────
bedrock = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_DEFAULT_REGION"))

def embed_text(text: str) -> list[float]:
    body = json.dumps({"inputText": text})
    response = bedrock.invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        body=body,
        contentType="application/json",
        accept="application/json"
    )
    result = json.loads(response["body"].read())
    return result["embedding"]

print("\n🔄 Embedding chunks via Bedrock Titan (this takes ~1-2 min)...")
embeddings = []
for i, chunk in enumerate(chunks):
    vec = embed_text(chunk.page_content)
    embeddings.append(vec)
    if (i + 1) % 10 == 0:
        print(f"   Embedded {i+1}/{len(chunks)} chunks...")

print(f"✅ All {len(chunks)} chunks embedded")
print(f"   Embedding dimension: {len(embeddings[0])}")

# ── 3. Store in FAISS ────────────────────────────────────────
print("\n💾 Building FAISS index...")
dim = len(embeddings[0])
embedding_matrix = np.array(embeddings, dtype="float32")

index = faiss.IndexFlatL2(dim)
index.add(embedding_matrix)

faiss.write_index(index, "faiss_index.bin")
with open("chunks_metadata.pkl", "wb") as f:
    pickle.dump(chunks, f)

print(f"✅ FAISS index saved — {index.ntotal} vectors indexed")
print("✅ Chunk metadata saved to chunks_metadata.pkl")
print("\n🎉 Day 2 complete! Run 03_query.py next.")
