# 📄 PDF Q&A with AWS Bedrock

A production-style Retrieval-Augmented Generation (RAG) application that lets you upload any PDF and ask questions about it — powered entirely by AWS services.

## 🏗️ Architecture

```
PDF Upload (Streamlit UI)
        │
        ▼
Amazon S3  ──────────────────── stores original PDF
        │
        ▼
Text Extraction (pypdf)
        │
        ▼
Chunking (LangChain RecursiveCharacterTextSplitter)
        │
        ▼
Amazon Bedrock – Titan Embeddings V2  ── converts chunks to vectors
        │
        ▼
FAISS Vector Index  ──────────────────── stores & searches vectors
        │
  [At query time]
        │
User Question ──► Titan Embeddings ──► FAISS retrieves top-4 chunks
                                                │
                                                ▼
                              Amazon Bedrock – Meta Llama 3 8B
                                                │
                                                ▼
                                     Answer + Source Pages
```

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| PDF Storage | Amazon S3 |
| Embeddings | Amazon Bedrock – Titan Embeddings V2 |
| LLM | Amazon Bedrock – Meta Llama 3 8B Instruct |
| Vector Store | FAISS (local) |
| Orchestration | LangChain |
| UI | Streamlit |

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- AWS account with Bedrock access
- Bedrock model access enabled for:
  - `amazon.titan-embed-text-v2:0`
  - `meta.llama3-8b-instruct-v1:0`

### Installation

```bash
git clone https://github.com/Kishorepec/rag-pdf-bedrock.git
cd rag-pdf-bedrock
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the root directory:

```env
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=ap-south-1
S3_BUCKET_NAME=your-bucket-name
```

### Run

```bash
# Full Streamlit UI
streamlit run app.py

# Terminal-only RAG test
python 03_query.py
```

## 📁 Project Structure

```
rag-pdf-app/
├── 01_load_and_chunk.py   # Download from S3, extract text, chunk
├── 02_embed_and_store.py  # Embed via Titan, store in FAISS
├── 03_query.py            # Terminal RAG query test
├── app.py                 # Streamlit chat UI
├── .env                   # AWS credentials (not committed)
└── requirements.txt       # Dependencies
```

## 💡 How RAG Works

1. **Indexing** — PDF is split into ~500 character chunks. Each chunk is converted into a 1024-dimension vector using Titan Embeddings and stored in a FAISS index.
2. **Retrieval** — When a question is asked, it's embedded using the same model. FAISS finds the 4 most semantically similar chunks using L2 distance.
3. **Generation** — The retrieved chunks are passed as context to Llama 3 on Bedrock. The model answers strictly from this context — no hallucination.

## 🔑 Key Concepts Demonstrated

- RAG pipeline end-to-end
- AWS Bedrock integration (Titan Embeddings + Llama 3)
- Vector similarity search with FAISS
- S3 as cloud PDF storage
- Streamlit for rapid AI app prototyping
- Grounded generation (model refuses to answer outside context)
