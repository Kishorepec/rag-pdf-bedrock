import boto3
import io
import os
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

load_dotenv()

BUCKET   = os.getenv("S3_BUCKET_NAME")
S3_KEY   = "Kishore_new_resume].pdf"   # ← exact filename you uploaded to S3

# ── 1. Download PDF from S3 into memory ──────────────────────
print(f"📥 Downloading {S3_KEY} from S3 bucket: {BUCKET}...")

s3 = boto3.client("s3")
obj = s3.get_object(Bucket=BUCKET, Key=S3_KEY)
pdf_bytes = io.BytesIO(obj["Body"].read())

print("✅ Downloaded successfully")

# ── 2. Extract text page by page ─────────────────────────────
reader = PdfReader(pdf_bytes)
pages = []

for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if text and text.strip():          # skip blank pages
        pages.append(Document(
            page_content=text,
            metadata={"page": i + 1, "source": S3_KEY}
        ))

print(f"✅ Extracted text from {len(pages)} pages")
print(f"\n--- Sample from Page 1 ---")
print(pages[0].page_content[:400])

# ── 3. Chunk the pages ───────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " "]
)

chunks = splitter.split_documents(pages)

print(f"\n✅ Split into {len(chunks)} chunks")
print(f"\n--- Sample Chunk ---")
print(chunks[0].page_content)
print(f"\n--- Chunk Metadata ---")
print(chunks[0].metadata)

print(f"\n🎯 Average chunk size: {sum(len(c.page_content) for c in chunks) // len(chunks)} chars")