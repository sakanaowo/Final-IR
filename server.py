"""
Student RAG Server - Offline RAG Competition
Endpoints: POST /upload, POST /ask
Usage: uv run python server.py
"""

import os

# Offline mode: load model from cache, no network calls
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import re
import logging
from typing import Optional

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import uvicorn

# ============================================================
# CONFIG
# ============================================================
STUDENT_ID = os.environ.get("STUDENT_ID", "B22DCVT028")  # <-- ĐỔI MÃ SV TẠI ĐÂY
TEACHER_BASE = os.environ.get("TEACHER_BASE", "http://192.168.50.218:8000/api/v1")
SERVER_HOST = os.environ.get("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.environ.get("SERVER_PORT", "5000"))
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "64"))
TOP_K = int(os.environ.get("TOP_K", "5"))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================
# MODELS
# ============================================================
logger.info("Loading embedding model...")
embed_model = SentenceTransformer("keepitreal/vietnamese-sbert")
logger.info("Embedding model loaded!")

llm_client = OpenAI(
    base_url=f"{TEACHER_BASE}/proxy",
    api_key=STUDENT_ID,
)


# ============================================================
# VECTOR STORE (in-memory, simple)
# ============================================================
class VectorStore:
    """Minimal in-memory vector store using numpy cosine similarity."""

    def __init__(self):
        self.chunks: list[str] = []
        self.embeddings: np.ndarray | None = None

    def clear(self):
        self.chunks = []
        self.embeddings = None

    def add(self, texts: list[str]):
        if not texts:
            return
        new_embeddings = embed_model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        self.chunks.extend(texts)
        if self.embeddings is None:
            self.embeddings = np.array(new_embeddings)
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

    def search(self, query: str, top_k: int = TOP_K) -> list[str]:
        if self.embeddings is None or len(self.chunks) == 0:
            return []
        q_emb = embed_model.encode([query], normalize_embeddings=True)
        scores = np.dot(self.embeddings, q_emb.T).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self.chunks[i] for i in top_indices]


store = VectorStore()


# ============================================================
# CHUNKING
# ============================================================
def chunk_text(
    text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """Split text into overlapping chunks by character count, respecting sentence boundaries."""
    # Tách theo câu (dấu chấm, chấm hỏi, chấm than, xuống dòng)
    sentences = re.split(r"(?<=[.!?\n])\s+", text.strip())

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Overlap: giữ lại phần cuối
            words = current_chunk.split()
            overlap_text = " ".join(words[-overlap // 4 :]) if overlap > 0 else ""
            current_chunk = overlap_text + " " + sentence
        else:
            current_chunk += " " + sentence if current_chunk else sentence

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    # Fallback: nếu ko tách được câu, cắt theo ký tự
    if not chunks and text.strip():
        for i in range(0, len(text), chunk_size - overlap):
            chunks.append(text[i : i + chunk_size].strip())

    return [c for c in chunks if c]


# ============================================================
# FASTAPI APP
# ============================================================
app = FastAPI(title="Student RAG Server")


# --- Schemas ---
class UploadRequest(BaseModel):
    doc_id: Optional[str] = None
    text: str


class UploadResponse(BaseModel):
    status: str
    doc_id: Optional[str] = None
    chunks: int


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: list[str] = []


# --- Endpoints ---
@app.post("/upload", response_model=UploadResponse)
async def upload(req: UploadRequest):
    """Nhận document từ Teacher Server, chunk + embed + store."""
    logger.info(f"/upload received | doc_id={req.doc_id} | text_len={len(req.text)}")

    # Clear store cũ và nạp document mới
    store.clear()
    text_chunks = chunk_text(req.text)
    store.add(text_chunks)

    logger.info(f"Indexed {len(text_chunks)} chunks")
    return UploadResponse(
        status="success",
        doc_id=req.doc_id,
        chunks=len(text_chunks),
    )


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    """Nhận câu hỏi, retrieve context, gọi LLM proxy, trả answer A/B/C/D."""
    logger.info(f"/ask received | question={req.question[:80]}...")

    # 1. Retrieve relevant chunks
    relevant = store.search(req.question, top_k=TOP_K)
    context = "\n---\n".join(relevant) if relevant else "(Không tìm thấy context)"

    # 2. Build prompt
    prompt = f"""Dựa trên ngữ cảnh bên dưới, hãy trả lời câu hỏi trắc nghiệm.

### Ngữ cảnh:
{context}

### Câu hỏi:
{req.question}

### Hướng dẫn:
- Đọc kỹ ngữ cảnh và câu hỏi.
- Chọn đáp án đúng nhất trong A, B, C, D.
- CHỈ trả lời ĐÚNG 1 KÝ TỰ: A hoặc B hoặc C hoặc D.
- KHÔNG giải thích, KHÔNG viết thêm gì."""

    # 3. Call proxy LLM
    try:
        response = llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Bạn là trợ lý trả lời câu hỏi trắc nghiệm. CHỈ trả lời 1 ký tự: A, B, C hoặc D.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=5,
        )
        raw_answer = response.choices[0].message.content.strip().upper()
    except Exception as e:
        logger.error(f"LLM error: {e}")
        raw_answer = "A"  # Fallback

    # 4. Extract single letter A/B/C/D
    answer = extract_answer(raw_answer)
    logger.info(f"Answer: {answer} (raw: {raw_answer})")

    return AskResponse(answer=answer, sources=relevant[:3])


def extract_answer(raw: str) -> str:
    """Trích xuất A/B/C/D từ output LLM."""
    raw = raw.strip().upper()
    # Thử match trực tiếp
    if raw in ("A", "B", "C", "D"):
        return raw
    # Tìm trong chuỗi
    match = re.search(r"[ABCD]", raw)
    if match:
        return match.group()
    return "A"  # Fallback


@app.get("/health")
async def health():
    return {"status": "ok", "student_id": STUDENT_ID}


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    logger.info(f"Starting Student Server on {SERVER_HOST}:{SERVER_PORT}")
    logger.info(f"   Student ID: {STUDENT_ID}")
    logger.info(f"   Teacher: {TEACHER_BASE}")
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT, log_level="info")
