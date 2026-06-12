"""
Student RAG Server - Offline RAG Competition
Endpoints: POST /upload, POST /ask
Usage: uv run python server.py
"""

import os

# Offline mode: load model from cache, no network calls
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from pathlib import Path
import re
import logging
import threading
from datetime import datetime
from typing import Optional

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from openai import OpenAI
import uvicorn

# ============================================================
# CONFIG
# ============================================================
from dotenv import load_dotenv

load_dotenv(".env.local")

STUDENT_ID = os.environ.get("STUDENT_ID", "B22DCVT028")  # <-- ĐỔI MÃ SV TẠI ĐÂY
TEACHER_BASE = os.environ.get("TEACHER_BASE", "http://192.168.50.218:8000/api/v1")
LLM_BASE_URL = os.environ.get("BASE_URL", f"{TEACHER_BASE}/proxy")
LLM_API_KEY = os.environ.get("API_KEY", STUDENT_ID)
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
SERVER_HOST = os.environ.get("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.environ.get("SERVER_PORT", "5000"))
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "64"))
TOP_K = int(os.environ.get("TOP_K", "5"))
RERANK_CANDIDATES = int(os.environ.get("RERANK_CANDIDATES", "15"))  # Retrieve nhiều hơn rồi rerank
VECTOR_DB_PATH = Path(os.environ.get("VECTOR_DB_PATH", "vector_store.npz"))
QUESTION_LOG_PATH = Path(os.environ.get("QUESTION_LOG_PATH", "teacher_questions.md"))

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

logger.info("Loading reranker model...")
reranker = CrossEncoder("itdainb/PhoRanker", max_length=256)
logger.info("Reranker model loaded!")

llm_client = OpenAI(
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY,
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

    def save(self, path: Path = VECTOR_DB_PATH):
        if self.embeddings is None or not self.chunks:
            logger.warning("Skip saving empty vector store")
            return
        np.savez_compressed(
            path,
            chunks=np.array(self.chunks, dtype=object),
            embeddings=self.embeddings,
        )
        logger.info(f"Saved vector store to {path} | chunks={len(self.chunks)}")

    def load(self, path: Path = VECTOR_DB_PATH) -> bool:
        if not path.exists():
            logger.info(f"No persisted vector store found at {path}")
            return False
        try:
            data = np.load(path, allow_pickle=True)
            chunks = data["chunks"].tolist()
            embeddings = data["embeddings"]
        except Exception as e:
            logger.error(f"Failed to load vector store from {path}: {e}")
            return False

        if not chunks or embeddings is None or len(chunks) != len(embeddings):
            logger.error(f"Invalid vector store at {path}")
            return False

        self.chunks = list(chunks)
        self.embeddings = np.array(embeddings)
        logger.info(f"Loaded vector store from {path} | chunks={len(self.chunks)}")
        return True

    def search(self, query: str, top_k: int = TOP_K) -> list[str]:
        if self.embeddings is None or len(self.chunks) == 0:
            return []
        q_emb = embed_model.encode([query], normalize_embeddings=True)
        scores = np.dot(self.embeddings, q_emb.T).flatten()

        # Lấy nhiều candidates hơn rồi rerank
        n_candidates = min(RERANK_CANDIDATES, len(self.chunks))
        candidate_indices = np.argsort(scores)[::-1][:n_candidates]
        candidates = [self.chunks[i] for i in candidate_indices]

        # Rerank bằng cross-encoder
        if len(candidates) > top_k:
            pairs = [(query, doc) for doc in candidates]
            rerank_scores = reranker.predict(pairs)
            reranked_indices = np.argsort(rerank_scores)[::-1][:top_k]
            return [candidates[i] for i in reranked_indices]

        return candidates[:top_k]


store = VectorStore()
store.load()


# ============================================================
# QUESTION LOGGING
# ============================================================
question_counter = 0
question_log_lock = threading.Lock()


def append_question_log(question: str) -> int:
    """Append the raw teacher question to a markdown log file."""
    global question_counter
    with question_log_lock:
        question_counter += 1
        question_num = question_counter

        QUESTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not QUESTION_LOG_PATH.exists():
            QUESTION_LOG_PATH.write_text("# Teacher Questions\n\n", encoding="utf-8")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = [
            f"## Question {question_num}",
            "",
            "- Local answer: pending",
            f"- Time: {timestamp}",
            "",
            "```text",
            question.strip(),
            "```",
            "",
        ]
        with QUESTION_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write("\n".join(entry) + "\n")
        return question_num


def rewrite_question_answer(question_num: int, answer: str):
    """Update the final local answer for one logged question."""
    if not QUESTION_LOG_PATH.exists():
        return
    with question_log_lock:
        text = QUESTION_LOG_PATH.read_text(encoding="utf-8")
        marker = f"## Question {question_num}\n\n"
        idx = text.find(marker)
        if idx == -1:
            return
        old = marker + "- Local answer: pending"
        new = marker + f"- Local answer: {answer}"
        text = text.replace(old, new, 1)
        QUESTION_LOG_PATH.write_text(text, encoding="utf-8")


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
    sources: list[str] = Field(default_factory=list)


# --- Endpoints ---
@app.post("/upload", response_model=UploadResponse)
async def upload(req: UploadRequest):
    """Nhận document từ Teacher Server, chunk + embed + store."""
    logger.info(f"/upload received | doc_id={req.doc_id} | text_len={len(req.text)}")

    # Clear store cũ và nạp document mới
    store.clear()
    text_chunks = chunk_text(req.text)
    store.add(text_chunks)
    store.save()

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
    question_num = None
    try:
        question_num = append_question_log(req.question)
    except Exception as e:
        logger.error(f"Question log error: {e}")

    if store.embeddings is None or not store.chunks:
        store.load()

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
            model=MODEL_NAME,
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
    if question_num is not None:
        try:
            rewrite_question_answer(question_num, answer)
        except Exception as e:
            logger.error(f"Question answer log error: {e}")
    logger.info(f"Answer: {answer} (raw: {raw_answer})")

    return AskResponse(answer=answer, sources=relevant[:3])


def extract_answer(raw: str) -> str:
    """Trích xuất A/B/C/D từ output LLM."""
    raw = raw.strip().upper()
    # Thử match trực tiếp
    if raw in ("A", "B", "C", "D"):
        return raw

    # Bắt các dạng phổ biến: "Đáp án: B", "Answer is B", "(B)", "B."
    patterns = [
        r"(?:ĐÁP\s*ÁN|DAP\s*AN|ANSWER|OPTION|CHỌN|CHON)\s*(?:LÀ|LA|IS|:)?\s*([ABCD])\b",
        r"\b([ABCD])\s*[\.\)]",
        r"\b([ABCD])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, raw)
        if match:
            return match.group(1)

    # Fallback cuối cùng nếu model vẫn trả kèm format lạ.
    match = re.search(r"[ABCD]", raw)
    if match:
        return match.group()
    return "A"  # Fallback


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "student_id": STUDENT_ID,
        "chunks": len(store.chunks),
        "vector_db_path": str(VECTOR_DB_PATH),
        "vector_db_exists": VECTOR_DB_PATH.exists(),
        "question_log_path": str(QUESTION_LOG_PATH),
        "questions_logged": question_counter,
    }


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    logger.info(f"Starting Student Server on {SERVER_HOST}:{SERVER_PORT}")
    logger.info(f"   Student ID: {STUDENT_ID}")
    logger.info(f"   Teacher: {TEACHER_BASE}")
    logger.info(f"   LLM: {MODEL_NAME} @ {LLM_BASE_URL}")
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT, log_level="info")
