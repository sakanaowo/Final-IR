"""
Setup script - Chạy trong 10 phút có mạng để tải model embedding + reranker.
Usage: uv run python setup.py
"""

import os

import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder


def resolve_model_device() -> str:
    requested_device = (
        os.environ.get("MODEL_DEVICE") or os.environ.get("DEVICE") or "auto"
    ).strip().lower()
    if requested_device not in ("", "auto"):
        if requested_device == "cuda" and not torch.cuda.is_available():
            print(
                "WARNING: MODEL_DEVICE=cuda was requested, but torch cannot see CUDA. "
                "Falling back to CPU."
            )
            return "cpu"
        return requested_device

    return "cuda" if torch.cuda.is_available() else "cpu"


DEVICE = resolve_model_device()

print("=" * 60)
print(f"  TORCH: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
print(f"  CUDA build: {torch.version.cuda}")
print(f"  Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

print("=" * 60)
print("  [1/2] DOWNLOADING: keepitreal/vietnamese-sbert")
print("=" * 60)

embed_model = SentenceTransformer("keepitreal/vietnamese-sbert", device=DEVICE)

# Test nhanh
test = embed_model.encode(["Xin chào Việt Nam"])
print(f"\nEmbedding model loaded! dim = {test.shape[1]}")

print()
print("=" * 60)
print("  [2/2] DOWNLOADING: itdainb/PhoRanker")
print("=" * 60)

reranker = CrossEncoder("itdainb/PhoRanker", max_length=256, device=DEVICE)

# Test nhanh
score = reranker.predict([("Xin chào", "Chào bạn")])
print(f"\nReranker loaded! test score = {score}")

print()
print("=" * 60)
print("  SETUP COMPLETE - Sẵn sàng thi offline!")
print("  Models cached at: ~/.cache/huggingface/")
print("=" * 60)
