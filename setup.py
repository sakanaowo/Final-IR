"""
Setup script - Chạy trong 10 phút có mạng để tải model embedding.
Usage: uv run python setup.py
"""

from sentence_transformers import SentenceTransformer

print("=" * 60)
print("  DOWNLOADING: keepitreal/vietnamese-sbert")
print("=" * 60)

model = SentenceTransformer("keepitreal/vietnamese-sbert")

# Test nhanh
test = model.encode(["Xin chào Việt Nam"])
print(f"\nModel loaded! Embedding dim = {test.shape[1]}")
print(f"   Model cached at: ~/.cache/huggingface/")
print("=" * 60)
print("  SETUP COMPLETE - Sẵn sàng thi offline!")
print("=" * 60)
