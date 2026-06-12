"""
Setup script - Chạy trong 10 phút có mạng để tải model embedding + reranker.
Usage: uv run python setup.py
"""

from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder

print("=" * 60)
print("  [1/2] DOWNLOADING: keepitreal/vietnamese-sbert")
print("=" * 60)

embed_model = SentenceTransformer("keepitreal/vietnamese-sbert")

# Test nhanh
test = embed_model.encode(["Xin chào Việt Nam"])
print(f"\nEmbedding model loaded! dim = {test.shape[1]}")

print()
print("=" * 60)
print("  [2/2] DOWNLOADING: itdainb/PhoRanker")
print("=" * 60)

reranker = CrossEncoder("itdainb/PhoRanker", max_length=256)

# Test nhanh
score = reranker.predict([("Xin chào", "Chào bạn")])
print(f"\nReranker loaded! test score = {score}")

print()
print("=" * 60)
print("  SETUP COMPLETE - Sẵn sàng thi offline!")
print("  Models cached at: ~/.cache/huggingface/")
print("=" * 60)
