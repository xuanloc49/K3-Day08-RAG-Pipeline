"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

from pathlib import Path

from rank_bm25 import BM25Okapi

from .task4_chunking_indexing import chunk_documents, load_documents

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"

# Corpus + BM25 index (lazy-init)
CORPUS: list[dict] = []  # List of {'content': str, 'metadata': dict}
_bm25: BM25Okapi | None = None


def _load_corpus() -> list[dict]:
    """Load markdown docs rồi chunk giống Task 4 để BM25 search trên cùng đơn vị."""
    docs = load_documents()
    if docs:
        return chunk_documents(docs)

    # Fallback: đọc raw .md nếu load_documents trống
    corpus = []
    if STANDARDIZED_DIR.exists():
        for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
            content = md_file.read_text(encoding="utf-8")
            if content.strip():
                corpus.append({
                    "content": content,
                    "metadata": {
                        "source": md_file.name,
                        "type": "legal" if "legal" in str(md_file) else "news",
                    },
                })
    return corpus


def _ensure_index() -> None:
    """Khởi tạo CORPUS + BM25 index một lần."""
    global CORPUS, _bm25
    if _bm25 is not None:
        return
    CORPUS = _load_corpus()
    if CORPUS:
        _bm25 = build_bm25_index(CORPUS)


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    tokenized_corpus = [doc["content"].lower().split() for doc in corpus]
    return BM25Okapi(tokenized_corpus)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    _ensure_index()
    if not CORPUS or _bm25 is None:
        return []

    tokenized_query = query.lower().split()
    scores = _bm25.get_scores(tokenized_query)

    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

    results = []
    for idx, score in ranked[:top_k]:
        if score > 0:
            results.append({
                "content": CORPUS[idx]["content"],
                "score": float(score),
                "metadata": CORPUS[idx]["metadata"],
            })
    return results


if __name__ == "__main__":
    # Test
    results = lexical_search("tuition fee payment methods", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
