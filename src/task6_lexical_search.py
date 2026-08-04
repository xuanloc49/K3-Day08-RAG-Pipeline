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

from typing import List, Dict, Any
import numpy as np

# TODO: Load corpus từ data/standardized/ hoặc từ vector store
CORPUS: List[Dict[str, Any]] = []  # List of {'content': str, 'metadata': dict}
_bm25_instance = None


def _get_corpus_and_bm25():
    global _bm25_instance, CORPUS
    if _bm25_instance is None:
        from .task4_chunking_indexing import load_documents, chunk_documents
        docs = load_documents()
        CORPUS = chunk_documents(docs)
        if CORPUS:
            from rank_bm25 import BM25Okapi
            tokenized_corpus = [doc["content"].lower().split() for doc in CORPUS]
            _bm25_instance = BM25Okapi(tokenized_corpus)
    return _bm25_instance, CORPUS


def build_bm25_index(corpus: List[Dict[str, Any]]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    global _bm25_instance, CORPUS
    CORPUS = corpus
    from rank_bm25 import BM25Okapi
    tokenized_corpus = [doc["content"].lower().split() for doc in corpus]
    _bm25_instance = BM25Okapi(tokenized_corpus)
    return _bm25_instance


def lexical_search(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
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
    bm25, corpus = _get_corpus_and_bm25()
    if not bm25 or not corpus:
        return []

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "content": corpus[idx]["content"],
            "score": float(scores[idx]),
            "metadata": corpus[idx]["metadata"]
        })
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    # Test
    results = lexical_search("tuition fee payment methods", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")

