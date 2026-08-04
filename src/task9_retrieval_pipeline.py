"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic search + lexical search + reranking + PageIndex fallback
thành một pipeline thống nhất.

Logic:
    1. Chạy semantic_search + lexical_search song song
    2. Merge kết quả (RRF hoặc weighted fusion)
    3. Rerank
    4. Nếu top result score < threshold → fallback sang PageIndex
    5. Return top_k results

⚠️ BẪY THƯỜNG GẶP — đọc kỹ trước khi code:
    Nếu bạn dùng điểm RRF đã fuse (Task 7) để so với score_threshold, bạn sẽ gặp bug
    thật: RRF max score luôn ≈ 1/(k+1) ≈ 0.0164 (k=60) BẤT KỂ nội dung có liên quan
    hay không. Nếu đặt threshold thấp (như 0.005) để "hợp" với thang điểm RRF, thực
    chất KHÔNG câu hỏi nào đủ thấp để trigger fallback nữa — kể cả query hoàn toàn vô
    nghĩa vẫn trả về kết quả "hybrid" (rác) thay vì fallback đúng như thiết kế.

    Cách sửa đúng: giữ điểm cosine similarity GỐC của semantic_search (trước khi qua
    RRF) làm căn cứ quyết định fallback, tách biệt khỏi điểm RRF dùng để sắp xếp kết
    quả cuối cùng. Calibrate threshold bằng cách tự đo: chạy vài câu hỏi chắc chắn
    liên quan và vài câu chắc chắn lạc đề/rác qua semantic_search, xem khoảng cách
    điểm số giữa hai nhóm rồi chọn ngưỡng nằm giữa.
"""

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

# TODO: Calibrate threshold này bằng cách tự đo điểm cosine của semantic_search
# cho câu hỏi liên quan vs câu hỏi lạc đề (xem ghi chú ở trên) — ĐỪNG copy nguyên
# giá trị mẫu, mỗi corpus/embedding model sẽ cho khoảng điểm khác nhau.
SCORE_THRESHOLD = 0.48   # Ngưỡng Cosine gốc tối thiểu (< 0.48 -> fallback PageIndex)
DEFAULT_TOP_K = 5
RERANK_METHOD = "rrf"  # "cross_encoder" | "mmr" | "rrf"


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
    use_semantic: bool = True,
    use_bm25: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic và các mode bật/tắt demo.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả cuối cùng
        score_threshold: Ngưỡng điểm cosine gốc tối thiểu (KHÔNG phải điểm RRF)
        use_reranking: Có áp dụng reranking hay không
        use_semantic: Có áp dụng Semantic Search (Dense) hay không
        use_bm25: Có áp dụng Lexical Search (BM25 Sparse) hay không

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': str  # 'hybrid', 'semantic', 'bm25', 'pageindex'
        }
    """
    dense_results = []
    sparse_results = []

    # Step 1: Chạy semantic search nếu được bật
    if use_semantic:
        try:
            dense_results = semantic_search(query, top_k=top_k * 2)
        except Exception as e:
            print(f"  ⚠ Semantic search error: {e}")
            dense_results = []

    # Step 2: Chạy lexical search nếu được bật
    if use_bm25:
        try:
            sparse_results = lexical_search(query, top_k=top_k * 2)
        except Exception as e:
            print(f"  ⚠ Lexical search error: {e}")
            sparse_results = []

    # Gom nhóm các danh sách tìm kiếm được
    ranked_lists = []
    if dense_results:
        ranked_lists.append(dense_results)
    if sparse_results:
        ranked_lists.append(sparse_results)

    # Xác định nhãn source mặc định
    if use_semantic and use_bm25:
        default_source_tag = "hybrid"
    elif use_semantic:
        default_source_tag = "semantic"
    elif use_bm25:
        default_source_tag = "bm25"
    else:
        default_source_tag = "none"

    # Step 3: Merge & Rerank
    if not ranked_lists:
        merged = []
    elif use_reranking:
        if len(ranked_lists) > 1 or RERANK_METHOD == "rrf":
            merged = rerank_rrf(ranked_lists, top_k=top_k * 2)
        else:
            merged = ranked_lists[0][:top_k * 2]

        if RERANK_METHOD != "rrf" and merged:
            try:
                merged = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
            except Exception as e:
                print(f"  ⚠ Reranking error with method {RERANK_METHOD}: {e}")
                merged = merged[:top_k]
    else:
        # Khi KHÔNG dùng reranking: gộp danh sách loại bỏ trùng lặp nội dung
        seen_content = set()
        merged = []
        for r_list in ranked_lists:
            for item in r_list:
                if item["content"] not in seen_content:
                    seen_content.add(item["content"])
                    merged.append(item.copy())

    for item in merged:
        if "source" not in item or not item["source"]:
            item["source"] = default_source_tag

    final_results = merged[:top_k]

    # Step 4: Check threshold & Fallback Logic
    need_fallback = False
    if use_semantic and dense_results:
        best_score = dense_results[0]["score"]
        if best_score < score_threshold:
            print(f"  ⚠ Semantic best score ({best_score:.3f}) < threshold ({score_threshold})")
            need_fallback = True
    elif not final_results:
        need_fallback = True

    if need_fallback:
        try:
            fallback = pageindex_search(query, top_k=top_k)
            if fallback:
                return fallback
        except Exception as e:
            print(f"  ⚠ PageIndex fallback error: {e}")

    return final_results


if __name__ == "__main__":
    test_queries = [
        "What is the tuition fee at RMIT Vietnam?",
        "How do I book a library study room?",
        "What scholarships are available for international students?",
        "xyzabc123nonsense",  # Query không có kết quả → test fallback
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] [{r['source']}] {r['content'][:80]}...")
