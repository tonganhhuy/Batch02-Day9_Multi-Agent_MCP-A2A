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
"""

# =============================================================================
# CONFIGURATION
# =============================================================================

SCORE_THRESHOLD = 0.3   # Nếu best score < threshold → fallback PageIndex
DEFAULT_TOP_K = 5
RERANK_METHOD = "cross_encoder"  # "cross_encoder" | "mmr" | "rrf"


def _fallback_rrf(ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60) -> list[dict]:
    rrf_scores = {}
    content_map = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            content_map.setdefault(key, item.copy())

    results = []
    for content, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]:
        item = content_map[content]
        item["score"] = float(score)
        results.append(item)

    return results


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.
    """
    # Step 1: Run semantic + lexical. Keep imports lazy so missing optional
    # dependencies do not prevent the Streamlit app from starting.
    dense_results = []
    sparse_results = []

    try:
        from .task5_semantic_search import semantic_search
        dense_results = semantic_search(query, top_k=top_k * 2)
    except Exception as e:
        print(f"Semantic search unavailable, continuing without dense results: {e}")

    try:
        from .task6_lexical_search import lexical_search
        sparse_results = lexical_search(query, top_k=top_k * 2)
    except Exception as e:
        print(f"Lexical search unavailable, continuing without sparse results: {e}")

    try:
        from .task7_reranking import rerank, rerank_rrf
    except Exception as e:
        print(f"Reranking unavailable, using simple RRF fallback: {e}")
        rerank = None
        rerank_rrf = _fallback_rrf

    # Step 2: Merge bằng RRF
    merged = rerank_rrf([dense_results, sparse_results], top_k=top_k * 2)
    for item in merged:
        item["source"] = "hybrid"

    # Step 3: Rerank
    if use_reranking and merged and rerank is not None:
        final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
    else:
        final_results = merged[:top_k]

    # Step 4: Check threshold → fallback
    if not final_results or final_results[0]["score"] < score_threshold:
        best_score = final_results[0]["score"] if final_results else 0.0
        print(f"  [Warning] Hybrid score ({best_score:.3f}) "
              f"< threshold ({score_threshold}). Fallback -> PageIndex")
        from .task8_pageindex_vectorless import pageindex_search
        fallback = pageindex_search(query, top_k=top_k)
        return fallback

    return final_results[:top_k]


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý",
        "Nghệ sĩ nào bị bắt vì sử dụng ma tuý năm 2024",
        "Luật phòng chống ma tuý 2021 quy định gì về cai nghiện",
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] [{r['source']}] {r['content'][:80]}...")
