"""
RAG Evaluation Pipeline — RAGAS.

Framework: RAGAS (chuẩn industry cho RAG evaluation)
pip install ragas datasets

Deliverables:
    - 4 metrics: faithfulness, answer_relevancy, context_recall, context_precision
    - A/B comparison: Config A (hybrid + rerank) vs Config B (dense-only)
    - Export results to results.md

Usage:
    cd <project_root>
    python -m group_project.evaluation.eval_pipeline
"""

import json
import logging
import sys
from pathlib import Path

# Project root → src/ imports work
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

log = logging.getLogger(__name__)

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"

METRICS_ORDER = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]


def load_golden_dataset() -> list[dict]:
    """Load golden dataset từ JSON file."""
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# RAG Pipeline Wrappers (for A/B testing)
# =============================================================================

class HybridRerankedPipeline:
    """Config A: Hybrid (semantic + BM25) → RRF merge → reranking."""

    name = "hybrid_rerank"
    description = "Hybrid search (semantic + BM25) với RRF reranking — full pipeline"

    def generate(self, question: str) -> dict:
        from src.task9_retrieval_pipeline import retrieve
        from src.task10_generation import generate_with_citation

        chunks = retrieve(question, top_k=5, use_reranking=True)
        return generate_with_citation(question, context_chunks=chunks)


class DenseOnlyPipeline:
    """Config B: Dense semantic search only, no BM25, no reranking."""

    name = "dense_only"
    description = "Dense-only search (semantic embedding), không BM25, không reranking"

    def generate(self, question: str) -> dict:
        from src.task5_semantic_search import semantic_search
        from src.task10_generation import generate_with_citation

        chunks = semantic_search(question, top_k=5)
        return generate_with_citation(question, context_chunks=chunks)


# =============================================================================
# RAGAS Evaluation
# =============================================================================

def evaluate_with_ragas(pipeline, golden_dataset: list[dict]):
    """
    Evaluate RAG pipeline sử dụng RAGAS.

    Returns:
        pd.DataFrame với columns: question, answer, contexts, ground_truth,
                                   faithfulness, answer_relevancy,
                                   context_recall, context_precision
    """
    import os
    import time
    from ragas import evaluate
    from ragas.run_config import RunConfig
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
    )
    from datasets import Dataset

    eval_data: dict[str, list] = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    # Step 1: Sinh câu trả lời từ pipeline với khoảng nghỉ 5s để tránh Rate Limit (15 RPM)
    for idx, item in enumerate(golden_dataset):
        if idx > 0:
            log.info("Sleeping 5 seconds to respect Gemini rate limits (15 RPM)...")
            time.sleep(5.0)

        try:
            result = pipeline.generate(item["question"])
            contexts = [c["content"] for c in result.get("sources", []) if c.get("content")]
        except Exception as e:
            log.error(f"Pipeline error on '{item['question'][:50]}': {e}")
            result = {"answer": ""}
            contexts = []

        eval_data["question"].append(item["question"])
        eval_data["answer"].append(result.get("answer", ""))
        eval_data["contexts"].append(contexts or [""])  # RAGAS yêu cầu danh sách không rỗng
        eval_data["ground_truth"].append(item["expected_answer"])

    dataset = Dataset.from_dict(eval_data)

    # Step 2: Cấu hình Ragas sử dụng Gemini làm giám khảo (nếu có key)
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    ragas_kwargs = {}

    if gemini_key:
        log.info("Configuring RAGAS to use Gemini for evaluation...")
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper

        gemini_llm = ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite",
            google_api_key=gemini_key,
            temperature=0.0
        )
        gemini_embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=gemini_key
        )

        ragas_kwargs["llm"] = LangchainLLMWrapper(gemini_llm)
        ragas_kwargs["embeddings"] = LangchainEmbeddingsWrapper(gemini_embeddings)
    else:
        log.warning("GEMINI_API_KEY not found in .env. RAGAS will fallback to OpenAI default.")

    # Step 3: Thiết lập RunConfig chạy tuần tự để tránh Rate Limit
    run_config = RunConfig(
        max_workers=1,      # Chạy tuần tự 1 luồng
        max_retries=10,     # Tự động thử lại 10 lần nếu bị chặn (HTTP 429)
        timeout=180,
    )

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        run_config=run_config,
        **ragas_kwargs
    )
    return result.to_pandas()


# =============================================================================
# A/B Comparison
# =============================================================================

def compare_configs(golden_dataset: list[dict]) -> dict:
    """
    A/B comparison giữa 2 configs:
        Config A — hybrid search + RRF reranking (full pipeline)
        Config B — dense-only search, no reranking

    Returns:
        dict[config_name, pd.DataFrame]
    """
    pipelines = [HybridRerankedPipeline(), DenseOnlyPipeline()]

    results = {}
    for pipeline in pipelines:
        log.info(f"Evaluating config: {pipeline.name} ...")
        results[pipeline.name] = evaluate_with_ragas(pipeline, golden_dataset)
        log.info(f"  Done: {pipeline.name}")

    return results


# =============================================================================
# Export Results
# =============================================================================

def export_results(results, comparison: dict):
    """
    Export evaluation results to results.md.

    Args:
        results: pd.DataFrame from a single run (used for worst-performer analysis)
        comparison: dict[config_name, pd.DataFrame] from compare_configs()
    """
    config_names = list(comparison.keys())

    # Per-config metric averages
    scores: dict[str, dict[str, float]] = {}
    for name, df in comparison.items():
        scores[name] = {
            m: round(float(df[m].mean()), 3)
            for m in METRICS_ORDER
            if m in df.columns
        }

    # ==== Build markdown ====
    lines = []

    lines += [
        "# RAG Evaluation Results\n\n",
        "## Framework sử dụng\n\n",
        "**RAGAS** — chuẩn industry cho RAG evaluation, đánh giá theo 3 trục: "
        "faithfulness, relevance, context quality.\n\n",
        "---\n\n",
        "## Overall Scores\n\n",
    ]

    # Score table header
    header_cols = " | ".join(config_names)
    sep_cols = "|".join(["--------"] * len(config_names))
    lines += [
        f"| Metric | {header_cols} | Δ |\n",
        f"|--------|{sep_cols}|---|\n",
    ]

    col_avgs = {n: [] for n in config_names}
    for m in METRICS_ORDER:
        vals = [scores.get(n, {}).get(m, 0.0) for n in config_names]
        for n, v in zip(config_names, vals):
            col_avgs[n].append(v)
        delta = vals[-1] - vals[0] if len(vals) >= 2 else 0.0
        score_str = " | ".join(f"{v:.3f}" for v in vals)
        lines.append(f"| {m} | {score_str} | {delta:+.3f} |\n")

    avgs = [sum(col_avgs[n]) / len(col_avgs[n]) for n in config_names]
    delta_avg = avgs[-1] - avgs[0] if len(avgs) >= 2 else 0.0
    avg_str = " | ".join(f"**{v:.3f}**" for v in avgs)
    lines.append(f"| **Average** | {avg_str} | {delta_avg:+.3f} |\n")

    # ==== A/B Analysis ====
    lines += ["\n---\n\n", "## A/B Comparison Analysis\n\n"]

    pipeline_descs = {
        "hybrid_rerank": "Hybrid search = semantic (dense) + BM25 (lexical) → RRF merge → cross-encoder reranking. Full pipeline với đầy đủ components.",
        "dense_only": "Dense-only = chỉ semantic search bằng embedding cosine similarity, không có BM25 và không reranking. Baseline đơn giản.",
    }

    for i, name in enumerate(config_names):
        label = "A" if i == 0 else "B"
        desc = pipeline_descs.get(name, name)
        lines.append(f"**Config {label} ({name}):**\n> {desc}\n\n")

    lines.append("**Kết luận:**\n")
    if len(avgs) >= 2:
        a, b = avgs[0], avgs[1]
        a_name, b_name = config_names[0], config_names[1]
        if a >= b:
            diff = a - b
            lines.append(
                f"> Config A ({a_name}) vượt trội Config B ({b_name}): "
                f"{a:.3f} vs {b:.3f} điểm trung bình (Δ={diff:+.3f}). "
                f"BM25 bổ sung recall cho từ khoá pháp luật cụ thể; "
                f"reranking cải thiện precision của kết quả cuối cùng.\n\n"
            )
        else:
            diff = b - a
            lines.append(
                f"> Config B ({b_name}) đạt điểm cao hơn Config A ({a_name}): "
                f"{b:.3f} vs {a:.3f} (Δ={diff:+.3f}). "
                f"Embedding đủ mạnh để capture semantic similarity cho dataset này; "
                f"thêm BM25 có thể gây noise với corpus nhỏ.\n\n"
            )

    # ==== Worst Performers ====
    lines += ["---\n\n", "## Worst Performers (Bottom 3)\n\n"]

    # Use first comparison config's df for worst-performer analysis
    analysis_df = results if results is not None else (next(iter(comparison.values())) if comparison else None)
    if analysis_df is not None:
        available = [m for m in METRICS_ORDER if m in analysis_df.columns]
        if available:
            df = analysis_df.copy()
            df["_avg"] = df[available].mean(axis=1)
            worst = df.nsmallest(3, "_avg")

            lines += [
                "| # | Question | Faithfulness | Answer Relevancy | Context Recall | Failure Stage | Root Cause |\n",
                "|---|----------|-------------|-----------------|----------------|---------------|------------|\n",
            ]
            for idx, (_, row) in enumerate(worst.iterrows(), 1):
                q = str(row.get("question", ""))[:55]
                f_s = row.get("faithfulness", 0.0)
                ar_s = row.get("answer_relevancy", 0.0)
                cr_s = row.get("context_recall", 0.0)

                if cr_s < 0.4:
                    stage, cause = "Retrieval", "Retriever không tìm được context liên quan"
                elif f_s < 0.5:
                    stage, cause = "Generation", "LLM hallucinate — câu trả lời không bám context"
                else:
                    stage, cause = "Answer", "Answer relevancy thấp — trả lời lạc đề"

                lines.append(
                    f"| {idx} | {q}... | {f_s:.3f} | {ar_s:.3f} | {cr_s:.3f} "
                    f"| {stage} | {cause} |\n"
                )

    # ==== Recommendations ====
    lines += [
        "\n---\n\n",
        "## Recommendations\n\n",
        "### Cải tiến 1: Nâng cấp chunking strategy\n",
        "**Action:** Dùng `MarkdownHeaderTextSplitter` để giữ nguyên cấu trúc điều khoản pháp luật "
        "(mỗi chunk = 1 điều, không cắt ngang điều khoản).  \n",
        "**Expected impact:** Tăng context precision ~0.10–0.15 vì chunk không bị cắt giữa chừng "
        "khiến LLM mất ngữ cảnh pháp lý.\n\n",
        "### Cải tiến 2: Vietnamese tokenizer cho BM25\n",
        "**Action:** Tích hợp `underthesea` hoặc `pyvi` để word-segment tiếng Việt trước BM25, "
        "thay vì whitespace tokenization hiện tại.  \n",
        "**Expected impact:** Tăng lexical search recall ~0.10 vì BM25 hiện nhầm lẫn token "
        "ghép âm tiết (VD: 'ma tuý' vs 'matuý').  \n\n",
        "### Cải tiến 3: Mở rộng corpus\n",
        "**Action:** Thu thập thêm Thông tư, Nghị định mới (sau 2022) và bài báo năm 2023–2025.  \n",
        "**Expected impact:** Tăng context recall ~0.15 cho câu hỏi về quy định mới và vụ án gần đây.\n",
    ]

    RESULTS_PATH.write_text("".join(lines), encoding="utf-8")
    log.info(f"Results written to {RESULTS_PATH}")
    print(f"Results exported → {RESULTS_PATH}")


# =============================================================================
# Option 1: DeepEval (alternative — commented out)
# =============================================================================

def evaluate_with_deepeval(pipeline, golden_dataset: list[dict]) -> dict:
    """
    Evaluate RAG pipeline sử dụng DeepEval.
    pip install deepeval

    Uncomment to use as alternative to RAGAS.
    """
    # from deepeval import evaluate
    # from deepeval.metrics import (
    #     FaithfulnessMetric,
    #     AnswerRelevancyMetric,
    #     ContextualRecallMetric,
    #     ContextualPrecisionMetric,
    # )
    # from deepeval.test_case import LLMTestCase
    #
    # test_cases = []
    # for item in golden_dataset:
    #     result = pipeline.generate(item["question"])
    #     test_cases.append(LLMTestCase(
    #         input=item["question"],
    #         actual_output=result["answer"],
    #         expected_output=item["expected_answer"],
    #         retrieval_context=[c["content"] for c in result["sources"]],
    #     ))
    #
    # metrics = [
    #     FaithfulnessMetric(threshold=0.7),
    #     AnswerRelevancyMetric(threshold=0.7),
    #     ContextualRecallMetric(threshold=0.7),
    #     ContextualPrecisionMetric(threshold=0.7),
    # ]
    # return evaluate(test_cases, metrics)
    raise NotImplementedError("Uncomment DeepEval code above to use this option.")


# =============================================================================
# Option 3: TruLens (alternative — commented out)
# =============================================================================

def evaluate_with_trulens(pipeline, golden_dataset: list[dict]) -> dict:
    """
    Evaluate RAG pipeline sử dụng TruLens.
    pip install trulens trulens-providers-openai

    Uncomment to use as alternative to RAGAS.
    """
    # from trulens.apps.custom import TruCustomApp
    # from trulens.core import Feedback
    # from trulens.providers.openai import OpenAI as TruOpenAI
    #
    # provider = TruOpenAI()
    # f_faithfulness = Feedback(provider.groundedness_measure_with_cot_reasons).on_output()
    # f_relevance = Feedback(provider.relevance).on_input_output()
    # f_context_relevance = Feedback(provider.context_relevance).on_input()
    #
    # tru_rag = TruCustomApp(
    #     pipeline,
    #     app_name="DrugLaw_RAG",
    #     feedbacks=[f_faithfulness, f_relevance, f_context_relevance],
    # )
    # with tru_rag as recording:
    #     for item in golden_dataset:
    #         pipeline.generate(item["question"])
    #
    # from trulens.dashboard import run_dashboard
    # run_dashboard()
    raise NotImplementedError("Uncomment TruLens code above to use this option.")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    golden_dataset = load_golden_dataset()
    print(f"Loaded {len(golden_dataset)} test cases from golden_dataset.json")

    # Single-config evaluation (Config A — full pipeline)
    pipeline_a = HybridRerankedPipeline()
    print(f"\nRunning RAGAS evaluation on Config A ({pipeline_a.name}) ...")
    results_df = evaluate_with_ragas(pipeline_a, golden_dataset)
    print(results_df[METRICS_ORDER].describe())

    # A/B comparison
    print("\nRunning A/B comparison (hybrid_rerank vs dense_only) ...")
    comparison = compare_configs(golden_dataset)

    # Export
    export_results(results_df, comparison)