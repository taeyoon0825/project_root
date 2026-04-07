from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import DEFAULT_METADATA_CSV, DEFAULT_QUERYSET_CSV, EVALUATION_DIR, ensure_project_dirs
from src.data.metadata_schema import load_metadata_frame
from src.embedding.build_indices import DenseSearchEngine
from src.embedding.vector_models import list_available_models
from src.search.keyword_search import KeywordSearchEngine
from src.search.text_source import DEFAULT_TEXT_SOURCE
from src.utils.io_utils import save_dataframe, save_json
from src.visualize.clustering import load_cluster_summary


def _topk_hit(results: pd.DataFrame, target_category: str, k: int) -> int:
    topk = results.head(k)
    return int((topk["category"] == target_category).any())


def evaluate_keyword_engine(
    queryset: pd.DataFrame,
    metadata: pd.DataFrame,
    method: str,
    text_source: str,
) -> tuple[pd.DataFrame, dict]:
    engine = KeywordSearchEngine(metadata, text_source=text_source)
    rows = []
    system_name = f"keyword-{method}"

    for _, query_row in queryset.iterrows():
        results = engine.search(query_row["query"], top_k=3, method=method)
        rows.append(
            {
                "system_name": system_name,
                "system_type": "keyword",
                "text_source": text_source,
                "query_id": query_row["query_id"],
                "query": query_row["query"],
                "target_category": query_row["target_category"],
                "top1_hit": _topk_hit(results, query_row["target_category"], 1),
                "top3_hit": _topk_hit(results, query_row["target_category"], 3),
                "top1_id": results.iloc[0]["id"],
                "top1_category": results.iloc[0]["category"],
            }
        )

    frame = pd.DataFrame(rows)
    summary = {
        "system_name": system_name,
        "system_type": "keyword",
        "text_source": text_source,
        "top1_accuracy": float(frame["top1_hit"].mean()),
        "top3_accuracy": float(frame["top3_hit"].mean()),
        "silhouette_score": None,
    }
    return frame, summary


def evaluate_dense_engine(
    queryset: pd.DataFrame,
    metadata: pd.DataFrame,
    model_alias: str,
    text_source: str,
) -> tuple[pd.DataFrame, dict]:
    engine = DenseSearchEngine(metadata, model_alias, text_source=text_source)
    engine.load()
    rows = []

    for _, query_row in queryset.iterrows():
        results = engine.search(query_row["query"], top_k=3)
        rows.append(
            {
                "system_name": model_alias,
                "system_type": "dense",
                "text_source": text_source,
                "query_id": query_row["query_id"],
                "query": query_row["query"],
                "target_category": query_row["target_category"],
                "top1_hit": _topk_hit(results, query_row["target_category"], 1),
                "top3_hit": _topk_hit(results, query_row["target_category"], 3),
                "top1_id": results.iloc[0]["id"],
                "top1_category": results.iloc[0]["category"],
            }
        )

    frame = pd.DataFrame(rows)
    try:
        cluster_summary = load_cluster_summary(model_alias, text_source=text_source, method="kmeans")
        silhouette_score = cluster_summary.get("silhouette_score")
    except Exception:
        silhouette_score = None

    summary = {
        "system_name": model_alias,
        "system_type": "dense",
        "text_source": text_source,
        "top1_accuracy": float(frame["top1_hit"].mean()),
        "top3_accuracy": float(frame["top3_hit"].mean()),
        "silhouette_score": silhouette_score,
    }
    return frame, summary


def build_source_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    base = summary[["system_name", "system_type", "text_source", "top1_accuracy", "top3_accuracy"]].copy()
    pivot = base.pivot_table(
        index=["system_name", "system_type"],
        columns="text_source",
        values=["top1_accuracy", "top3_accuracy"],
    )
    pivot.columns = ["__".join(column).strip() for column in pivot.columns.to_flat_index()]
    pivot = pivot.reset_index()
    if "top1_accuracy__original_transcript" in pivot.columns and "top1_accuracy__stt_transcript" in pivot.columns:
        pivot["top1_delta_stt_minus_original"] = (
            pivot["top1_accuracy__stt_transcript"] - pivot["top1_accuracy__original_transcript"]
        )
    if "top3_accuracy__original_transcript" in pivot.columns and "top3_accuracy__stt_transcript" in pivot.columns:
        pivot["top3_delta_stt_minus_original"] = (
            pivot["top3_accuracy__stt_transcript"] - pivot["top3_accuracy__original_transcript"]
        )
    return pivot


def evaluate_all(
    metadata_path: Path = DEFAULT_METADATA_CSV,
    queryset_path: Path = DEFAULT_QUERYSET_CSV,
    text_sources: tuple[str, ...] = ("stt_transcript", "original_transcript"),
    include_optional: bool = False,
) -> dict[str, pd.DataFrame]:
    ensure_project_dirs()
    metadata = load_metadata_frame(metadata_path)
    queryset = pd.read_csv(queryset_path)

    detail_frames = []
    summary_rows = []

    for text_source in text_sources:
        for method in ["bm25", "tfidf"]:
            detail_frame, summary = evaluate_keyword_engine(queryset, metadata, method, text_source)
            detail_frames.append(detail_frame)
            summary_rows.append(summary)

        for model_alias in list(list_available_models(include_optional=include_optional).keys()):
            detail_frame, summary = evaluate_dense_engine(queryset, metadata, model_alias, text_source)
            detail_frames.append(detail_frame)
            summary_rows.append(summary)

    detail = pd.concat(detail_frames, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    comparison = build_source_comparison(summary)

    save_dataframe(EVALUATION_DIR / "retrieval_eval_detail.csv", detail)
    save_dataframe(EVALUATION_DIR / "retrieval_eval_summary.csv", summary)
    save_dataframe(EVALUATION_DIR / "retrieval_eval_source_comparison.csv", comparison)
    save_json(EVALUATION_DIR / "retrieval_eval_summary.json", summary.to_dict(orient="records"))
    return {"detail": detail, "summary": summary, "comparison": comparison}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval systems on sample query set.")
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_CSV)
    parser.add_argument("--queryset-path", type=Path, default=DEFAULT_QUERYSET_CSV)
    parser.add_argument(
        "--text-sources",
        nargs="+",
        default=[DEFAULT_TEXT_SOURCE, "original_transcript"],
        choices=["stt_transcript", "original_transcript", "combined"],
    )
    parser.add_argument("--include-optional", action="store_true")
    args = parser.parse_args()

    outputs = evaluate_all(
        args.metadata_path,
        args.queryset_path,
        text_sources=tuple(args.text_sources),
        include_optional=args.include_optional,
    )
    print(outputs["summary"].to_string(index=False))


if __name__ == "__main__":
    main()
