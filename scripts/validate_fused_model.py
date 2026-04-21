from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import DEFAULT_METADATA_CSV, EVALUATION_DIR
from src.evaluation.evaluate import evaluate_all
from src.search.text_source import DEFAULT_TEXT_SOURCE


def main() -> None:
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    if not DEFAULT_METADATA_CSV.exists():
        report_path = EVALUATION_DIR / "fused_model_report.md"
        report_path.write_text(
            "# Fused Model Validation Report\n\nNo metadata file was found. Add data first, then re-run this script.\n",
            encoding="utf-8",
        )
        print(report_path)
        return
    namespace = "fused_validation"
    outputs = evaluate_all(
        text_sources=(DEFAULT_TEXT_SOURCE,),
        artifact_namespace=namespace,
        include_optional=False,
        include_static_reference=True,
    )
    summary = outputs["summary"].copy()
    summary = summary.loc[summary["text_source"] == DEFAULT_TEXT_SOURCE].reset_index(drop=True)
    key_systems = [
        "keyword-bm25",
        "paraphrase-multilingual-MiniLM-L12-v2",
        "multilingual-e5-base",
        "fused-retrieval",
    ]
    filtered = summary.loc[summary["system_name"].isin(key_systems)].copy()
    filtered.to_csv(EVALUATION_DIR / "fused_vs_single_metrics.csv", index=False, encoding="utf-8-sig")

    detail = outputs["detail"].copy()
    fused_detail = detail.loc[
        (detail["system_name"] == "fused-retrieval") & (detail["text_source"] == DEFAULT_TEXT_SOURCE)
    ].reset_index(drop=True)
    weight_rows = fused_detail[["query_id", "query", "adaptive_reason", "raw_score_explanation"]].copy()
    weight_rows.to_csv(EVALUATION_DIR / "fusion_weight_analysis.csv", index=False, encoding="utf-8-sig")

    lines: list[str] = []
    lines.append("# Fused Model Validation Report")
    lines.append("")
    lines.append("## Metrics Comparison")
    lines.append(filtered.to_markdown(index=False))
    lines.append("")
    lines.append("## Fused Weight Reasoning")
    lines.append(weight_rows.head(30).to_markdown(index=False))
    report_path = EVALUATION_DIR / "fused_model_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
