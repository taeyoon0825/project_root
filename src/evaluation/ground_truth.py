from __future__ import annotations

from typing import Any

import pandas as pd


GROUND_TRUTH_COLUMNS = [
    "query_id",
    "query",
    "query_preview",
    "target_category",
    "target_source_type",
    "relevant_ids",
    "relevant_file_names",
    "relevant_line_numbers",
    "relevant_segment_indexes",
    "relevant_segment_texts",
    "relevant_count",
    "evaluation_level",
    "ground_truth_rule",
]


def _split_values(value: Any) -> list[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    values = [text]
    for separator in [",", ";", "|"]:
        expanded: list[str] = []
        for item in values:
            expanded.extend(item.split(separator))
        values = expanded
    return [item.strip() for item in values if item.strip()]


def _split_int_values(value: Any) -> list[int]:
    items: list[int] = []
    for item in _split_values(value):
        try:
            items.append(int(item))
        except ValueError:
            continue
    return items


def _truncate_text(text: Any, length: int = 80) -> str:
    value = str(text or "").strip()
    if len(value) <= length:
        return value
    return value[: length - 3] + "..."


def resolve_relevant_ids(query_row: pd.Series, metadata: pd.DataFrame) -> set[str]:
    explicit_ids = set(_split_values(query_row.get("relevant_id", "")))
    explicit_ids.update(_split_values(query_row.get("relevant_ids", "")))

    explicit_file_names = set(_split_values(query_row.get("relevant_file_name", "")))
    explicit_file_names.update(_split_values(query_row.get("relevant_file_names", "")))
    if explicit_file_names:
        matched_ids = metadata.loc[
            metadata["file_name"].fillna("").astype(str).isin(explicit_file_names),
            "id",
        ].fillna("").astype(str)
        explicit_ids.update(set(matched_ids))

    if explicit_ids:
        return explicit_ids

    filtered = metadata.copy()
    target_category = str(query_row.get("target_category", "")).strip()
    target_source_type = str(query_row.get("target_source_type", "")).strip()
    if target_category:
        filtered = filtered.loc[filtered["category"].astype(str) == target_category]
    if target_source_type:
        filtered = filtered.loc[filtered["source_type"].astype(str) == target_source_type]
    return set(filtered["id"].fillna("").astype(str))


def evaluation_definition_text(queryset: pd.DataFrame) -> str:
    def _has_values(columns: list[str]) -> bool:
        for column in columns:
            if column not in queryset.columns:
                continue
            series = queryset[column].fillna("").astype(str).str.strip()
            if series.ne("").any():
                return True
        return False

    parts: list[str] = []
    if _has_values(["relevant_id", "relevant_ids"]):
        parts.append("relevant_id/relevant_ids")
    if _has_values(["relevant_file_name", "relevant_file_names"]):
        parts.append("relevant_file_name")
    if _has_values(["target_category"]):
        parts.append("category")
    if _has_values(["target_source_type"]):
        parts.append("source_type")
    if _has_values(["relevant_line_number", "relevant_line_numbers"]):
        parts.append("line")
    if _has_values(["relevant_segment_index", "relevant_segment_indexes"]):
        parts.append("segment_index")
    if _has_values(["relevant_segment_text", "relevant_segment_texts"]):
        parts.append("segment_text")
    return " + ".join(parts) if parts else "metadata id"


def normalize_ground_truth_queryset(queryset: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    if queryset.empty:
        return pd.DataFrame(columns=GROUND_TRUTH_COLUMNS)

    rows: list[dict[str, Any]] = []
    metadata_lookup = (
        metadata[["id", "file_name"]]
        .fillna("")
        .astype(str)
        .drop_duplicates(subset=["id"])
        .set_index("id")["file_name"]
        .to_dict()
    )

    for index, query_row in queryset.iterrows():
        query_id = str(query_row.get("query_id", "")).strip() or f"query_{index + 1:03d}"
        query = str(query_row.get("query", "")).strip()
        if not query:
            continue

        relevant_ids = resolve_relevant_ids(query_row, metadata)
        relevant_file_names = sorted(
            set(_split_values(query_row.get("relevant_file_name", "")))
            | set(_split_values(query_row.get("relevant_file_names", "")))
            | {metadata_lookup.get(doc_id, "") for doc_id in relevant_ids if metadata_lookup.get(doc_id, "")}
        )
        line_numbers = sorted(
            set(_split_int_values(query_row.get("relevant_line_number", "")))
            | set(_split_int_values(query_row.get("relevant_line_numbers", "")))
        )
        segment_indexes = sorted(
            set(_split_int_values(query_row.get("relevant_segment_index", "")))
            | set(_split_int_values(query_row.get("relevant_segment_indexes", "")))
        )
        segment_texts = sorted(
            set(_split_values(query_row.get("relevant_segment_text", "")))
            | set(_split_values(query_row.get("relevant_segment_texts", "")))
        )
        evaluation_level = "segment" if line_numbers or segment_indexes or segment_texts else "file"

        rule_parts: list[str] = []
        if _split_values(query_row.get("relevant_id", "")) or _split_values(query_row.get("relevant_ids", "")):
            rule_parts.append("relevant_id")
        if _split_values(query_row.get("relevant_file_name", "")) or _split_values(query_row.get("relevant_file_names", "")):
            rule_parts.append("relevant_file_name")
        if str(query_row.get("target_category", "")).strip():
            rule_parts.append("category")
        if str(query_row.get("target_source_type", "")).strip():
            rule_parts.append("source_type")
        if line_numbers:
            rule_parts.append("line")
        if segment_indexes:
            rule_parts.append("segment_index")
        if segment_texts:
            rule_parts.append("segment_text")

        rows.append(
            {
                "query_id": query_id,
                "query": query,
                "query_preview": _truncate_text(query),
                "target_category": str(query_row.get("target_category", "")).strip(),
                "target_source_type": str(query_row.get("target_source_type", "")).strip(),
                "relevant_ids": ", ".join(sorted(relevant_ids)),
                "relevant_file_names": ", ".join(relevant_file_names),
                "relevant_line_numbers": ", ".join(map(str, line_numbers)),
                "relevant_segment_indexes": ", ".join(map(str, segment_indexes)),
                "relevant_segment_texts": " | ".join(segment_texts),
                "relevant_count": len(relevant_ids),
                "evaluation_level": evaluation_level,
                "ground_truth_rule": " + ".join(rule_parts) if rule_parts else "derived_from_metadata",
            }
        )

    return pd.DataFrame(rows, columns=GROUND_TRUTH_COLUMNS)


def build_incremental_probe_queryset(metadata: pd.DataFrame, target_ids: set[str]) -> pd.DataFrame:
    columns = ["query_id", "query", "target_category", "relevant_id", "target_source_type"]
    if metadata.empty or not target_ids:
        return pd.DataFrame(columns=columns)

    selected = metadata.loc[metadata["id"].astype(str).isin(target_ids)].reset_index(drop=True)
    probe_rows = []
    for row in selected.itertuples(index=False):
        keywords = " ".join(_split_values(getattr(row, "keywords", ""))[:6]).strip()
        transcript = str(getattr(row, "stt_transcript", "") or getattr(row, "original_transcript", "")).strip()
        snippet = " ".join(transcript.split()[:12]).strip()
        query = " ".join(
            part for part in [str(getattr(row, "title", "")).strip(), keywords, snippet] if part
        ).strip()
        if not query:
            query = str(getattr(row, "file_name", "")).strip() or str(getattr(row, "id", "")).strip()
        probe_rows.append(
            {
                "query_id": f"probe_{str(getattr(row, 'id', '')).lower()}",
                "query": query,
                "target_category": str(getattr(row, "category", "")).strip(),
                "relevant_id": str(getattr(row, "id", "")).strip(),
                "target_source_type": str(getattr(row, "source_type", "")).strip(),
            }
        )
    return pd.DataFrame(probe_rows, columns=columns)
