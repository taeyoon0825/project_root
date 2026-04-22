from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import ADAPTIVE_DIR, EVALUATION_DIR


RECENT_PERFORMANCE_PROFILE_JSON = ADAPTIVE_DIR / "recent_performance_profile.json"


@dataclass
class BucketPerformance:
    query_count: int = 0
    semantic_success_rate: float = 0.0
    mrr: float = 0.0
    ndcg: float = 0.0
    topk_hit_rate: float = 0.0
    delta_semantic_success_rate: float = 0.0
    delta_mrr: float = 0.0
    delta_ndcg: float = 0.0
    delta_topk_hit_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PerformanceProfile:
    buckets: dict[str, BucketPerformance] = field(default_factory=dict)
    recent_summary_files: list[str] = field(default_factory=list)
    recent_delta_files: list[str] = field(default_factory=list)
    reranker_value_prior: float = 0.0
    overall_semantic_success_rate: float = 0.0
    overall_semantic_gain: float = 0.0
    used_fallback_stats: bool = False
    fallback_reason: str = ""

    def bucket(self, name: str) -> BucketPerformance:
        return self.buckets.get(name, BucketPerformance())

    def to_dict(self) -> dict[str, Any]:
        return {
            "buckets": {key: value.to_dict() for key, value in self.buckets.items()},
            "recent_summary_files": self.recent_summary_files,
            "recent_delta_files": self.recent_delta_files,
            "reranker_value_prior": self.reranker_value_prior,
            "overall_semantic_success_rate": self.overall_semantic_success_rate,
            "overall_semantic_gain": self.overall_semantic_gain,
            "used_fallback_stats": self.used_fallback_stats,
            "fallback_reason": self.fallback_reason,
        }


def _bucket_from_dict(payload: dict[str, Any]) -> BucketPerformance:
    return BucketPerformance(
        query_count=int(payload.get("query_count", 0) or 0),
        semantic_success_rate=float(payload.get("semantic_success_rate", 0.0) or 0.0),
        mrr=float(payload.get("mrr", 0.0) or 0.0),
        ndcg=float(payload.get("ndcg", 0.0) or 0.0),
        topk_hit_rate=float(payload.get("topk_hit_rate", 0.0) or 0.0),
        delta_semantic_success_rate=float(payload.get("delta_semantic_success_rate", 0.0) or 0.0),
        delta_mrr=float(payload.get("delta_mrr", 0.0) or 0.0),
        delta_ndcg=float(payload.get("delta_ndcg", 0.0) or 0.0),
        delta_topk_hit_rate=float(payload.get("delta_topk_hit_rate", 0.0) or 0.0),
    )


def _profile_from_dict(payload: dict[str, Any]) -> PerformanceProfile:
    buckets = {
        str(name): _bucket_from_dict(bucket_payload if isinstance(bucket_payload, dict) else {})
        for name, bucket_payload in dict(payload.get("buckets", {})).items()
    }
    return PerformanceProfile(
        buckets=buckets,
        recent_summary_files=[str(path) for path in list(payload.get("recent_summary_files", []))],
        recent_delta_files=[str(path) for path in list(payload.get("recent_delta_files", []))],
        reranker_value_prior=float(payload.get("reranker_value_prior", 0.0) or 0.0),
        overall_semantic_success_rate=float(payload.get("overall_semantic_success_rate", 0.0) or 0.0),
        overall_semantic_gain=float(payload.get("overall_semantic_gain", 0.0) or 0.0),
        used_fallback_stats=bool(payload.get("used_fallback_stats", False)),
        fallback_reason=str(payload.get("fallback_reason", "") or ""),
    )


def _recent_files(pattern: str, limit: int = 6) -> list[Path]:
    files = sorted(EVALUATION_DIR.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    return files[:limit]


def _load_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path).fillna("")
    except Exception:
        return pd.DataFrame()


def _weighted_mean(values: list[tuple[float, float]]) -> float:
    if not values:
        return 0.0
    total_weight = sum(weight for _, weight in values)
    if total_weight <= 0:
        return 0.0
    return float(sum(value * weight for value, weight in values) / total_weight)


def _build_profile_from_bucket_rows(
    bucket_rows: dict[str, dict[str, list[tuple[float, float]]]],
    recent_summary_paths: list[str],
    recent_delta_paths: list[str],
) -> PerformanceProfile:
    if not bucket_rows:
        return PerformanceProfile(used_fallback_stats=True, fallback_reason="missing_recent_semantic_evaluation")

    buckets: dict[str, BucketPerformance] = {}
    for bucket, metrics in bucket_rows.items():
        buckets[bucket] = BucketPerformance(
            query_count=int(round(_weighted_mean(metrics.get("query_count", [])))),
            semantic_success_rate=_weighted_mean(metrics.get("semantic_success_rate", [])),
            mrr=_weighted_mean(metrics.get("mrr", [])),
            ndcg=_weighted_mean(metrics.get("ndcg", [])),
            topk_hit_rate=_weighted_mean(metrics.get("topk_hit_rate", [])),
            delta_semantic_success_rate=_weighted_mean(metrics.get("delta_semantic_success_rate", [])),
            delta_mrr=_weighted_mean(metrics.get("delta_mrr", [])),
            delta_ndcg=_weighted_mean(metrics.get("delta_ndcg", [])),
            delta_topk_hit_rate=_weighted_mean(metrics.get("delta_topk_hit_rate", [])),
        )

    semantic_buckets = [name for name in buckets if name not in {"overall", "exact_keyword"}]
    value_terms: list[float] = []
    for name in semantic_buckets:
        bucket = buckets[name]
        value_terms.append(
            float(
                np.mean(
                    [
                        max(0.0, bucket.delta_semantic_success_rate),
                        max(0.0, bucket.delta_mrr),
                        max(0.0, bucket.delta_ndcg),
                        1.0 - bucket.semantic_success_rate,
                    ]
                )
            )
        )
    overall = buckets.get("overall", BucketPerformance())
    return PerformanceProfile(
        buckets=buckets,
        recent_summary_files=recent_summary_paths,
        recent_delta_files=recent_delta_paths,
        reranker_value_prior=float(np.mean(value_terms)) if value_terms else 0.0,
        overall_semantic_success_rate=overall.semantic_success_rate,
        overall_semantic_gain=overall.delta_semantic_success_rate,
        used_fallback_stats=False,
        fallback_reason="",
    )


def _build_profile_from_semantic_artifacts(summary_files: list[Path], delta_files: list[Path]) -> PerformanceProfile:
    if not summary_files and not delta_files:
        return PerformanceProfile(used_fallback_stats=True, fallback_reason="missing_recent_semantic_evaluation")

    bucket_rows: dict[str, dict[str, list[tuple[float, float]]]] = {}
    recent_summary_paths: list[str] = []
    recent_delta_paths: list[str] = []

    for rank, path in enumerate(summary_files, start=1):
        frame = _load_csv(path)
        if frame.empty or "query_type" not in frame.columns or "system" not in frame.columns:
            continue
        recent_summary_paths.append(str(path))
        weight_multiplier = 1.0 / rank
        after_frame = frame.loc[frame["system"].astype(str).isin(["fused_after", "fused"])]
        for row in after_frame.to_dict(orient="records"):
            bucket = str(row.get("query_type", "")).strip() or "overall"
            query_count = float(row.get("query_count", 0) or 0)
            weight = max(1.0, query_count) * weight_multiplier
            bucket_rows.setdefault(bucket, {}).setdefault("query_count", []).append((query_count, weight))
            bucket_rows[bucket].setdefault("semantic_success_rate", []).append(
                (float(row.get("semantic_success_rate", 0.0) or 0.0), weight)
            )
            bucket_rows[bucket].setdefault("mrr", []).append((float(row.get("mrr", 0.0) or 0.0), weight))
            bucket_rows[bucket].setdefault("ndcg", []).append((float(row.get("ndcg", 0.0) or 0.0), weight))
            bucket_rows[bucket].setdefault("topk_hit_rate", []).append((float(row.get("topk_hit_rate", 0.0) or 0.0), weight))

    for rank, path in enumerate(delta_files, start=1):
        frame = _load_csv(path)
        if frame.empty or "query_type" not in frame.columns:
            continue
        recent_delta_paths.append(str(path))
        weight_multiplier = 1.0 / rank
        for row in frame.to_dict(orient="records"):
            bucket = str(row.get("query_type", "")).strip() or "overall"
            query_count = max(
                float(row.get("after_query_count", 0) or 0.0),
                float(row.get("before_query_count", 0) or 0.0),
            )
            weight = max(1.0, query_count) * weight_multiplier
            for metric in ["semantic_success_rate", "mrr", "ndcg", "topk_hit_rate"]:
                bucket_rows.setdefault(bucket, {}).setdefault(f"delta_{metric}", []).append(
                    (float(row.get(f"delta_{metric}", 0.0) or 0.0), weight)
                )

    return _build_profile_from_bucket_rows(bucket_rows, recent_summary_paths, recent_delta_paths)


def _build_profile_from_runtime_evaluation(summary_files: list[Path], delta_files: list[Path]) -> PerformanceProfile:
    if not summary_files and not delta_files:
        return PerformanceProfile(used_fallback_stats=True, fallback_reason="missing_runtime_evaluation")

    bucket_rows: dict[str, dict[str, list[tuple[float, float]]]] = {}
    recent_summary_paths: list[str] = []
    recent_delta_paths: list[str] = []

    for rank, path in enumerate(summary_files, start=1):
        frame = _load_csv(path)
        if frame.empty or "system_name" not in frame.columns or "parameter_mode" not in frame.columns:
            continue
        fused = frame.loc[
            (frame["system_name"].astype(str) == "fused-retrieval")
            & (frame["parameter_mode"].astype(str) == "adaptive")
        ]
        if fused.empty:
            continue
        recent_summary_paths.append(str(path))
        weight_multiplier = 1.0 / rank
        for row in fused.to_dict(orient="records"):
            query_count = float(row.get("query_count", 0.0) or 0.0)
            weight = max(1.0, query_count) * weight_multiplier
            bucket_rows.setdefault("overall", {}).setdefault("query_count", []).append((query_count, weight))
            bucket_rows["overall"].setdefault("semantic_success_rate", []).append(
                (float(row.get("accuracy_at_1", 0.0) or 0.0), weight)
            )
            bucket_rows["overall"].setdefault("mrr", []).append((float(row.get("mrr_at_k", 0.0) or 0.0), weight))
            bucket_rows["overall"].setdefault("ndcg", []).append((float(row.get("ndcg_at_k", 0.0) or 0.0), weight))
            bucket_rows["overall"].setdefault("topk_hit_rate", []).append(
                (float(row.get("topk_hit_rate", 0.0) or 0.0), weight)
            )

    for rank, path in enumerate(delta_files, start=1):
        frame = _load_csv(path)
        if frame.empty or "system_name" not in frame.columns:
            continue
        fused = frame.loc[frame["system_name"].astype(str) == "fused-retrieval"]
        if fused.empty:
            continue
        recent_delta_paths.append(str(path))
        weight_multiplier = 1.0 / rank
        for row in fused.to_dict(orient="records"):
            query_count = max(
                float(row.get("top_k_before", 0.0) or 0.0),
                float(row.get("top_k_after", 0.0) or 0.0),
                1.0,
            )
            weight = query_count * weight_multiplier
            bucket_rows.setdefault("overall", {}).setdefault("delta_semantic_success_rate", []).append(
                (float(row.get("accuracy_at_1_delta", 0.0) or 0.0), weight)
            )
            bucket_rows["overall"].setdefault("delta_mrr", []).append(
                (float(row.get("mrr_at_k_delta", 0.0) or 0.0), weight)
            )
            bucket_rows["overall"].setdefault("delta_ndcg", []).append(
                (float(row.get("ndcg_at_k_delta", 0.0) or 0.0), weight)
            )
            bucket_rows["overall"].setdefault("delta_topk_hit_rate", []).append(
                (float(row.get("recall_at_k_delta", row.get("topk_hit_rate_delta", 0.0)) or 0.0), weight)
            )

    profile = _build_profile_from_bucket_rows(bucket_rows, recent_summary_paths, recent_delta_paths)
    if profile.used_fallback_stats:
        profile.fallback_reason = "missing_runtime_fused_rows"
    return profile


def save_recent_performance_profile(profile: PerformanceProfile, path: Path = RECENT_PERFORMANCE_PROFILE_JSON) -> Path:
    ADAPTIVE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _load_saved_profile(path: Path = RECENT_PERFORMANCE_PROFILE_JSON) -> PerformanceProfile | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return _profile_from_dict(payload)


def load_recent_performance_profile() -> PerformanceProfile:
    saved = _load_saved_profile()
    if saved is not None:
        return saved

    summary_files = _recent_files("*semantic*summary.csv")
    delta_files = _recent_files("*semantic*delta.csv")
    profile = _build_profile_from_semantic_artifacts(summary_files, delta_files)
    if not profile.used_fallback_stats:
        save_recent_performance_profile(profile)
        return profile

    runtime_profile = _build_profile_from_runtime_evaluation(
        _recent_files("*retrieval_eval_summary.csv"),
        _recent_files("*retrieval_eval_mode_comparison.csv"),
    )
    if not runtime_profile.used_fallback_stats:
        save_recent_performance_profile(runtime_profile)
        return runtime_profile

    return profile
