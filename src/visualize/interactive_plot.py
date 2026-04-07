from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.config import PLOTS_DIR
from src.embedding.build_indices import artifact_stem


def projection_columns(method: str, dimensions: int) -> list[str]:
    if method == "pca":
        return [f"PC{i}" for i in range(1, dimensions + 1)]
    prefix = method.upper()
    return [f"{prefix}{i}" for i in range(1, dimensions + 1)]


def projection_artifact_path(model_alias: str, text_source: str, method: str, dimensions: int) -> Path:
    if method == "pca":
        dimensions = 3
    return PLOTS_DIR / f"{artifact_stem(model_alias, text_source)}_{method}_{dimensions}d_projection.csv"


def reducer_artifact_path(model_alias: str, text_source: str, method: str, dimensions: int) -> Path:
    if method == "pca":
        dimensions = 3
    return PLOTS_DIR / f"{artifact_stem(model_alias, text_source)}_{method}_{dimensions}d_model.joblib"


def load_projection_frame(model_alias: str, text_source: str, method: str, dimensions: int) -> pd.DataFrame:
    return pd.read_csv(projection_artifact_path(model_alias, text_source, method, dimensions))


def load_reducer_if_available(model_alias: str, text_source: str, method: str, dimensions: int):
    path = reducer_artifact_path(model_alias, text_source, method, dimensions)
    if not path.exists():
        return None
    return joblib.load(path)


def project_query_vector(model_alias: str, text_source: str, method: str, dimensions: int, query_vector):
    reducer = load_reducer_if_available(model_alias, text_source, method, dimensions)
    if reducer is None:
        return None
    if method == "pca":
        return reducer.transform([query_vector])[0]
    if method == "umap" and hasattr(reducer, "transform"):
        return reducer.transform([query_vector])[0]
    return None


def build_projection_figure(
    frame: pd.DataFrame,
    method: str,
    dimensions: int,
    color_by: str,
    top_result_ids: list[str],
    query_point=None,
    query_label: str | None = None,
    title: str = "",
):
    columns = projection_columns(method, dimensions)
    plot_frame = frame.copy()
    plot_frame["color_group"] = plot_frame[color_by].astype(str)
    hover_columns = [
        "id",
        "category",
        "title",
        "file_name",
        "preview",
        "cluster_id",
        "normalized_score",
    ]
    hover_columns = [column for column in hover_columns if column in plot_frame.columns]

    if dimensions == 3:
        fig = px.scatter_3d(
            plot_frame,
            x=columns[0],
            y=columns[1],
            z=columns[2],
            color="color_group",
            hover_data=hover_columns,
            title=title,
        )
    else:
        fig = px.scatter(
            plot_frame,
            x=columns[0],
            y=columns[1],
            color="color_group",
            hover_data=hover_columns,
            title=title,
        )

    highlight_frame = plot_frame.loc[plot_frame["id"].isin(top_result_ids)].copy()
    if not highlight_frame.empty:
        highlight_frame["normalized_score"] = highlight_frame["normalized_score"].fillna(0.0).astype(float)
        highlight_customdata = highlight_frame[["title", "normalized_score"]].fillna("").to_numpy()
        if dimensions == 3:
            fig.add_trace(
                go.Scatter3d(
                    x=highlight_frame[columns[0]],
                    y=highlight_frame[columns[1]],
                    z=highlight_frame[columns[2]],
                    mode="markers",
                    name="Top-K Results",
                    marker=dict(size=9, color="black", symbol="diamond"),
                    text=highlight_frame["id"],
                    customdata=highlight_customdata,
                    hovertemplate="id=%{text}<br>title=%{customdata[0]}<br>score=%{customdata[1]:.4f}<extra></extra>",
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=highlight_frame[columns[0]],
                    y=highlight_frame[columns[1]],
                    mode="markers",
                    name="Top-K Results",
                    marker=dict(size=13, color="black", symbol="diamond"),
                    text=highlight_frame["id"],
                    customdata=highlight_customdata,
                    hovertemplate="id=%{text}<br>title=%{customdata[0]}<br>score=%{customdata[1]:.4f}<extra></extra>",
                )
            )

    if query_point is not None:
        if dimensions == 3:
            fig.add_trace(
                go.Scatter3d(
                    x=[query_point[0]],
                    y=[query_point[1]],
                    z=[query_point[2]],
                    mode="markers+text",
                    name="Query",
                    marker=dict(size=12, color="red", symbol="cross"),
                    text=[query_label or "query"],
                    textposition="top center",
                    hovertemplate=f"query={query_label or 'query'}<extra></extra>",
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=[query_point[0]],
                    y=[query_point[1]],
                    mode="markers+text",
                    name="Query",
                    marker=dict(size=14, color="red", symbol="x"),
                    text=[query_label or "query"],
                    textposition="top center",
                    hovertemplate=f"query={query_label or 'query'}<extra></extra>",
                )
            )

    fig.update_layout(height=650, legend_title_text=color_by)
    return fig
