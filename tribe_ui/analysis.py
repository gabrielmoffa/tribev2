from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def activation_timeseries(preds_path: Path) -> pd.DataFrame:
    preds = np.load(preds_path)
    return pd.DataFrame(
        {
            "second": np.arange(len(preds), dtype=int),
            "absolute_mean": np.abs(preds).mean(axis=1),
            "std": preds.std(axis=1),
            "positive_peak": preds.max(axis=1),
            "negative_peak": preds.min(axis=1),
        }
    )


def activation_plot(preds_path: Path) -> go.Figure:
    df = activation_timeseries(preds_path)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["second"],
            y=df["absolute_mean"],
            mode="lines+markers",
            name="Overall activation",
            line={"width": 3, "color": "#d43f3a"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["second"],
            y=df["std"],
            mode="lines+markers",
            name="Spatial spread",
            line={"width": 2, "color": "#2a6fbb"},
        )
    )
    fig.update_layout(
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
        height=280,
        xaxis_title="Video second",
        yaxis_title="Predicted response magnitude",
        template="plotly_white",
        legend={"orientation": "h", "y": 1.12},
    )
    return fig


def roi_summary(preds_path: Path, top_k: int = 8) -> pd.DataFrame:
    """Summarize strongest Destrieux cortical parcels for lightweight UI review."""

    from nilearn import datasets

    atlas = datasets.fetch_atlas_surf_destrieux(verbose=0)
    labels = np.array(atlas["labels"])
    map_both = np.concatenate([atlas["map_left"], atlas["map_right"]])
    preds = np.load(preds_path)
    rows = []
    for t, row in enumerate(preds):
        means = []
        abs_means = []
        for idx, name in enumerate(labels):
            if idx == 0:
                continue
            mask = map_both == idx
            if not mask.any():
                continue
            means.append((name, float(row[mask].mean())))
            abs_means.append((name, float(np.abs(row[mask]).mean())))
        for rank, (name, value) in enumerate(
            sorted(means, key=lambda item: item[1], reverse=True)[:top_k],
            start=1,
        ):
            rows.append(
                {
                    "second": t,
                    "rank": rank,
                    "direction": "positive",
                    "region": name,
                    "value": value,
                }
            )
        for rank, (name, value) in enumerate(
            sorted(abs_means, key=lambda item: item[1], reverse=True)[:top_k],
            start=1,
        ):
            rows.append(
                {
                    "second": t,
                    "rank": rank,
                    "direction": "absolute",
                    "region": name,
                    "value": value,
                }
            )
    return pd.DataFrame(rows)


def plain_language_summary(preds_path: Path) -> str:
    df = activation_timeseries(preds_path)
    peak = df.loc[df["absolute_mean"].idxmax()]
    return (
        f"Peak overall predicted cortical response is around second "
        f"{int(peak['second'])}. The chart is a model-derived magnitude summary, "
        "not a clinical or direct emotion measurement."
    )

