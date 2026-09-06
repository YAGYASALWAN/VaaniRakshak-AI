"""Reproducible JSON/CSV reports and optional headless matplotlib plots."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .inspection import balance_table, summarize


def write_json(path: Path, value: Any) -> None:
    """Write sorted strict JSON; NaN and infinity are rejected."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def write_reports(frame: pd.DataFrame, output: Path, *, bias: dict | None = None,
                  plots: bool = False) -> dict[str, Any]:
    """Write statistics of supplied source rows; generated outputs belong outside raw data."""
    output.mkdir(parents=True, exist_ok=True)
    summary = summarize(frame)
    write_json(output / "dataset_summary.json", summary)
    for name, fields in {
        "class_balance": ["label"], "language_balance": ["label", "language"],
        "speaker_balance": ["label", "speaker_id"],
        "generator_balance": ["generator"], "language_generator": ["language", "generator"],
    }.items():
        subset = frame[frame.label == "spoof"] if "generator" in fields else frame
        balance_table(subset, fields).to_csv(output / f"{name}.csv", index=False)
    if bias is not None:
        write_json(output / "bias_report.json", bias)
    if plots:
        plot_distributions(frame, output / "plots")
    return summary


def plot_distributions(frame: pd.DataFrame, output: Path) -> None:
    """Plot observed durations/hours; unknown metadata stays explicitly excluded."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    known = pd.to_numeric(frame.duration).dropna()
    bins = np.histogram_bin_edges(known, bins=20) if len(known) else np.linspace(0, 1, 21)
    for label, group in frame.groupby("label", sort=True):
        ax.hist(pd.to_numeric(group.duration).dropna(), bins=bins, alpha=0.55, label=label)
    ax.set(xlabel="Original clip duration (seconds)", ylabel="Clips",
           title="Observed durations — supplied metadata only")
    if len(frame):
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / "duration_histogram.png", dpi=140)
    plt.close(fig)
    for name, fields, metric in (
        ("hours_by_class", ["label"], "hours"),
        ("hours_by_language", ["label", "language"], "hours"),
        ("speakers_by_language", ["label", "language"], "unique_speakers"),
        ("hours_by_generator", ["generator"], "hours"),
    ):
        subset = frame[frame.label == "spoof"] if "generator" in fields else frame
        table = balance_table(subset, fields)
        fig, ax = plt.subplots(figsize=(8, max(3, len(table) * 0.25)))
        labels = table[fields].fillna("unknown").astype(str).agg(" / ".join, axis=1)
        values = table[metric]
        ax.barh(labels[values.notna()], values.dropna())
        ax.set(xlabel=metric.replace("_", " "), title=name.replace("_", " "))
        fig.tight_layout()
        fig.savefig(output / f"{name}.png", dpi=140)
        plt.close(fig)
