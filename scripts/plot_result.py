#!/usr/bin/env python3
import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="matplotlib-"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

BASE = "#1e1e2e"
MANTLE = "#181825"
SURFACE0 = "#313244"
SURFACE1 = "#45475a"
TEXT = "#cdd6f4"
SUBTEXT0 = "#a6adc8"
BLUE = "#89b4fa"
GREEN = "#a6e3a1"
YELLOW = "#f9e2af"
PEACH = "#fab387"
RED = "#f38ba8"
LAVENDER = "#b4befe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize TransNetV2 result.json predictions."
    )
    parser.add_argument(
        "--input",
        default="~/Downloads/result.json",
        help="Path to result.json",
    )
    parser.add_argument(
        "--output",
        default="~/Downloads/result_plot.png",
        help="Path to output PNG",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Reference threshold line for single_frame_predictions",
    )
    parser.add_argument(
        "--zoom-start",
        type=int,
        default=None,
        help="Optional start frame for a zoomed subplot",
    )
    parser.add_argument(
        "--zoom-end",
        type=int,
        default=None,
        help="Optional end frame for a zoomed subplot",
    )
    return parser.parse_args()


def load_result(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def scene_boundaries(scenes: list[list[int]]) -> list[int]:
    return [scene[0] for scene in scenes[1:]]


def extract_scenes(data: dict[str, Any]) -> list[list[int]]:
    scenes = data.get("scenes")
    if isinstance(scenes, list) and scenes:
        return scenes

    preview_frames = data.get("scene_preview_frames", [])
    extracted_scenes = []
    for scene in preview_frames:
        start_frame = scene.get("start_frame")
        end_frame = scene.get("end_frame")
        if isinstance(start_frame, int) and isinstance(end_frame, int):
            extracted_scenes.append([start_frame, end_frame])
    return extracted_scenes


def add_boundaries(ax: plt.Axes, boundaries: list[int]) -> None:
    for boundary in boundaries:
        ax.axvline(boundary, color=SUBTEXT0, linestyle=":", linewidth=0.8, alpha=0.35)


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(MANTLE)
    for spine in ax.spines.values():
        spine.set_color(SURFACE1)
    ax.tick_params(colors=SUBTEXT0)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)


def plot_series(
    ax: plt.Axes,
    x: np.ndarray,
    single: np.ndarray,
    many: np.ndarray,
    threshold: float,
    title: str,
) -> None:
    ax.plot(x[: len(single)], single, label="single_frame_predictions", lw=1.2, color=BLUE)
    ax.plot(x[: len(many)], many, label="all_frame_predictions", lw=1.2, alpha=0.9, color=PEACH)
    ax.axhline(
        threshold,
        color=RED,
        linestyle="--",
        linewidth=1,
        label=f"threshold={threshold}",
    )
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("score")
    ax.set_title(title)
    ax.grid(color=SURFACE0, alpha=0.45)
    style_axis(ax)


def plot_predictions(data: dict[str, Any], output_path: Path, args: argparse.Namespace) -> None:
    frame_count = int(data.get("frame_count", 0))
    scenes = extract_scenes(data)
    single = np.asarray(data.get("single_frame_predictions", []), dtype=float)
    many = np.asarray(data.get("all_frame_predictions", []), dtype=float)
    boundaries = scene_boundaries(scenes)
    x = np.arange(frame_count)

    has_zoom = args.zoom_start is not None and args.zoom_end is not None
    row_count = 3 if has_zoom else 2
    fig, axes = plt.subplots(
        row_count,
        1,
        figsize=(18, 10 if has_zoom else 7.5),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3, 2, 1.1] if has_zoom else [3, 1.1]},
    )
    fig.patch.set_facecolor(BASE)

    ax_main = axes[0]
    plot_series(
        ax_main,
        x,
        single,
        many,
        args.threshold,
        (
            f"TransNetV2 Predictions | task_id={data.get('task_id', '-')}, "
            f"frames={frame_count}, scenes={len(scenes)}"
        ),
    )
    add_boundaries(ax_main, boundaries)
    ax_main.legend(loc="upper right")

    if has_zoom:
        zoom_start = max(0, args.zoom_start)
        zoom_end = min(frame_count, args.zoom_end)
        zoom_slice = slice(zoom_start, zoom_end)
        zoom_x = np.arange(zoom_start, zoom_end)
        ax_zoom = axes[1]
        plot_series(
            ax_zoom,
            zoom_x,
            single[zoom_slice],
            many[zoom_slice],
            args.threshold,
            f"Zoomed Range | frames {zoom_start}..{zoom_end}",
        )
        add_boundaries(
            ax_zoom,
            [b for b in boundaries if zoom_start <= b <= zoom_end],
        )
        scene_ax = axes[2]
    else:
        scene_ax = axes[1]

    for index, (start, end) in enumerate(scenes):
        color = GREEN if index % 2 == 0 else YELLOW
        scene_ax.axvspan(start, end, color=color, alpha=0.95)

    add_boundaries(scene_ax, boundaries)
    scene_ax.set_xlim(0, frame_count if frame_count > 0 else 1)
    scene_ax.set_yticks([])
    scene_ax.set_ylabel("scene")
    scene_ax.set_title("Scene Segments")
    scene_ax.set_xlabel("frame")
    style_axis(scene_ax)
    scene_ax.grid(color=SURFACE0, alpha=0.2, axis="x")

    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    data = load_result(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_predictions(data, output_path, args)

    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
