"""Report charts, rendered in light and dark variants for the README."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from vlm_inspect.eval.report import METHOD_NAMES

# Reference data-viz palette: first three categorical slots, validated per mode.
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "muted": "#52514e",
        "grid": "#e4e3df",
        "series": ["#2a78d6", "#eb6834", "#1baf7a"],
    },
    "dark": {
        "surface": "#1a1a19",
        "text": "#ffffff",
        "muted": "#c3c2b7",
        "grid": "#3a3937",
        "series": ["#3987e5", "#d95926", "#199e70"],
    },
}


def _figure(theme: dict[str, Any], size: tuple[float, float]) -> tuple[Any, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=size, dpi=150)
    fig.patch.set_facecolor(theme["surface"])
    ax.set_facecolor(theme["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["grid"])
    ax.tick_params(colors=theme["muted"], labelsize=9)
    ax.set_axisbelow(True)
    return fig, ax


def _finish(
    fig: Any,
    ax: Any,
    theme: dict[str, Any],
    *,
    title: str,
    subtitle: str,
    path: Path,
    legend_cols: int,
) -> Path:
    import matplotlib.pyplot as plt

    fig.suptitle(
        title, x=0.06, y=0.97, ha="left", fontsize=12, fontweight="bold", color=theme["text"]
    )
    fig.text(0.06, 0.905, subtitle, ha="left", va="top", fontsize=8.5, color=theme["muted"])
    legend = ax.legend(
        frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=legend_cols
    )
    for text in legend.get_texts():
        text.set_color(theme["text"])
    fig.tight_layout(rect=(0, 0, 1, 0.85))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=theme["surface"])
    plt.close(fig)
    return path


def auroc_by_category(data: dict[str, Any], out: Path) -> list[Path]:
    order: list[str] = data["method_order"]
    methods = data["methods"]
    categories = [c for c in next(iter(methods.values())) if c != "mean"] + ["mean"]
    written = []
    for mode, theme in THEMES.items():
        fig, ax = _figure(theme, (8, 3.8))
        ax.grid(True, axis="y", color=theme["grid"], linewidth=0.8)
        width = 0.8 / len(order)
        for i, (method, color) in enumerate(zip(order, theme["series"], strict=False)):
            xs = [c + (i - (len(order) - 1) / 2) * width for c in range(len(categories))]
            ys = [methods[method].get(cat, {}).get("auroc", math.nan) for cat in categories]
            ax.bar(xs, ys, width=width * 0.92, color=color, label=METHOD_NAMES.get(method, method))
            for x, y in zip(xs, ys, strict=True):
                if not math.isnan(y):
                    ax.annotate(
                        f"{y:.2f}",
                        xy=(x, y),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        fontsize=7.5,
                        color=theme["text"],
                    )
        ax.axhline(0.5, color=theme["muted"], linewidth=1, linestyle=(0, (4, 3)))
        ax.annotate(
            "chance",
            xy=(len(categories) - 0.5, 0.5),
            xytext=(0, 3),
            textcoords="offset points",
            ha="right",
            fontsize=7.5,
            color=theme["muted"],
        )
        ax.set_xticks(range(len(categories)), [c if c != "mean" else "mean" for c in categories])
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("image-level AUROC", color=theme["muted"], fontsize=9)
        written.append(
            _finish(
                fig,
                ax,
                theme,
                title="Defect detection quality by part",
                subtitle="Held-out VisA images, same for every method. "
                "1.0 = perfect ranking, 0.5 = chance.",
                path=out.with_name(f"{out.name}-{mode}.png"),
                legend_cols=len(order),
            )
        )
    return written


def quality_vs_latency(data: dict[str, Any], out: Path) -> list[Path]:
    order = [m for m in data["method_order"] if m in data.get("latency", {})]
    if not order:
        return []
    written = []
    for mode, theme in THEMES.items():
        fig, ax = _figure(theme, (8, 3.8))
        ax.grid(True, color=theme["grid"], linewidth=0.8)
        for method, color in zip(data["method_order"], theme["series"], strict=False):
            if method not in data["latency"]:
                continue
            x = data["latency"][method]["p50_ms"] / 1000
            y = data["methods"][method]["mean"]["auroc"]
            ax.scatter(
                [x],
                [y],
                s=90,
                color=color,
                edgecolor=theme["surface"],
                linewidth=2,
                zorder=3,
                label=METHOD_NAMES.get(method, method),
            )
            ax.annotate(
                f"{y:.2f} AUROC · {x:.2f} s" if x < 1 else f"{y:.2f} AUROC · {x:.1f} s",
                xy=(x, y),
                xytext=(8, -3),
                textcoords="offset points",
                fontsize=8,
                color=theme["text"],
            )
        ax.set_xscale("log")
        ax.set_xlabel(
            "p50 latency per image on a 4-vCPU CPU runner (s, log scale)",
            color=theme["muted"],
            fontsize=9,
        )
        ax.set_ylabel("mean AUROC over parts", color=theme["muted"], fontsize=9)
        ax.margins(x=0.35, y=0.25)
        written.append(
            _finish(
                fig,
                ax,
                theme,
                title="Quality vs cost",
                subtitle="Latency measured for every method on the same machine and images.",
                path=out.with_name(f"{out.name}-{mode}.png"),
                legend_cols=len(order),
            )
        )
    return written


def render_all(data: dict[str, Any], assets_dir: Path) -> list[Path]:
    if not data["method_order"]:
        return []
    return [
        *auroc_by_category(data, assets_dir / "auroc_by_part"),
        *quality_vs_latency(data, assets_dir / "quality_vs_latency"),
    ]
