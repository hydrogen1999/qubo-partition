"""Publication-style plotting (matplotlib, headless-safe)."""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update(
    {
        "figure.dpi": 140,
        "savefig.dpi": 200,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.bbox": "tight",
    }
)


def plot_vertex_cover(
    graph: nx.Graph,
    chosen: set,
    optimal: set | None = None,
    title: str = "",
    path: str | None = None,
    seed: int = 0,
):
    """Draw a graph with cover nodes highlighted."""
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    pos = nx.spring_layout(graph, seed=seed)
    colors = ["#d62728" if n in chosen else "#dddddd" for n in graph.nodes()]
    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color="#888888", width=1.2)
    nx.draw_networkx_nodes(
        graph, pos, ax=ax, node_color=colors, edgecolors="#333333", node_size=420
    )
    nx.draw_networkx_labels(graph, pos, ax=ax, font_size=8)
    if optimal is not None:
        # outline the exact-optimum cover for comparison
        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=list(optimal),
            ax=ax,
            node_color="none",
            edgecolors="#1f77b4",
            node_size=560,
            linewidths=2.0,
        )
    ax.set_title(title)
    ax.set_axis_off()
    _save(fig, path)


def plot_segmentation(
    image: np.ndarray,
    fg_seeds: np.ndarray,
    bg_seeds: np.ndarray,
    annealed: np.ndarray,
    optimal: np.ndarray,
    truth: np.ndarray | None = None,
    title: str = "",
    path: str | None = None,
    truth_label: str = "ground truth",
):
    """Show image + seeds, annealed mask, max-flow mask, and (optional) truth."""
    panels = 3 + (truth is not None)
    fig, axes = plt.subplots(1, panels, figsize=(3.0 * panels, 3.2))

    ax = axes[0]
    ax.imshow(image, cmap="gray", vmin=0, vmax=1)
    fr, fc = np.where(fg_seeds)
    br, bc = np.where(bg_seeds)
    ax.scatter(
        fc, fr, c="#2ca02c", s=22, marker="o", label="fg seed", edgecolors="k", linewidths=0.4
    )
    ax.scatter(
        bc, br, c="#d62728", s=22, marker="s", label="bg seed", edgecolors="k", linewidths=0.4
    )
    ax.set_title("image + seeds")
    ax.legend(fontsize=7, loc="upper right")

    axes[1].imshow(annealed, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("annealed mask")

    axes[2].imshow(optimal, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("max-flow optimum")

    if truth is not None:
        axes[3].imshow(truth.astype(float), cmap="gray", vmin=0, vmax=1)
        axes[3].set_title(truth_label)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title)
    _save(fig, path)


def plot_gap_vs_size(
    sizes: Iterable[int],
    best_gaps: Iterable[float],
    mean_gaps: Iterable[float],
    std_gaps: Iterable[float],
    title: str = "Optimality gap vs. instance size",
    path: str | None = None,
):
    """Best gap and mean +/- std gap as a function of node count."""
    sizes = np.asarray(list(sizes))
    best = np.asarray(list(best_gaps))
    mean = np.asarray(list(mean_gaps))
    std = np.asarray(list(std_gaps))

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(sizes, best, "o-", color="#1f77b4", label="best gap")
    ax.plot(sizes, mean, "s--", color="#ff7f0e", label="mean gap")
    ax.fill_between(sizes, mean - std, mean + std, color="#ff7f0e", alpha=0.2, label="+/- 1 std")
    ax.axhline(0.0, color="#333333", lw=0.8)
    ax.set_xlabel("number of nodes")
    ax.set_ylabel("energy gap")
    ax.set_title(title)
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_success_vs_param(
    xs: Iterable[float],
    success: Iterable[float],
    xlabel: str,
    title: str,
    path: str | None = None,
):
    """Success rate (fraction of reads reaching the optimum) vs. a parameter."""
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.plot(list(xs), list(success), "o-", color="#2ca02c")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("success rate")
    ax.set_title(title)
    _save(fig, path)


def _save(fig, path: str | None):
    if path:
        fig.savefig(path)
        plt.close(fig)
    else:  # pragma: no cover
        return fig
