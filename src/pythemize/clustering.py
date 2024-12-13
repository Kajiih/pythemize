"""
Tools for clustering colors used in themes.

TODO: Check how to handle nans. -> Currently transformed to 0 when getting color coordinate
    -> better to just drop ?

Todo:
    - Test grouping with different algorithms.
    - Test grouping by taking lightness or lightness + saturation into account
        -> For now only exactly 2 channels is supported
    - Test different color spaces
    - Try grouping using several themes (by using more dimensions)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, LiteralString

import attrs
import matplotlib.pyplot as plt
from attrs import frozen
from coloraide import Color
from coloraide.spaces.okhsl import Okhsl
from matplotlib.lines import Line2D
from nested_dict_tools import flatten_dict
from sklearn.cluster import DBSCAN, KMeans  #   # type stubs issue
from sklearn.cluster._hdbscan.hdbscan import (  # pyright: ignore[reportMissingTypeStubs]
    HDBSCAN,  # noqa: PLC2701
)

from pythemize.plot import DARK_BACKGROUND_COLOR, LIGHT_BACKGROUND_COLOR, PlotColors
from pythemize.utils import load_theme_colors

if TYPE_CHECKING:
    from collections.abc import Iterable

    from matplotlib.axes import Axes
    from numpy import float64, int_
    from numpy.typing import NDArray

Color.register(Okhsl())
"""Points representing colors to cluster."""

# Make a dict mapping from colors to theme element that have these colors because we want every color to appear only once

type SupportedClusterer = KMeans | DBSCAN | HDBSCAN


@frozen(kw_only=True)
class ClusterData:
    """Data returned by fitting a color cluster."""

    labels: NDArray[int_]  # shape (nb_samples,)
    cluster_centers: NDArray[float64]  # shape (nb_clusters, nb_features)

    @classmethod
    def from_fitted_clusterer(cls, clusterer: SupportedClusterer) -> ClusterData:
        """Initialize from a fitted clusterer instance."""
        match clusterer:
            case KMeans():
                cluster_centers = clusterer.cluster_centers_
            case DBSCAN():
                cluster_centers = clusterer.components_
            case HDBSCAN():
                cluster_centers = clusterer.centroids_

        labels: NDArray[int_] = clusterer.labels_  # pyright: ignore[reportAssignmentType]
        return cls(labels=labels, cluster_centers=cluster_centers)


# TODO: Add support for hct color space
@attrs.frozen(kw_only=True)
class ColorClusterer:
    """Base class for color theme clusterer."""


    space: str = "okhsl"
    # cluster_channels: tuple[int, int] | tuple[int] = (0, 1)  # hue (h) and saturation (s)
    cluster_channels: tuple[str, str] | tuple[str] = ("h", "s")  # hue (h) and saturation (s)

    def extract_color_points(self, colors: Iterable[Color]) -> list[tuple[float, ...]]:
        """Extract color data points from color instances."""
        return [
            tuple(
                color.convert(space=self.space).get(channel, nans=False)
                for channel in self.cluster_channels
            )
            for color in colors
        ]

    def fit[Clusterer: SupportedClusterer](
        self, colors: Iterable[Color], clusterer: Clusterer
    ) -> Clusterer:
        """Fit the cluster with the colors' data."""
        color_points = self.extract_color_points(colors=colors)

        clusterer.fit(X=color_points)

        return clusterer

    def fit_predict(self, colors: Iterable[Color], clusterer: SupportedClusterer) -> ClusterData:
        """Fit the clusters with the colors' data and predict cluster data."""
        clusterer = self.fit(colors=colors, clusterer=clusterer)

        return ClusterData.from_fitted_clusterer(clusterer=clusterer)

    def get_clusters_figure(
        self,
        original_colors: Iterable[Color],  # Supposed to be in self.space color space
        cluster_data: ClusterData,
        cluster_color_map: Literal["centroid_center"] | LiteralString = "centroid_center",
        background_color: Color = DARK_BACKGROUND_COLOR,
        ax: Axes | None = None,
    ) -> Axes:
        """Create a plot of the color points on a hue/saturation space."""
        cluster_centers = cluster_data.cluster_centers

        if cluster_color_map == "centroid_center":
            label_colors = [
                Color(self.space, [h, s, 0.5], 1).convert("srgb").to_string(hex=True)
                for h, s in cluster_centers
            ]
        else:
            cmap = plt.get_cmap(cluster_color_map)
            label_colors = [cmap(i) for i in range(len(cluster_centers))]

        color_points_hex = [color.convert("srgb").to_string(hex=True) for color in original_colors]
        cluster_colors = [label_colors[label] for label in cluster_data.labels]

        # === Scatter ===

        # Create figure and axes if not provided
        if ax is None:
            _, ax = plt.subplots()

        plot_colors = PlotColors.from_background_color(background_color)
        background_hex_color = background_color.to_string(hex=True)
        lines_hex_color = plot_colors.line.to_string(hex=True)

        ax.set_title(f"Clusters in {self.space} space")
        ax.set_xlabel("Hue")
        ax.set_ylabel("Saturation")
        ax.set_facecolor(background_hex_color)
        # Color points and clusters
        cluster_marker = "o"
        ax.scatter(
            [color.get("hue") for color in original_colors],
            [color.get("saturation") for color in original_colors],
            c=cluster_colors,
            edgecolors=color_points_hex,
            s=100,
            marker=cluster_marker,
            alpha=1,
            linewidths=2,
        )

        # Centroids
        centroid_marker = "X"
        ax.scatter(
            [center[0] for center in cluster_centers],
            [center[1] for center in cluster_centers],
            c=label_colors[: len(cluster_centers)],
            s=150,
            marker=centroid_marker,
            edgecolors=lines_hex_color,
            linewidths=1.5,
        )

        # === Legend ===
        legend_forground_hex_color = plt.rcParams["text.color"]
        legend_handles = [
            # Color points
            Line2D(
                [0],
                [0],
                marker=cluster_marker,
                markerfacecolor="#00000000",
                markeredgecolor=legend_forground_hex_color,
                markeredgewidth=2,
                markersize=10,
                label="Colors",
                linestyle="None",
            ),
            # Clusters
            Line2D(
                [0],
                [0],
                marker=cluster_marker,
                markerfacecolor=legend_forground_hex_color,
                markeredgecolor="#00000000",
                markersize=8,
                label="Clusters",
                linestyle="None",
            ),
            # Centroids
            Line2D(
                [0],
                [0],
                marker=centroid_marker,
                markerfacecolor="#00000000",
                markeredgecolor=legend_forground_hex_color,
                markeredgewidth=1.5,
                markersize=11,
                label="Centroids",
                linestyle="None",
            ),
        ]

        ax.legend(handles=legend_handles, title="Clusters", loc="best")

        ax.grid(visible=False)

        return ax


# === Testing ===
plt.style.use("dark_background")

themes_dark_path = Path("./reference_themes/dark")
# themes_dark_path = Path("../../reference_themes/dark")
ref_themes_dark = {
    "empty": "empty-theme",
    "arcane": "arcane-color-theme",
    "blueberry": "bearded-theme-surprising-blueberry",
    "cpp": "cpptools_dark_vs_new-color-theme",
    "dark_modern": "dark_modern",
}
space = "okhsl"

selected_channel = ("h", "s")
space_class = Color.CS_MAP[space]

range_k = (4, 11)  # (included, excluded)

# Remove Nones
ref_themes = {
    name: load_theme_colors(themes_dark_path / (ref_themes_dark[name] + ".json"), space)
    for name in ref_themes_dark
}

ref_theme = ref_themes["blueberry"]
flat_theme = flatten_dict(ref_theme["colors"], sep="/")

theme_colors = list(flat_theme.values())

# === Kmeans ===
k = 9
clusterer = KMeans(n_clusters=k)


color_clusterer = ColorClusterer()
cluster_data = color_clusterer.fit_predict(colors=theme_colors, clusterer=clusterer)

# === Plot ===
axes: list[Axes]
FIG_SIZE = 5
fig, axes = plt.subplots(1, 2, figsize=(2 * FIG_SIZE, FIG_SIZE))

ax0 = color_clusterer.get_clusters_figure(
    original_colors=flat_theme.values(),
    cluster_data=cluster_data,
    background_color=LIGHT_BACKGROUND_COLOR,
    # cluster_color_map="tab10",
    ax=axes[0],
)

ax1 = color_clusterer.get_clusters_figure(
    original_colors=flat_theme.values(),
    cluster_data=cluster_data,
    # cluster_color_map="tab10",
    ax=axes[1],
)
plt.show()
