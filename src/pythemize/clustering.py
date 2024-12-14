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

import itertools
from math import ceil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, LiteralString
from warnings import deprecated

import attrs
import matplotlib.pyplot as plt
import numpy as np
from attrs import frozen
from coloraide import Color
from coloraide.spaces.okhsl import Okhsl
from matplotlib.lines import Line2D
from nested_dict_tools import flatten_dict
from sklearn.cluster import DBSCAN, KMeans  #   # type stubs issue
from sklearn.cluster._hdbscan.hdbscan import (  # pyright: ignore[reportMissingTypeStubs]
    HDBSCAN,  # noqa: PLC2701
)
from sklearn.preprocessing import (
    MinMaxScaler,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)
from yellowbrick.cluster.elbow import (  # pyright: ignore[reportMissingTypeStubs]
    KElbowVisualizer,
    kelbow_visualizer,
)

from pythemize.plot import DARK_BACKGROUND_COLOR, LIGHT_BACKGROUND_COLOR, PlotColors
from pythemize.utils import load_theme_colors

if TYPE_CHECKING:
    from collections.abc import Iterable

    from matplotlib.axes import Axes
    from numpy import float64, int_
    from numpy.typing import NDArray

Color.register(Okhsl())

# Make a dict mapping from colors to theme element that have these colors because we want every color to appear only once

type SupportedClusterer = KMeans | DBSCAN | HDBSCAN
"""Supported sklearn cluster algorithms."""
type SupportedNormalizer = (
    StandardScaler | RobustScaler | PowerTransformer | QuantileTransformer | MinMaxScaler
)
"""Supported sklearn normalizer."""
# type ColorPoints = list[tuple[float, ...]]
type ColorPoints = NDArray[float64]
"""Points representing colors to cluster in a certain color space."""


@frozen(kw_only=True)
class ClusterData:
    """Data returned by fitting a color cluster."""

    labels: NDArray[int_]  # shape (nb_samples,)
    cluster_centers: NDArray[float64]  # shape (nb_clusters, nb_features)

    @classmethod
    @deprecated("Use from_fitted_color_clusterer")
    def _from_fitted_clusterer_old(cls, clusterer: SupportedClusterer) -> ClusterData:
        """Initialize from a fitted clusterer instance. Cluster centers are not unscaled."""
        cluster_centers = cls.extract_cluster_centers(clusterer)

        labels: NDArray[int_] = clusterer.labels_  # pyright: ignore[reportAssignmentType]
        return cls(labels=labels, cluster_centers=cluster_centers)

    @classmethod
    def from_fitted_color_clusterer(
        cls, color_clusterer: ColorClusterer[SupportedClusterer]
    ) -> ClusterData:
        """Initialize from a fitted color clusterer instance. Cluster_centers are unscaled."""
        cluster_centers = cls.extract_cluster_centers(clusterer)
        cluster_centers = color_clusterer.normalizer_inverse_transform(cluster_centers)

        labels: NDArray[int_] = color_clusterer.clusterer.labels_  # pyright: ignore[reportAssignmentType]
        return cls(labels=labels, cluster_centers=cluster_centers)

    @staticmethod
    def extract_cluster_centers(clusterer: SupportedClusterer) -> NDArray[float64]:
        """Extract what is considered cluster centers from the fitted clusterer."""
        match clusterer:
            case KMeans():
                return clusterer.cluster_centers_
            case DBSCAN():
                return clusterer.components_
            case HDBSCAN():
                return clusterer.centroids_


# TODO: Add support for hct color space
@attrs.frozen(kw_only=True)
class ColorClusterer[Clusterer: SupportedClusterer]:
    """Base class for color theme clusterer."""

    clusterer: Clusterer
    normalizer: SupportedNormalizer | None = None
    space: str = "okhsl"
    # space: str = "oklab"  # "okhsl"
    # cluster_channels: tuple[int, int] | tuple[int] = (0, 1)  # hue (h) and saturation (s)
    cluster_channels: tuple[str, str] | tuple[str] = ("h", "s")  # hue (h) and saturation (s)
    # cluster_channels: tuple[str, str] | tuple[str] = ("a", "b")  # hue (h) and saturation (s)

    def extract_color_points(self, colors: Iterable[Color]) -> ColorPoints:
        """Extract color data points from color instances."""
        return np.asarray([
            tuple(
                color.convert(space=self.space).get(channel, nans=False)
                for channel in self.cluster_channels
            )
            for color in colors
        ])

    def normalize(self, color_points: ColorPoints) -> ColorPoints:
        """Normalize the color_points."""
        if self.normalizer is None:
            return color_points
        self.normalizer.fit(color_points)

        return self.normalizer.transform(color_points)

    def normalizer_inverse_transform(self, X: NDArray[Any]) -> NDArray[Any]:
        """Perform the normalizer's inverse transform on the input."""
        return X if self.normalizer is None else self.normalizer.inverse_transform(X)

    def fit(self, colors: Iterable[Color]) -> None:
        """Normalize and fit the cluster with the colors' data."""
        color_points = self.extract_color_points(colors=colors)
        color_points = self.normalize(color_points)

        self.clusterer.fit(X=color_points)

    def fit_predict(self, colors: Iterable[Color]) -> ClusterData:
        """Normalize and fit the clusters with the colors' data and predict cluster data."""
        self.fit(colors=colors)
        return ClusterData.from_fitted_color_clusterer(color_clusterer=self)

    def plot_clusters_figure(
        self,
        original_colors: Iterable[Color],  # Supposed to be in self.space color space
        cluster_data: ClusterData,
        cluster_color_map: Literal["centroid_center"] | LiteralString = "centroid_center",
        background_color: Color = DARK_BACKGROUND_COLOR,
        ax: Axes | None = None,
        with_centers: bool = True,
        with_legend: bool = True,
    ) -> None:
        """Create a plot of the color points on a hue/saturation space."""
        cluster_centers = cluster_data.cluster_centers

        if cluster_color_map == "centroid_center":
            label_colors = [Color(self.space, [0, 0.5, 0.5], 1) for _ in cluster_centers]
            for i, channel in enumerate(self.cluster_channels):
                label_colors = [
                    color.set(channel, cluster_center[i])
                    for color, cluster_center in zip(label_colors, cluster_centers, strict=True)
                ]
            label_colors = [color.convert("srgb").to_string(hex=True) for color in label_colors]
        else:
            cmap = plt.get_cmap(cluster_color_map)
            label_colors = [cmap(i) for i in range(len(cluster_centers))]

        color_points_hex = [color.convert("srgb").to_string(hex=True) for color in original_colors]
        cluster_colors = [label_colors[label] for label in cluster_data.labels]

        # === Scatter ===
        if ax is None:
            _, ax = plt.subplots()

        plot_colors = PlotColors.from_background_color(background_color)
        background_hex_color = background_color.to_string(hex=True)
        lines_hex_color = plot_colors.line.to_string(hex=True)

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
        if with_centers:
            ax.scatter(
                [center[0] for center in cluster_centers],
                [center[1] for center in cluster_centers]
                if len(self.cluster_channels) > 1
                else np.full_like(cluster_centers, 0.5),
                c=label_colors[: len(cluster_centers)],
                s=150,
                marker=centroid_marker,
                edgecolors=lines_hex_color,
                linewidths=1.5,
            )

        # === Legend ===
        if with_legend:
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
            ]
            # Centroids
            if with_centers:
                legend_handles.append(
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
                    )
                )

            ax.legend(handles=legend_handles, title="Clusters", loc="best")

            ax.set_xlabel("Hue")
            ax.set_ylabel("Saturation")
        else:
            ax.tick_params(
                left=False, right=False, labelleft=False, labelbottom=False, bottom=False
            )

        ax.set_title(
            # f"Clusters in {self.space} space "
            f"{self.clusterer.__class__.__name__}, {self.normalizer.__class__.__name__}"
        )

        ax.grid(visible=False)
        ax.set_facecolor(background_hex_color)

    def k_elbow(
        self,
        colors: Iterable[Color],
        k_range: tuple[int, int],
        metric: Literal["distortion", "silhouette", "calinski_harabasz"] = "distortion",
        ax: Axes | None = None,
    ) -> KElbowVisualizer:
        """Return a fitter yellowbrick k elbow visualizer."""
        color_points = self.extract_color_points(colors)

        return kelbow_visualizer(
            model=self.clusterer,
            X=color_points,
            k=k_range,  # pyright: ignore[reportArgumentType]
            ax=ax,
            metric=metric,
            show=False,
            timings=False,
        )


# === Testing ===

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

# Remove Nones
ref_themes = {
    name: load_theme_colors(themes_dark_path / (ref_themes_dark[name] + ".json"), space)
    for name in ref_themes_dark
}

ref_theme = ref_themes["blueberry"]
flat_theme = flatten_dict(ref_theme["colors"], sep="/")

theme_colors = list(flat_theme.values())

# === Kmeans ===
k_range = (2, 14)
nb_init = 100

normalizers = (
    None,
    # MinMaxScaler(),
    # PowerTransformer(),
    # QuantileTransformer(),
    # QuantileTransformer(output_distribution="normal"),
    # RobustScaler(),
    # StandardScaler(),
)
clusterers = (
    *[KMeans(n_clusters=i, init="random", n_init=nb_init) for i in range(*k_range)],
    # *[DBSCAN(min_samples=10, eps=eps) for eps in [2**n for n in range(-5, 1)]],
    # HDBSCAN(min_cluster_size=2, store_centers="centroid"),
)

# === Plot ===
plt.style.use("dark_background")

nb_axes = len(normalizers) * len(clusterers)
nb_row = nb_col = ceil(nb_axes ** (1 / 2))
if nb_axes <= nb_row * (nb_col - 1):
    nb_col -= 1
# FIG_SIZE = 5
axes: NDArray[Any]
# fig, axes = plt.subplots(1, 2, figsize=(2 * FIG_SIZE, FIG_SIZE))
fig, axes = plt.subplots(nb_row, nb_col)
fig.set_layout_engine("constrained")
flat_axes = axes.flatten()

for i, (clusterer, normalizer) in enumerate(itertools.product(clusterers, normalizers)):
    print(
        f"Clustering: {clusterer.__class__.__name__}, Normalizer: {normalizer.__class__.__name__}"
    )
    color_clusterer = ColorClusterer(clusterer=clusterer, normalizer=normalizer)
    cluster_data = color_clusterer.fit_predict(colors=theme_colors)

    ax1 = color_clusterer.plot_clusters_figure(
        original_colors=flat_theme.values(),
        cluster_data=cluster_data,
        # cluster_color_map="tab10",
        ax=flat_axes[i],
        with_legend=False,
        with_centers=False,
    )

    # ax0 = color_clusterer.get_clusters_figure(
    #     original_colors=flat_theme.values(),
    #     cluster_data=cluster_data,
    #     background_color=LIGHT_BACKGROUND_COLOR,
    #     # cluster_color_map="tab10",
    #     ax=axes[0],
    # )


# === Elbow method ===
plt.style.use("default")
fig2, axes2 = plt.subplots(1, 3)
color_clusterer = ColorClusterer(clusterer=KMeans(n_init=nb_init))
viz = color_clusterer.k_elbow(theme_colors, k_range, "distortion", ax=axes2[0])
color_clusterer.k_elbow(theme_colors, k_range, "silhouette", ax=axes2[1])
color_clusterer.k_elbow(theme_colors, k_range, "calinski_harabasz", ax=axes2[2])
plt.show()
