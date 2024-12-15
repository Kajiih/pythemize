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

    Optimize:
    - Switching back and froth from colors to color points
    - Computing the colors and color points only once for every run (clusterer takes a list of clusterer instead)
    - Stop copying colors for each projection etc, instead modify the color directly (or at least stop copying and modifying..?)
"""

from __future__ import annotations

import itertools
from math import ceil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, Literal, LiteralString, TypedDict, TypeVar

import matplotlib.pyplot as plt
import numpy as np
from attrs import field, frozen
from coloraide import Color
from coloraide.spaces.hct import HCT
from coloraide.spaces.okhsl import Okhsl
from kmedoids import KMedoids  # pyright: ignore[reportMissingTypeStubs]
from matplotlib.lines import Line2D
from nested_dict_tools import flatten_dict
from sklearn.cluster import DBSCAN, KMeans  # type stubs issue
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
    from collections.abc import Iterable, Sequence

    from coloraide.spaces import Space
    from matplotlib.axes import Axes
    from numpy import float64, int_
    from numpy.typing import NDArray

Color.register(Okhsl())
Color.register(HCT())

# Make a dict mapping from colors to theme element that have these colors because we want every color to appear only once

# TODO: Fix DBSCAN and HDBSCAN
type SupportedClusterer = KMeans | DBSCAN | HDBSCAN | KMedoids
"""Supported sklearn cluster algorithms."""
type SupportedNormalizer = (
    StandardScaler | RobustScaler | PowerTransformer | QuantileTransformer | MinMaxScaler
)
"""Supported sklearn normalizer."""
type ColorPoints = NDArray[float64]
"""Points representing colors in the clustering color space."""
type DistanceMatrix = NDArray[float64]
"""Kernel matrix of distances between colors in the clustering color space."""

type SubspaceChannels = tuple[str] | tuple[str, str] | tuple[str, str, str]
"""Different size of channel tuples."""

Channels_co = TypeVar("Channels_co", covariant=True, bound=SubspaceChannels)


# TODO: Go back to 3.12 synthax and let covariance being inferred when the following issue is solved: https://github.com/python/mypy/issues/17623
@frozen()
class ColorSubspace(Generic[Channels_co]):
    """Subspace of a color space."""

    base_space: LiteralString
    """Name of the base color space."""
    channels: Channels_co
    """Channels of the subspace."""
    space_inst: Space = field(init=False)
    """Instance of the base space class."""

    @space_inst.default  # pyright: ignore[reportAttributeAccessIssue,reportUntypedFunctionDecorator]
    def _space_inst_factory(self) -> Space:
        return Color.CS_MAP[self.base_space]

    channels_out: SubspaceChannels = field(init=False)
    """Channels of the base space but out of the subspace."""

    @channels_out.default  # pyright: ignore[reportAttributeAccessIssue,reportUntypedFunctionDecorator]
    def _channels_out_factory(self) -> SubspaceChannels:
        return tuple(  # pyright: ignore[reportReturnType]
            str(channel) for channel in self.space_inst.channels if channel not in self.channels
        )

    ref_color: Color = field(init=False)
    """Reference color for projections on the subspace."""

    @ref_color.default  # pyright: ignore[reportAttributeAccessIssue,reportUntypedFunctionDecorator]
    def _ref_color_factory(self) -> Color:
        coordinates = [(channel.high - channel.low) / 2 for channel in self.space_inst.CHANNELS]
        return Color(color=self.base_space, data=coordinates)

    # TODO: Improve to directly Create color instances with the good coordinates instead of copying base color and setting channel values
    def project(
        self,
        colors: Iterable[Color],
        ref_color: Color | None = None,
    ) -> list[Color]:
        """Return colors with values of channels outside of the color subspace equal to those of the default color."""
        if ref_color is None:
            ref_color = self.ref_color
        else:
            ref_color = ref_color.convert(space=self.base_space)

        colors_projected = [Color(color) for color in colors]

        for color in colors_projected:
            for channel in self.channels_out:
                color.set(channel, self.ref_color.get(channel))

        return colors_projected

    def to_color_points(self, colors: Iterable[Color]) -> ColorPoints:
        """Extract color data points from color instances."""
        return np.asarray([
            tuple(
                color.convert(space=self.base_space).get(channel, nans=False)
                for channel in self.channels
            )
            for color in colors
        ])

    # TODO: Improve to directly Create color instances with the good coordinates instead of copying base color and setting channel values
    def color_points_to_colors(self, color_points: ColorPoints) -> list[Color]:
        """Return the projected color corresponding to each color points."""
        colors = [Color(self.ref_color) for _ in color_points]

        for color, color_point in zip(colors, color_points, strict=True):
            for i, channel in enumerate(self.channels):
                color.set(channel, color_point[i])

        return colors

    def compute_distance_matrix(self, colors: Sequence[Color]) -> DistanceMatrix:
        """Compute the distance matrix of the colors in the color subspace."""
        colors_projected = self.project(colors)

        n = len(colors_projected)
        distance_matrix = np.zeros((n, n), dtype=float)
        space = self.base_space

        for i in range(n):
            for j in range(i + 1, n):
                dist = colors_projected[i].distance(colors_projected[j], space=space)
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist

        return distance_matrix


DEFAULT_COLOR = Color(color="okhsl", data=[0, 0.5, 0.5])
OKLAB_DEFAULT_SUBSPACE = ColorSubspace(base_space="oklab", channels=("a", "b"))
OKLCH_DEFAULT_SUBSPACE = ColorSubspace(base_space="oklch", channels=("h", "c"))
OKHSL_FULL_SUBSPACE = ColorSubspace(base_space="okhsl", channels=("h", "s", "l"))
OKHSL_DEFAULT_SUBSPACE = ColorSubspace(base_space="okhsl", channels=("h", "s"))
OKHSL_NO_HUE_SUBSPACE = ColorSubspace(base_space="okhsl", channels=("s", "l"))
OKHSL_HUE_SUBSPACE = ColorSubspace(base_space="okhsl", channels=("h",))
HCT_DEFAULT_SUBSPACE = ColorSubspace(base_space="hct", channels=("h", "c"))
HCT_HUE_SUBSPACE = ColorSubspace(base_space="hct", channels=("h",))


def _has_precomputed_metric(clusterer: SupportedClusterer) -> bool:
    """Return wether the clusterer's metric is precomputed."""
    return "metric" in clusterer.__dict__ and clusterer.metric == "precomputed"  # pyright: ignore[reportAttributeAccessIssue]


@frozen(kw_only=True)
class ClusterData:
    """Data returned by fitting a color cluster."""

    labels: NDArray[int_]  # shape (nb_samples,)
    cluster_centers: NDArray[float64]  # shape (nb_clusters, nb_features)

    @classmethod
    def from_fitted_color_clusterer(
        cls, color_clusterer: ColorClusterer[SupportedClusterer], colors: Sequence[Color]
    ) -> ClusterData:
        """Initialize from a fitted color clusterer instance. Cluster_centers are unscaled."""
        cluster_centers = cls.extract_cluster_centers(color_clusterer, colors)
        cluster_centers = color_clusterer.normalizer_inverse_transform(cluster_centers)

        labels: NDArray[int_] = color_clusterer.clusterer.labels_  # pyright: ignore[reportAssignmentType]
        return cls(labels=labels, cluster_centers=cluster_centers)

    @staticmethod
    def extract_cluster_centers(
        color_clusterer: ColorClusterer[SupportedClusterer], colors: Sequence[Color]
    ) -> NDArray[float64]:
        """Extract what is considered cluster centers from the fitted clusterer."""
        clusterer = color_clusterer.clusterer
        match clusterer:
            case KMeans():
                return clusterer.cluster_centers_
            case DBSCAN():
                return clusterer.components_
            case HDBSCAN():
                return clusterer.centroids_
            case KMedoids():
                if clusterer.cluster_centers_ is not None:
                    return clusterer.cluster_centers_
                medoid_colors = [colors[idx] for idx in clusterer.medoid_indices_]
                return color_clusterer.clustering_subspace.to_color_points(medoid_colors)


@frozen(kw_only=True)
class ColorClusterer[Clusterer: SupportedClusterer]:
    """Base class for color theme clusterer."""

    clusterer: Clusterer
    """Instance of clusterer to use for clustering."""
    normalizer: SupportedNormalizer | None = None
    """Instance of scaler to use for clustering."""
    clustering_subspace: ColorSubspace[SubspaceChannels] = OKLAB_DEFAULT_SUBSPACE
    """Color subspace in which the clustering is performed."""
    plot_subspace: ColorSubspace[tuple[str, str]] = OKHSL_DEFAULT_SUBSPACE
    """Color subspace in which to plot."""

    def normalize(self, color_points: ColorPoints) -> ColorPoints:
        """Normalize the color_points."""
        if self.normalizer is None:
            return color_points
        self.normalizer.fit(color_points)

        return self.normalizer.transform(color_points)

    def normalizer_inverse_transform(self, X: NDArray[Any]) -> NDArray[Any]:
        """Perform the normalizer's inverse transform on the input."""
        return X if self.normalizer is None else self.normalizer.inverse_transform(X)

    def fit(self, colors: Sequence[Color], distance_matrix: DistanceMatrix | None = None) -> None:
        """Normalize and fit the cluster with the colors' data."""
        if _has_precomputed_metric(self.clusterer):
            x = (
                distance_matrix
                if distance_matrix is not None
                else self.clustering_subspace.compute_distance_matrix(colors)
            )
        else:
            color_points = self.clustering_subspace.to_color_points(colors=colors)
            x = self.normalize(color_points)
        self.clusterer.fit(X=x)

    def fit_predict(
        self, colors: Sequence[Color], distance_matrix: DistanceMatrix | None = None
    ) -> ClusterData:
        """Normalize and fit the clusters with the colors' data and predict cluster data."""
        self.fit(colors=colors, distance_matrix=distance_matrix)
        return ClusterData.from_fitted_color_clusterer(color_clusterer=self, colors=colors)

    def plot_clusters_figure(  # noqa: PLR0913
        self,
        original_colors: Iterable[Color],  # len(nb_samples)
        cluster_data: ClusterData,
        *,
        cluster_color_map: Literal["cluster_center"] | LiteralString = "cluster_center",
        background_color: Color = DARK_BACKGROUND_COLOR,
        ax: Axes | None = None,
        with_original_point_color: bool = True,
        with_centers: bool = True,
        with_legend: bool = True,
    ) -> None:
        """Create a plot of the color points on a hue/saturation space."""
        # Define colors of each cluster
        cluster_centers = cluster_data.cluster_centers
        cluster_colors = self.clustering_subspace.color_points_to_colors(cluster_centers)
        if cluster_color_map == "cluster_center":
            cluster_plot_colors = [
                color.convert("srgb").to_string(hex=True) for color in cluster_colors
            ]
        else:
            cmap = plt.get_cmap(cluster_color_map)
            cluster_plot_colors = [cmap(i) for i in range(len(cluster_centers))]

        # === Scatter ===
        if ax is None:
            _, ax = plt.subplots()

        # Color points
        # coordinates = coordinates of the original color in the plot space
        # inner color = color of the cluster the point belongs to
        # outer color = original color
        original_colors_in_plot_space = [
            color.convert(self.plot_subspace.base_space) for color in original_colors
        ]

        cluster_marker = "o"
        ax.scatter(
            [color.get(self.plot_subspace.channels[0]) for color in original_colors_in_plot_space],
            [color.get(self.plot_subspace.channels[1]) for color in original_colors_in_plot_space],
            c=[cluster_plot_colors[label] for label in cluster_data.labels],
            edgecolors=[color.convert("srgb").to_string(hex=True) for color in original_colors],
            s=100,
            marker=cluster_marker,
            alpha=1,
            linewidths=2 if with_original_point_color else 0,
        )

        # Cluster centers
        # coordinates = coordinates of the center of the cluster in the plot space
        # inner color = color of the cluster
        # outer color = color of line depending the background color
        cluster_colors_in_plot_space = [
            color.convert(self.plot_subspace.base_space, fit=True) for color in cluster_colors
        ]
        plot_colors = PlotColors.from_background_color(background_color)
        center_marker = "X"
        if with_centers:
            ax.scatter(
                [
                    color.get(self.plot_subspace.channels[0])
                    for color in cluster_colors_in_plot_space
                ],
                [
                    color.get(self.plot_subspace.channels[1])
                    for color in cluster_colors_in_plot_space
                ],
                c=cluster_plot_colors[: len(cluster_centers)],
                s=150,
                marker=center_marker,
                edgecolors=plot_colors.line.to_string(hex=True),
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
                        marker=center_marker,
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
        ax.set_facecolor(background_color.to_string(hex=True))

    def k_elbow(
        self,
        colors: Iterable[Color],
        k_range: tuple[int, int],
        metric: Literal["distortion", "silhouette", "calinski_harabasz"] = "distortion",
        ax: Axes | None = None,
    ) -> KElbowVisualizer:
        """Return a fitter yellowbrick k elbow visualizer."""
        color_points = self.clustering_subspace.to_color_points(colors)

        return kelbow_visualizer(
            model=self.clusterer,
            X=color_points,
            k=k_range,  # pyright: ignore[reportArgumentType]
            ax=ax,
            metric=metric,
            show=False,
            timings=False,
        )


def main() -> None:  # noqa: PLR0914
    """Test."""
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
        # *[KMeans(n_clusters=i, init="random", n_init=nb_init) for i in range(*k_range)],
        *[KMedoids(n_clusters=i, metric="precomputed") for i in range(*k_range)],
        # *[
        #     DBSCAN(min_samples=40, eps=eps, metric="precomputed")
        #     for eps in [2**n for n in range(-7, 2)]
        # ],
        # *[HDBSCAN(min_cluster_size=n, store_centers="centroid") for n in range(2, 11)],
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
    fig, axes = plt.subplots(nb_row, nb_col)  # pyright: ignore[reportAssignmentType]
    fig.set_layout_engine("constrained")
    flat_axes = axes.flatten()

    class SubspacesDict(TypedDict):
        clustering_subspace: ColorSubspace[SubspaceChannels]
        plot_subspace: ColorSubspace[tuple[str, str]]

    subspaces = SubspacesDict(
        clustering_subspace=OKHSL_HUE_SUBSPACE,
        plot_subspace=OKLAB_DEFAULT_SUBSPACE,
    )

    distance_matrix = subspaces["clustering_subspace"].compute_distance_matrix(theme_colors)

    for i, (clusterer, normalizer) in enumerate(itertools.product(clusterers, normalizers)):
        print(
            f"Clustering: {clusterer.__class__.__name__}, Normalizer: {normalizer.__class__.__name__}"
        )
        color_clusterer = ColorClusterer(clusterer=clusterer, normalizer=normalizer, **subspaces)
        cluster_data = color_clusterer.fit_predict(
            colors=theme_colors, distance_matrix=distance_matrix
        )

        color_clusterer.plot_clusters_figure(
            original_colors=theme_colors,
            cluster_data=cluster_data,
            # cluster_color_map="tab10",
            ax=flat_axes[i],
            # with_original_point_color=False,
            with_legend=False,
            # with_centers=False,
        )

        # ax0 = color_clusterer.get_clusters_figure(
        #     original_colors=theme_colors,
        #     cluster_data=cluster_data,
        #     background_color=LIGHT_BACKGROUND_COLOR,
        #     # cluster_color_map="tab10",
        #     ax=axes[0],
        # )

    # === Elbow method ===
    DO_ELBOW = False
    if DO_ELBOW:
        plt.style.use("default")
        _, axes2 = plt.subplots(1, 3)
        color_clusterer = ColorClusterer(clusterer=KMeans(n_init=nb_init), **subspaces)
        color_clusterer.k_elbow(theme_colors, k_range, "distortion", ax=axes2[0])
        color_clusterer.k_elbow(theme_colors, k_range, "silhouette", ax=axes2[1])
        color_clusterer.k_elbow(theme_colors, k_range, "calinski_harabasz", ax=axes2[2])

    plt.show()


if __name__ == "__main__":
    main()
