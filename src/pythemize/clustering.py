"""
Tools for clustering colors used in themes.

TODO: Check how to handle nans. -> Currently transformed to 0 when getting color coordinate
    //-> better to just drop ?
    -> Make a cluster specific for them

Todo:
    - Try grouping using several themes (by using more dimensions)
    Optimize:
    - Switching back and froth from colors to color points
    - Computing the colors and color points only once for every run (clusterer takes a list of clusterer instead)
    - Stop copying colors for each projection etc, instead modify the color directly (or at least stop copying and modifying..?)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, Literal, LiteralString, TypeVar, cast, reveal_type

import matplotlib.pyplot as plt
import numpy as np
from attrs import field, frozen
from coloraide import Color
from coloraide.spaces.hct import HCT
from coloraide.spaces.okhsl import Okhsl
from kmedoids import DynkResult, KMedoids, dynmsc  # pyright: ignore[reportMissingTypeStubs]
from matplotlib.lines import Line2D
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

from pythemize.dev_aux import plot_subspace_distances
from pythemize.plot import DARK_BACKGROUND_COLOR, PlotColors

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from coloraide.spaces import Space
    from matplotlib.axes import Axes
    from numpy import float64, int_
    from numpy.typing import NDArray

Color.register(Okhsl())
Color.register(HCT())

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


# TODO: Go back to 3.12 syntax and let covariance being inferred when the following issue is solved: https://github.com/python/mypy/issues/17623
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

    @property
    def nb_channels(self) -> int:
        """Number of channels of the color subspace."""
        return len(self.channels)

    def get_name(self) -> str:
        """Name of the subspace."""
        return f"{self.base_space}({', '.join(self.channels)})"

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

        colors_projected = [Color(color).convert(self.base_space) for color in colors]

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

    def compute_distance_matrix(
        self, colors: Sequence[Color], rescale: bool = False
    ) -> DistanceMatrix:
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

        if rescale:
            # Min-Max Normalization
            min_val = np.min(distance_matrix)
            max_val = np.max(distance_matrix)
            distance_matrix = (distance_matrix - min_val) / (max_val - min_val)

        return distance_matrix

    def cluster(
        self,
        colors: Sequence[Color],
        clusterer: SupportedClusterer,
        normalizer: SupportedNormalizer | None = None,
        distance_matrix: DistanceMatrix | None = None,
    ) -> ClusterData:
        """Cluster the data in the color space with the given clusterer and return the cluster data."""
        color_clusterer = ColorClusterer(
            clusterer=clusterer, normalizer=normalizer, clustering_subspace=self
        )

        return color_clusterer.fit_predict(colors=colors, distance_matrix=distance_matrix)

    def k_elbow(
        self,
        colors: Iterable[Color],
        kmeans_clusterer: KMeans,
        k_range: tuple[int, int],
        metric: Literal["distortion", "silhouette", "calinski_harabasz"] = "distortion",
        ax: Axes | None = None,
    ) -> KElbowVisualizer:
        """Return a fitted yellowbrick k elbow visualizer."""
        color_points = self.to_color_points(colors)

        return kelbow_visualizer(
            model=kmeans_clusterer,
            X=color_points,
            k=k_range,  # pyright: ignore[reportArgumentType]
            ax=ax,
            metric=metric,
            show=False,
            timings=False,
        )

    def dynmsc(
        self,
        colors: Sequence[Color],
        k_range: tuple[int, int],
        distance_matrix: DistanceMatrix | None = None,
    ) -> tuple[ClusterData, DynkResult]:
        """
        Cluster the data using kmedoids.

        k_range: [k_min, k_max) (k_max excluded)
        """
        if distance_matrix is None:
            distance_matrix = self.compute_distance_matrix(colors)

        dynk_result = dynmsc(distance_matrix, k_range[1] - 1, k_range[0])
        cluster_data = ClusterData.from_dynk_result(
            dynk_result=dynk_result,
            clustering_subspace=self,
            colors=colors,
        )

        return cluster_data, dynk_result


type ColorSubspaceND = ColorSubspace[SubspaceChannels]
"""Color subspace with any number of dimensions."""

type ColorSubspace2D = ColorSubspace[tuple[str, str]]
"""Special color subspace type with exactly 2 dimensions."""


@frozen()
class ColorPlotSubspace(ColorSubspace[tuple[str, str]]):
    """Special color subspace for plotting colors."""

    @classmethod
    def from_color_subspace(cls, color_subspace: ColorSubspace2D) -> ColorPlotSubspace:
        """Initialize a ColorPlotSubspace from a 2D color subspace."""
        return cls(base_space=color_subspace.base_space, channels=color_subspace.channels)

    def plot_colors(
        self,
        colors: Iterable[Color] | None = None,
        *,
        cluster_data: ClusterData | None = None,
        convert_colors: bool = True,
        ax: Axes | None = None,
        with_title: bool = True,
    ) -> None:
        """Plot the colors in the subspace."""
        # Verify that argument's compatibility
        if (cluster_data is None) == (colors is None):
            raise ValueError(  # noqa: TRY003
                "Exactly one of `colors` or `cluster_data` paramters should be passed."
            )
        if colors is None:
            cluster_data = cast(ClusterData, cluster_data)
            colors = cluster_data.original_colors

        if convert_colors:
            colors = [color.convert(self.base_space) for color in colors]

        if cluster_data is None:
            # Plot only colors
            inner_colors = colors
            outer_colors = None
        else:
            # Plot cluster color inside and color points color outside
            inner_colors = cluster_data.clus

        if ax is None:
            _, ax = plt.subplots()

        ax.scatter(
            x=[color.get(self.channels[0]) for color in colors],
            y=[color.get(self.channels[1]) for color in colors],
            s=100,
            c=[color.convert("srgb").to_string(hex=True) for color in colors],
        )

        if with_title:
            ax.set_title(self.get_name())

        ax.grid(visible=False)


DEFAULT_COLOR = Color(color="okhsl", data=[0, 0.5, 0.5])
OKLAB_DEFAULT_SUBSPACE = ColorPlotSubspace(base_space="oklab", channels=("a", "b"))
OKLAB_FULL_SUBSPACE = ColorSubspace(base_space="oklab", channels=("l", "a", "b"))
OKLCH_DEFAULT_SUBSPACE = ColorPlotSubspace(base_space="oklch", channels=("h", "c"))
OKLCH_HUE_SUBSPACE = ColorSubspace(base_space="oklch", channels=("h",))
OKHSL_FULL_SUBSPACE = ColorSubspace(base_space="okhsl", channels=("h", "s", "l"))
OKHSL_DEFAULT_SUBSPACE = ColorPlotSubspace(base_space="okhsl", channels=("h", "s"))
OKHSL_NO_HUE_SUBSPACE = ColorPlotSubspace(base_space="okhsl", channels=("s", "l"))
OKHSL_HUE_SUBSPACE = ColorSubspace(base_space="okhsl", channels=("h",))
HCT_DEFAULT_SUBSPACE = ColorPlotSubspace(base_space="hct", channels=("h", "c"))
HCT_HUE_SUBSPACE = ColorSubspace(base_space="hct", channels=("h",))


type VectorNorm = int | float
"""Possible norm for vectors, includes np.inf and -np.inf."""


@frozen()
class ColorMultiSubspace:
    """Container for multiple color subspaces."""

    subspaces: tuple[ColorSubspaceND, ...]
    """Contained subspaces."""
    subspaces_start_indexes: tuple[int, ...] = field(init=False)
    """First index of each subspaces in color points."""

    @subspaces_start_indexes.default  # pyright: ignore[reportAttributeAccessIssue,reportUntypedFunctionDecorator]
    def _subspaces_start_indexes_factory(self) -> tuple[int, ...]:
        start_indexes = [0]
        for subspace in self.subspaces:
            start_indexes.append(start_indexes[-1] + subspace.nb_channels)
        return tuple(start_indexes)

    @property
    def main_subspace(self) -> ColorSubspaceND:
        """Main color subspace in which colors are projected."""
        return self.subspaces[0]

    def get_name(self) -> str:
        """Name of the subspace."""
        return f"{'_X_'.join(subspace.get_name() for subspace in self.subspaces)}"

    def project(
        self,
        colors: Iterable[Color],
        ref_color: Color | None = None,
    ) -> list[Color]:
        """Return colors with values of channels outside of the main color subspace equal to those of the default color."""
        return self.main_subspace.project(colors=colors, ref_color=ref_color)

    def to_color_points(self, colors: Iterable[Color]) -> ColorPoints:
        """Extract color data points from color instances."""
        # Concatenate on the feature dimension
        return np.concatenate(
            [subspace.to_color_points(colors) for subspace in self.subspaces], axis=1
        )

    def color_points_to_colors(self, color_points: ColorPoints) -> list[Color]:
        """Return the projected color corresponding to each color points."""
        # Project on the main subspace
        main_subspace_color_points = color_points[:, 0 : self.main_subspace.nb_channels]
        return self.main_subspace.color_points_to_colors(main_subspace_color_points)

    def compute_distance_matrix(
        self, colors: Sequence[Color], norm_ord: VectorNorm = 2
    ) -> DistanceMatrix:
        """
        Compute the distance matrix of the colors in the color subspaces.

        The global is distance is the norm applied to the rescaled distances
        within each subspace.
        """
        distance_matrices = np.asarray([
            subspace.compute_distance_matrix(colors=colors, rescale=True)
            for subspace in self.subspaces
        ])

        n = len(colors)
        distance_matrix = np.zeros((n, n), dtype=float)

        for i in range(n):
            for j in range(i + 1, n):
                distance_matrix[i, j] = np.linalg.norm(distance_matrices[:, i, j], ord=norm_ord)
                distance_matrix[j, i] = distance_matrix[i, j]

        return distance_matrix

    def cluster(
        self,
        colors: Sequence[Color],
        clusterer: SupportedClusterer,
        normalizer: SupportedNormalizer | None = None,
        distance_matrix: DistanceMatrix | None = None,
    ) -> ClusterData:
        """Cluster the data in the color space with the given clusterer and return the cluster data."""
        color_clusterer = ColorClusterer(
            clusterer=clusterer, normalizer=normalizer, clustering_subspace=self
        )

        return color_clusterer.fit_predict(colors=colors, distance_matrix=distance_matrix)

    def k_elbow(
        self,
        colors: Iterable[Color],
        kmeans_clusterer: KMeans,
        k_range: tuple[int, int],
        metric: Literal["distortion", "silhouette", "calinski_harabasz"] = "distortion",
        ax: Axes | None = None,
    ) -> KElbowVisualizer:
        """Return a fitted yellowbrick k elbow visualizer."""
        color_points = self.to_color_points(colors)

        return kelbow_visualizer(
            model=kmeans_clusterer,
            X=color_points,
            k=k_range,  # pyright: ignore[reportArgumentType]
            ax=ax,
            metric=metric,
            show=False,
            timings=False,
        )

    def dynmsc(
        self,
        colors: Sequence[Color],
        k_range: tuple[int, int],
        distance_matrix: DistanceMatrix | None = None,
    ) -> tuple[ClusterData, DynkResult]:
        """
        Cluster the data using kmedoids.

        k_range: [k_min, k_max) (k_max excluded)
        """
        if distance_matrix is None:
            distance_matrix = self.compute_distance_matrix(colors)

        dynk_result = dynmsc(distance_matrix, k_range[1] - 1, k_range[0])
        cluster_data = ClusterData.from_dynk_result(
            dynk_result=dynk_result,
            clustering_subspace=self,
            colors=colors,
        )

        return cluster_data, dynk_result


type ColorSubspaceLike = ColorSubspaceND | ColorMultiSubspace


def has_precomputed_metric(clusterer: SupportedClusterer) -> bool:
    """Return wether the clusterer's metric is precomputed."""
    return "metric" in clusterer.__dict__ and clusterer.metric == "precomputed"  # pyright: ignore[reportAttributeAccessIssue]


@frozen(kw_only=True)
class ClusterData:
    """Data returned by fitting a color cluster."""

    original_colors: Sequence[Color]
    """Clustered colors."""
    labels: NDArray[int_]
    """Labels of each color."""
    cluster_centers: NDArray[float64]
    """Centers points of clusters."""
    color_subspace: ColorSubspaceLike
    """Color subspace in which the clustering has been performed."""

    cluster_colors: list[Color] = field(init=False)
    """Color of the cluster centers in the clustering subspace."""

    @cluster_colors.default  # pyright: ignore[reportAttributeAccessIssue,reportUntypedFunctionDecorator]
    def _cluster_colors_factory(self) -> list[Color]:
        return self.color_subspace.color_points_to_colors(self.cluster_centers)

    @classmethod
    def from_fitted_color_clusterer(
        cls, color_clusterer: ColorClusterer[SupportedClusterer], colors: Sequence[Color]
    ) -> ClusterData:
        """Initialize from a fitted color clusterer instance. Cluster_centers are unscaled."""
        cluster_centers = cls.extract_cluster_centers(color_clusterer, colors)
        # Data are not scaled if precomputed
        if not has_precomputed_metric(color_clusterer.clusterer):
            cluster_centers = color_clusterer.normalizer_inverse_transform(cluster_centers)

        labels: NDArray[int_] = color_clusterer.clusterer.labels_  # pyright: ignore[reportAssignmentType]
        return cls(
            original_colors=colors,
            labels=labels,
            cluster_centers=cluster_centers,
            color_subspace=color_clusterer.clustering_subspace,
        )

    @classmethod
    def from_dynk_result(
        cls,
        dynk_result: DynkResult,
        clustering_subspace: ColorSubspaceLike,
        colors: Sequence[Color],
    ) -> ClusterData:
        """Initialize from a kmedoids dynk_result instance."""
        cluster_centers = clustering_subspace.to_color_points([
            colors[idx] for idx in dynk_result.medoids
        ])

        return cls(
            original_colors=colors,
            labels=dynk_result.labels,
            cluster_centers=cluster_centers,
            color_subspace=clustering_subspace,
        )

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

    def select_clusters(self, labels: Sequence[int]) -> ClusterData:
        """Return a n instance of cluster data with the selected clusters."""
        colors_filtered = [
            color for i, color in enumerate(self.original_colors) if self.labels[i] in set(labels)
        ]

        return ClusterData(
            original_colors=colors_filtered,
            labels=self.labels[np.isin(self.labels, labels)],
            cluster_centers=self.cluster_centers[labels],
            color_subspace=self.color_subspace,
        )

    def plot_clusters(  # noqa: PLR0913
        self,
        plot_subspace: ColorSubspace2D = OKHSL_DEFAULT_SUBSPACE,
        *,
        ax: Axes | None = None,
        cluster_color_map: Literal["cluster_center"] | LiteralString = "cluster_center",
        background_color: Color = DARK_BACKGROUND_COLOR,
        with_original_point_color: bool = True,
        with_centers: bool = True,
        with_legend: bool = True,
    ) -> None:
        """Plot the colors and clusters in the given plot_subspace."""
        # Define colors of each cluster
        cluster_centers = self.cluster_centers
        cluster_colors = self.color_subspace.color_points_to_colors(cluster_centers)
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
            color.convert(plot_subspace.base_space) for color in self.original_colors
        ]

        cluster_marker = "o"
        ax.scatter(
            [color.get(plot_subspace.channels[0]) for color in original_colors_in_plot_space],
            [color.get(plot_subspace.channels[1]) for color in original_colors_in_plot_space],
            c=[cluster_plot_colors[label] for label in self.labels],
            edgecolors=[
                color.convert("srgb").to_string(hex=True) for color in self.original_colors
            ],
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
            color.convert(plot_subspace.base_space, fit=True) for color in cluster_colors
        ]
        plot_colors = PlotColors.from_background_color(background_color)
        center_marker = "X"
        if with_centers:
            ax.scatter(
                [color.get(plot_subspace.channels[0]) for color in cluster_colors_in_plot_space],
                [color.get(plot_subspace.channels[1]) for color in cluster_colors_in_plot_space],
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

        ax.grid(visible=False)
        ax.set_facecolor(background_color.to_string(hex=True))


@frozen(kw_only=True)
class ColorClusterer[Clusterer: SupportedClusterer]:
    """Base class for color theme clusterer."""

    clusterer: Clusterer
    """Instance of sklearn clusterer."""
    normalizer: SupportedNormalizer | None = None
    """Instance of scaler to use for clustering."""
    clustering_subspace: ColorSubspaceLike = OKLAB_DEFAULT_SUBSPACE
    """Color subspace in which the clustering is performed."""

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
        if has_precomputed_metric(self.clusterer):
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
