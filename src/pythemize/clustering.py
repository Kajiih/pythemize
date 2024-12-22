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

from collections.abc import Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    LiteralString,
    TypedDict,
    TypeVar,
    cast,
    overload,
)
from warnings import deprecated

import matplotlib.pyplot as plt
import numpy as np
from attrs import evolve, field, frozen
from coloraide import Color
from coloraide.spaces.hct import HCT
from coloraide.spaces.okhsl import Okhsl
from kajihs_utils.pyplot import auto_subplot
from kmedoids import DynkResult, KMedoids, dynmsc  # pyright: ignore[reportMissingTypeStubs]
from numpy import int_
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

from pythemize.plot import PlotColors

if TYPE_CHECKING:
    from collections.abc import Iterable

    from coloraide.spaces import Space
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy import float64
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

    channel_bounds: tuple[tuple[float, float], ...] = field(init=False)
    """Bounds of each channels."""

    @channel_bounds.default  # pyright: ignore[reportAttributeAccessIssue,reportUntypedFunctionDecorator]
    def _channel_bounds_fatory(self) -> tuple[tuple[float, float], ...]:
        all_channels = self.space_inst.CHANNELS

        res: list[tuple[float, float]] = []
        for channel_name in self.channels:
            for channel in all_channels:
                if channel_name == str(channel):
                    res.append((channel.low, channel.high))
                    break
            else:
                raise RuntimeError("This should not happen.")  # noqa: TRY003

        return tuple(res)

    channels_out: SubspaceChannels = field(init=False)
    """Channels of the base space but out of the subspace."""
    # TODO? Remove alpha channel from this

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

    @overload
    def plot_colors(
        self,
        colors: Iterable[Color],
        *,
        # cluster_data: ClusterData | None = None,
        convert_colors: bool = True,
        ax: Axes | None = None,
        with_title: bool = True,
    ) -> None: ...

    @overload
    def plot_colors(
        self,
        # colors: Iterable[Color] | None = None,
        *,
        cluster_data: ClusterData,
        convert_colors: bool = True,
        ax: Axes | None = None,
        with_title: bool = True,
    ) -> None: ...

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
        # Verify the arguments' compatibility
        if (cluster_data is None) == (colors is None):
            raise ValueError(  # noqa: TRY003
                "Exactly one of `colors` or `cluster_data` parameters should be passed."
            )
        if colors is None:
            cluster_data = cast(ClusterData, cluster_data)
            colors = cluster_data.original_colors

        if convert_colors:
            colors = [color.convert(self.base_space) for color in colors]

        # Define colors of points
        colors_hex = [color.convert("srgb").to_string(hex=True) for color in colors]
        if cluster_data is None:
            # Plot only colors
            inner_colors = colors_hex
            outer_colors = None
        else:
            # Plot cluster color inside and color points color outside
            inner_colors = [cluster_data.cluster_colors[label] for label in cluster_data.labels]
            inner_colors = [color.convert("srgb").to_string(hex=True) for color in inner_colors]
            outer_colors = colors_hex

        if ax is None:
            _, ax = plt.subplots()
        ax.grid(visible=False)

        ax.scatter(
            x=[color.get(self.channels[0]) for color in colors],
            y=[color.get(self.channels[1]) for color in colors],
            s=100,
            c=inner_colors,
            edgecolors=outer_colors,
            linewidths=2,
        )

        if with_title:
            ax.set_title(self.get_name())

    def plot_cluster_centers(
        self,
        cluster_data: ClusterData,
        *,
        convert_colors: bool = True,
        ax: Axes | None = None,
    ) -> None:
        """Plot the cluster centers."""
        if ax is None:
            _, ax = plt.subplots()

        background_color = ax.get_facecolor()
        line_color = PlotColors.from_background_color(background_color).line

        cluster_colors = cluster_data.cluster_colors
        if convert_colors:
            cluster_colors = [color.convert(self.base_space, fit=True) for color in cluster_colors]

        ax.scatter(
            [color.get(self.channels[0]) for color in cluster_colors],
            [color.get(self.channels[1]) for color in cluster_colors],
            c=[color.convert("srgb").to_string(hex=True) for color in cluster_colors],
            s=150,
            marker="X",
            edgecolors=line_color.to_string(hex=True),
            linewidths=1.5,
        )

    def plot_separate_clusters(
        self,
        cluster_data: ClusterData,
        *,
        convert_colors: bool = True,
    ) -> tuple[Figure, np.ndarray[tuple[int], Any]]:
        """Plot all clusters on separate axes."""
        fig, axes = auto_subplot(cluster_data.nb_clusters, transposed=True)
        fig.set_layout_engine("constrained")

        x_y_lim, y_lim = (0, 0), (0, 0)
        ax: Axes
        for i, ax in enumerate(axes):
            cluster_i = cluster_data.select_clusters(i)
            self.plot_colors(
                cluster_data=cluster_i, convert_colors=convert_colors, ax=ax, with_title=False
            )
            ax.set_title(f"Label: {i}", fontsize=15)
            # ax.set_xlim(*self.channel_bounds[0])
            # ax.set_ylim(*self.channel_bounds[1])

        # === Set the same limits to every axes ===
        x_lims = [ax.get_xlim() for ax in axes]
        y_lims = [ax.get_ylim() for ax in axes]

        n = 15 / 100
        x_y_lims = []
        for lims in (x_lims, y_lims):
            x_y_lim = min(lim[0] for lim in lims), max(lim[1] for lim in lims)

            # Enlarge by n%
            span = x_y_lim[1] - x_y_lim[0]
            x_y_lim = (x_y_lim[0] - span * n / 2, x_y_lim[1] + span * n / 2)

            x_y_lims.append(x_y_lim)

        x_lim, y_lim = x_y_lims

        for ax in axes:
            ax.set_xlim(*x_lim)
            ax.set_ylim(*y_lim)
        return fig, axes

    def plot_colors_and_clusters_centers(
        self,
        cluster_data: ClusterData,
        *,
        convert_colors: bool = True,
        ax: Axes | None = None,
        with_title: bool = True,
    ) -> None:
        """Plot the colors and the cluster centers."""
        if ax is None:
            _, ax = plt.subplots()
        self.plot_colors(
            cluster_data=cluster_data, convert_colors=convert_colors, ax=ax, with_title=with_title
        )
        self.plot_cluster_centers(cluster_data=cluster_data, convert_colors=convert_colors, ax=ax)


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

    @property
    def base_space(self) -> LiteralString:
        """Base space of the main subspace."""
        return self.main_subspace.base_space

    def get_name(self) -> str:
        """Name of the subspace."""
        return f"{'_X_'.join(subspace.get_name() for subspace in self.subspaces)}"

    # TODO: Replace by a projection on the first space with the most dimensions to lose less information
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

    # TODO: Replace by a projection on the first space with the most dimensions to lose less information
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


class SerializedClusterData(TypedDict):
    """Serializable cluster data."""

    labels: list[int]
    cluster_centers: list[float]
    color_subspace: str


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

    @property
    def nb_clusters(self) -> int:
        """The number of clusters."""
        return len(self.cluster_centers)

    # def __attrs_post_init__(self):
    #     """Automatically reorder clusters at initialization."""
    #     sorted_indices = np.lexsort(self.cluster_centers.T[::-1])

    #     # Reorder cluster centers and colors
    #     new_cluster_centers = self.cluster_centers[sorted_indices]
    #     new_cluster_colors = [self.cluster_colors[i] for i in sorted_indices]

    #     # Update labels to match the new cluster order
    #     label_map = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted_indices)}
    #     new_labels = np.array(self.labels.shape, dtype=int_)
    #     for i, label in enumerate(self.labels):
    #         new_labels[i] = label_map[label]

    #     # Replace attributes with reordered values
    #     object.__setattr__(self, "cluster_centers", new_cluster_centers)
    #     object.__setattr__(self, "cluster_colors", new_cluster_colors)
    #     object.__setattr__(self, "labels", new_labels)

    @classmethod
    def from_labels(
        cls, labels: Iterable[int], colors: Sequence[Color], color_subspace: ColorSubspaceLike
    ) -> ClusterData:
        """
        Instantiate from a sequence of labels and compute cluster centers as the average color in a cluster.

        Labels have to be contiguous.
        """
        labels = np.asarray(labels)
        nb_clusters = max(labels) + 1

        # Group colors by their cluster labels.
        colors_by_label = [
            [color for color, color_label in zip(colors, labels, strict=True) if color_label == i]
            for i in range(nb_clusters)
        ]

        # Compute averaged colors for each cluster using Color.average.
        averaged_colors = [
            Color.average(cluster_colors, space=color_subspace.base_space)
            for cluster_colors in colors_by_label
        ]

        # Convert averaged colors to points in the desired color space.
        cluster_centers = np.asarray([
            color_subspace.to_color_points([avg_color]) for avg_color in averaged_colors
        ])

        return cls(
            original_colors=colors,
            labels=labels,
            cluster_centers=cluster_centers,
            color_subspace=color_subspace,
        )

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

    def select_clusters(self, labels: Sequence[int] | int) -> ClusterData:
        """
        Return a n instance of cluster data with the selected clusters.

        The labels of the new cluster data will be contiguous, and the clusters
        will follow the provided order. e.g., if labels = [5, 0], the colors
        with label 5 will be labeled 0, those with label, will be labeled 1, and
        the others will be dropped.
        """
        if not isinstance(labels, Sequence):
            labels = [labels]

        # Ensure labels are unique and contiguous
        labels = list(dict.fromkeys(labels))  # Removes duplicates while preserving order
        label_mapping = {labels[i]: i for i in range(len(labels))}

        contiguous_labels = np.array([
            label_mapping[label] for label in self.labels if label in label_mapping
        ])

        colors_filtered = [
            color for i, color in enumerate(self.original_colors) if self.labels[i] in set(labels)
        ]

        return evolve(
            self,
            original_colors=colors_filtered,
            labels=contiguous_labels,
            cluster_centers=self.cluster_centers[labels],
        )

    def ordered(self) -> ClusterData:
        """Return an ordered version of cluster data with clusters in lexicographic order of their center coordinates."""
        new_indices = np.lexsort(self.cluster_centers.T[::-1])

        return self.select_clusters(new_indices)

    def serialize(self) -> SerializedClusterData:
        """Serialize labels, space name and cluster centers into dict."""
        return SerializedClusterData(
            labels=self.labels.tolist(),  # pyright: ignore[reportArgumentType]
            cluster_centers=self.cluster_centers.tolist(),  # pyright: ignore[reportArgumentType]
            color_subspace=self.color_subspace.get_name(),
        )

    @classmethod
    def deserialize(
        cls,
        d: SerializedClusterData,
        colors: Sequence[Color],
    ) -> ClusterData:
        """Deserialize the cluster data."""
        subspace_name, channels = d["color_subspace"].split("(")
        channels = tuple(channels.strip(")").split(","))
        return cls(
            original_colors=colors,
            labels=np.asarray(d["labels"]),
            cluster_centers=np.asarray(d["cluster_centers"]),
            color_subspace=ColorSubspace(subspace_name, channels),  # pyright: ignore[reportArgumentType]
        )

    def relabel(
        self,
        merged_labels: Iterable[Iterable[int]] | None = None,
        color_relabel: dict[int, int] | None = None,
    ) -> ClusterData:
        """
        Merge the clusters with labels as keys in cluster_relabel and colors with indices as keys to their associated value.

        All centers are recomputed by using the from_label methods of ClusterData.
        """
        labels = self.labels

        # Merge labels
        if merged_labels is not None:
            cluster_relabel = {
                label: next(iter(label_group))
                for label_group in merged_labels
                for label in label_group
            }
            labels = [cluster_relabel.get(label, label) for label in labels]

        # Change single colors' labels
        if color_relabel is not None:
            labels = [color_relabel.get(i, label) for i, label in enumerate(labels)]  # noqa: FURB140

        # Make labels contiguous from 0 to nb_labels
        all_labels = sorted(set(labels))
        label_mapping = {all_labels[i]: i for i in range(len(all_labels))}

        contiguous_labels = [label_mapping[label] for label in labels]

        return ClusterData.from_labels(
            labels=contiguous_labels,
            colors=self.original_colors,
            color_subspace=self.color_subspace,
        )


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
