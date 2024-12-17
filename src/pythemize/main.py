"""
Main module.

TODO: Make a dict mapping from colors to theme element that have these colors because we want every color to appear only once
"""

import itertools
import pickle as pkl
from math import ceil
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

import matplotlib.pyplot as plt
from kajihs_utils.pyplot import auto_subplot
from kmedoids import KMedoids  # pyright: ignore[reportMissingTypeStubs]
from nested_dict_tools import flatten_dict
from sklearn.cluster import KMeans
from sklearn.cluster._dbscan import DBSCAN
from sklearn.cluster._hdbscan.hdbscan import HDBSCAN
from sklearn.preprocessing._data import StandardScaler

from pythemize.clustering import (
    HCT_DEFAULT_SUBSPACE,
    HCT_HUE_SUBSPACE,
    OKHSL_DEFAULT_SUBSPACE,
    OKHSL_FULL_SUBSPACE,
    OKHSL_HUE_SUBSPACE,
    OKLAB_DEFAULT_SUBSPACE,
    OKLAB_FULL_SUBSPACE,
    OKLCH_DEFAULT_SUBSPACE,
    OKLCH_HUE_SUBSPACE,
    ClusterData,
    ColorClusterer,
    ColorMultiSubspace,
    ColorSubspace2D,
    ColorSubspaceND,
)
from pythemize.plot import LIGHT_BACKGROUND_COLOR
from pythemize.utils import load_theme_colors

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import NDArray


def main() -> None:  # noqa: PLR0914
    """Test."""
    themes_dark_path = Path("./reference_themes/dark")
    # themes_dark_path = Path("../../reference_themes/dark")
    ref_themes_dark = {
        # "empty": "empty-theme",
        # "arcane": "arcane-color-theme",
        "blueberry": "bearded-theme-surprising-blueberry",
        # "cpp": "cpptools_dark_vs_new-color-theme",
        # "dark_modern": "dark_modern",
    }
    selected_theme = "blueberry"
    data_path = Path("data") / selected_theme
    data_path.mkdir(parents=True, exist_ok=True)
    space = "oklch"

    # Remove Nones
    ref_themes = {
        name: load_theme_colors(themes_dark_path / (ref_themes_dark[name] + ".json"), space)
        for name in ref_themes_dark
    }
    # TODO: Remove colors with nans and create one cluster with them

    ref_theme = ref_themes[selected_theme]
    flat_theme = flatten_dict(ref_theme["colors"], sep="/")

    theme_colors = list(flat_theme.values())
    # # Test with same saturation everywhere
    # theme_colors = [color.set("okhsl.s", 1) for color in theme_colors]

    # === Kmeans ===
    # k_range = (2, 14)
    k_range = (7, 13)
    # k_range = (4, 16)
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
        # *[KMeans(n_clusters=i, n_init=nb_init) for i in range(*k_range)],
        *[KMedoids(n_clusters=i, metric="precomputed") for i in range(*k_range)],
        # *[
        #     DBSCAN(min_samples=40, eps=eps, metric="precomputed")
        #     for eps in [2**n for n in range(-7, 2)]
        # ],
        # *[HDBSCAN(min_cluster_size=n, store_centers="centroid") for n in range(2, 14)],
    )

    clustering_subspaces = (
        # OKLAB_FULL_SUBSPACE,
        OKLAB_DEFAULT_SUBSPACE,
        # OKHSL_HUE_SUBSPACE,
        OKLCH_HUE_SUBSPACE,
        # HCT_HUE_SUBSPACE,
    )
    plot_spaces = (
        # OKHSL_DEFAULT_SUBSPACE,
        # HCT_DEFAULT_SUBSPACE,
        OKLCH_DEFAULT_SUBSPACE,
        OKLAB_DEFAULT_SUBSPACE,
    )

    # Plot colors
    for style in ("default", "dark_background"):
        plt.style.use(style)

        fig, axes = auto_subplot(len(plot_spaces))
        for i, plot_space in enumerate(plot_spaces):
            plot_space.plot_colors(theme_colors, ax=axes[i])
        fig.set_layout_engine("constrained")
        fig.savefig(data_path / f"colors_{style}.png", dpi=300)

    clustering_subspace: ColorSubspaceND | ColorMultiSubspace
    for clustering_subspace in (*clustering_subspaces, ColorMultiSubspace(clustering_subspaces)):
        distance_matrix = clustering_subspace.compute_distance_matrix(theme_colors)
        clustering_space_data_path = data_path / clustering_subspace.get_name()
        clustering_space_data_path.mkdir(exist_ok=True)

        # === Cluster ===
        all_cluster_data: list[ClusterData] = []
        for clusterer, normalizer in itertools.product(clusterers, normalizers):
            print(
                f"Clustering: {clusterer.__class__.__name__}, Normalizer: {normalizer.__class__.__name__}"
            )
            cluster_data = clustering_subspace.cluster(
                colors=theme_colors,
                clusterer=clusterer,
                normalizer=normalizer,
                distance_matrix=distance_matrix,
            )
            all_cluster_data.append(cluster_data)

            pkl_path = clustering_space_data_path / f"{cluster_data.nb_cluster}_clusters.pkl"
            with pkl_path.open("wb") as f:
                pkl.dump(cluster_data, f)

        # === Plot ===
        for plot_subspace in plot_spaces:
            nb_axes = len(normalizers) * len(clusterers)
            nb_row = nb_col = ceil(nb_axes ** (1 / 2))
            if nb_axes <= nb_row * (nb_col - 1):
                nb_col -= 1
            # FIG_SIZE = 5w
            axes: NDArray[Any]
            # fig, axes = plt.subplots(1, 2, figsize=(2 * FIG_SIZE, FIG_SIZE))
            # plt.get_current_fig_manager().full_screen_toggle()
            fig, axes = plt.subplots(nb_row, nb_col)  # pyright: ignore[reportAssignmentType]
            suptitle = f"{clustering_subspace.get_name()}_space_{plot_subspace.get_name()}_plot"
            fig.suptitle(suptitle)
            fig.set_layout_engine("constrained")
            flat_axes = axes.flatten()

            for i, cluster_data in enumerate(all_cluster_data):
                cluster_data.plot_clusters(
                    plot_subspace=plot_subspace,
                    ax=flat_axes[i],
                    # cluster_color_map="tab10",
                    # with_original_point_color=False,
                    with_legend=False,
                    # with_centers=False,
                    background_color=LIGHT_BACKGROUND_COLOR,
                )

                # ax0 = color_clusterer.get_clusters_figure(
                #     original_colors=theme_colors,
                #     cluster_data=cluster_data,
                #     background_color=LIGHT_BACKGROUND_COLOR,
                #     # cluster_color_map="tab10",
                #     ax=axes[0],
                # )

            fig.savefig(
                clustering_space_data_path / f"{plot_subspace.get_name()}_plot.png", dpi=300
            )
            plt.close()

        # === DynMSC ===
        dynk_cluster_data, dynk_res = clustering_subspace.dynmsc(
            colors=theme_colors, k_range=k_range, distance_matrix=distance_matrix
        )
        fig, silhouette_ax = plt.subplots()
        silhouette_ax.plot(range(*k_range), dynk_res.losses)
        fig.savefig(clustering_space_data_path / "space_silhouette.png")
        # dynk_cluster_data.plot_clusters(plot_subspace)

        # === Elbow method ===
        DO_ELBOW = False
        if DO_ELBOW:
            plt.style.use("default")
            _, axes2 = plt.subplots(1, 3)
            ax: Axes
            kmeans_clusterer = KMeans(n_init=nb_init)
            metrics = ("distortion", "silhouette", "calinski_harabasz")

            for metric, ax in zip(metrics, axes2, strict=True):
                ax.set_title(metric)
                clustering_subspace.k_elbow(theme_colors, kmeans_clusterer, k_range, metric, ax)

    # Sort cluster by
    # Plot clusters 1 by 1
    # Group clusters


if __name__ == "__main__":
    main()
