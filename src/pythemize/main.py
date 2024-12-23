"""
Main module.

TODO: Make a dict mapping from colors to theme element that have these colors because we want every color to appear only once
"""

import itertools
from pathlib import Path
from typing import TYPE_CHECKING

import json5
import matplotlib.pyplot as plt
import yaml
from coloraide import Color
from kajihs_utils.pyplot import auto_subplot
from kmedoids import KMedoids  # pyright: ignore[reportMissingTypeStubs]
from nested_dict_tools import flatten_dict, unflatten_dict
from rich.pretty import pprint
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
    OKLCH_FULL_SUBSPACE,
    OKLCH_HUE_SUBSPACE,
    ClusterData,
    ColorMultiSubspace,
    ColorSubspaceND,
)
from pythemize.colors import (
    HEXTECH_BLUE,
    HEXTECH_GOLD,
    HEXTECH_MAGIC,
    JINX_BLUE,
    JINX_PURPLE,
    KJ_PURPLE,
    SHIMMER_PINK,
    ZAUN_GREEN,
)
from pythemize.utils import ThemeColorDict, ThemeDict, load_theme_colors

if TYPE_CHECKING:
    from matplotlib.axes import Axes

FLATTENING_SEP = "/"
BASE_THEME_PATH = Path("/Users/julian/dev_projects/pythemize/reference_themes/dark/kj-base_bk.json")
NEW_THEME_PATH = Path("/Users/julian/dev_projects/vscode_themes/arcane-theme/themes/kj-base.json")


def main() -> None:  # noqa: PLR0914
    """Test."""
    themes_dark_path = Path("./reference_themes/dark")
    # themes_dark_path = Path("../../reference_themes/dark")
    ref_themes_dark = {
        # "empty": "empty-theme",
        # "arcane": "arcane-color-theme",
        "blueberry": "bearded-theme-surprising-blueberry",
        "kj-base": "kj-base_bk",
        # "cpp": "cpptools_dark_vs_new-color-theme",
        # "dark_modern": "dark_modern",
    }
    selected_theme = "kj-base"
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
    flat_theme = flatten_dict(ref_theme["colors"], sep=FLATTENING_SEP)

    theme_colors = list(flat_theme.values())
    # # Test with same saturation everywhere
    # theme_colors = [color.set("okhsl.s", 1) for color in theme_colors]

    DO_CLUSTER = False
    if DO_CLUSTER:
        # === Kmeans ===
        # k_range = (2, 14)
        k_range = (7, 16)
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
        for style in ("dark_background", "bmh"):
            plt.style.use(style)

            fig, axes = auto_subplot(len(plot_spaces))
            for i, plot_space in enumerate(plot_spaces):
                plot_space.plot_colors(theme_colors, ax=axes[i])
            fig.set_layout_engine("constrained")
            fig.savefig(data_path / f"colors_{style}.png", dpi=300)

        clustering_subspace: ColorSubspaceND | ColorMultiSubspace
        for clustering_subspace in (
            *clustering_subspaces,
            ColorMultiSubspace(clustering_subspaces),
        ):
            distance_matrix = clustering_subspace.compute_distance_matrix(theme_colors)
            clustering_space_data_path = data_path / clustering_subspace.get_name()
            cluster_data_path = clustering_space_data_path / "cluster_data"
            cluster_data_path.mkdir(exist_ok=True, parents=True)

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
                ).ordered()
                all_cluster_data.append(cluster_data)

                # pkl_path = cluster_data_path / f"{cluster_data.nb_clusters}_clusters.pkl"
                # with pkl_path.open("wb") as f:
                #     pkl.dump(cluster_data, f)
                labels_path = cluster_data_path / f"{cluster_data.nb_clusters}_clusters_labels.yaml"
                with labels_path.open("w") as f:
                    yaml.dump(cluster_data.serialize(), f)

            # === Plot ===
            for plot_subspace in plot_spaces:
                fig, axes = auto_subplot(len(normalizers) * len(clusterers))
                suptitle = f"{clustering_subspace.get_name()}_space_{plot_subspace.get_name()}_plot"
                fig.suptitle(suptitle)
                fig.set_layout_engine("constrained")

                for i, cluster_data in enumerate(all_cluster_data):
                    plot_subspace.plot_colors_and_clusters_centers(
                        cluster_data=cluster_data,
                        ax=axes[i],
                    )

                    separate_cluster_fig, _ = plot_subspace.plot_separate_clusters(cluster_data)

                    cluster_data_i_path = (
                        clustering_space_data_path / f"{cluster_data.nb_clusters}_clusters"
                    )
                    cluster_data_i_path.mkdir(parents=True, exist_ok=True)
                    separate_cluster_fig.savefig(
                        cluster_data_i_path / f"{plot_subspace.get_name()}_plot.png",
                        dpi=300,
                    )
                    plt.close()

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
                plt.style.use("bmh")
                _, axes2 = plt.subplots(1, 3)
                ax: Axes
                kmeans_clusterer = KMeans(n_init=nb_init)
                metrics = ("distortion", "silhouette", "calinski_harabasz")

                for metric, ax in zip(metrics, axes2, strict=True):
                    ax.set_title(metric)
                    clustering_subspace.k_elbow(theme_colors, kmeans_clusterer, k_range, metric, ax)

    plt.close()
    # Sort clusters
    # Plot clusters 1 by 1
    # Group clusters
    selected_cluster_paths = (
        # "data/blueberry/oklch(h)/cluster_data/12_clusters.pkl",
        "data/blueberry/oklch(h)/cluster_data/12_clusters_labels.yaml",
        # "data/blueberry/oklab(a, b)_X_oklch(h)/cluster_data/10_clusters.pkl",
    )
    selected_cluster_paths = tuple(Path(p) for p in selected_cluster_paths)

    cluster_path = selected_cluster_paths[0]
    with cluster_path.open("rb") as f:
        cluster_data = ClusterData.deserialize(yaml.safe_load(f), colors=theme_colors)

    plot_space = OKLCH_DEFAULT_SUBSPACE

    PLOT_ORIGNAL_SUBSPACE = False
    if PLOT_ORIGNAL_SUBSPACE:
        plot_space.illustrate_clusters(cluster_data)

    # === Export ===
    export_space = OKLCH_FULL_SUBSPACE

    # color_coordinates = plot_space.to_color_points(theme_colors)
    relabeled_cluster_data = cluster_data.relabel(
        # cluster_relabel={7: 6, 8: 6, 1: 0, 11: 0},
        merged_labels=[{6, 7, 8}, {0, 1, 11}],
        # color_relabel={
        #     int(find_closest(color_coordinates, [0.7, 0.16])): 1,
        #     int(find_closest(color_coordinates, [361, 0.01])): 11,
        # },
        export_subspace=export_space,
    )

    plot_space.illustrate_clusters(relabeled_cluster_data)
    pprint(relabeled_cluster_data.cluster_colors)
    # TODO: Check why there's a missing cluster on the PLOT_ORIGNAL_SUBSPACE

    ref_colors = [Color(color) for color in relabeled_cluster_data.cluster_colors]

    background_color = ref_colors[5]
    highlight_color = ref_colors[0]
    gold_color = ref_colors[2]

    # === Jinx ===
    # ref_colors[7].set("h", ref_colors[0].get("oklch.h"))
    # background_color.set("h", JINX_BLUE.get("oklch.h"))
    # # background_color.set("l", background_color.get("l") / 1.1)
    # highlight_color.set("h", SHIMMER_PINK.get("oklch.h"))

    # === Jinx 2 ===
    # ref_colors[7].set("h", ref_colors[0].get("oklch.h"))
    # background_color.set("h", KJ_PURPLE.get("oklch.h"))
    # # background_color.set("l", background_color.get("l") / 1.05)
    # highlight_color.set("h", JINX_BLUE.get("oklch.h"))

    # === KJ ===
    ref_colors[7].set("h", ref_colors[0].get("oklch.h"))
    background_color.set("h", HEXTECH_BLUE.get("oklch.h"))
    gold_color.set("h", HEXTECH_GOLD.get("oklch.h"))
    # background_color.set("l", background_color.get("l") / 1.05)
    highlight_color.set("h", KJ_PURPLE.get("oklch.h"))

    # === Hextech ===
    # ref_colors[2].set("h", ref_colors[0].get("oklch.h"))
    # ref_colors[7].set("h", HEXTECH_MAGIC.get("oklch.h"))
    # background_color.set("h", HEXTECH_BLUE.get("oklch.h"))
    # highlight_color.set("h", HEXTECH_GOLD.get("oklch.h"))

    # === Zaun ===
    # ref_colors[7].set("h", ref_colors[5].get("oklch.h"))
    # background_color.set("h", ZAUN_GREEN.get("oklch.h"))
    # background_color.set("l", background_color.get("l") / 1.1)
    # highlight_color.set("h", SHIMMER_PINK.get("oklch.h"))

    new_clusters = relabeled_cluster_data.shift_clusters(
        ref_colors=ref_colors, export_subspace=export_space
    )
    pprint(new_clusters.cluster_colors)

    plot_space.illustrate_clusters(new_clusters)

    new_colors = [
        color.convert("srgb").to_string(hex=True) for color in new_clusters.original_colors
    ]
    new_flat_theme = dict(zip(flat_theme.keys(), new_colors, strict=True))
    new_theme = ThemeDict(colors=unflatten_dict(new_flat_theme, sep=FLATTENING_SEP))

    with BASE_THEME_PATH.open("r") as f:
        theme = json5.load(f)

    theme["colors"] = new_theme["colors"]
    with NEW_THEME_PATH.open("w") as f:
        json5.dump(theme, f)

    plt.show()


if __name__ == "__main__":
    main()
