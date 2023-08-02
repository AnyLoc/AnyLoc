# View the PCA plot for an algorithm
"""
    Clustering descriptors across all datasets
"""

# %%
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
import sys


# %%
def plot_pca(descs_db: dict, show_plot: bool=False, 
            cache_fname=None, title="Dino", fit_db_tf_qu: bool=False):
    # Colors and markers
    db_colors = {
        "Oxford": "#008000",
        "gardens": "#004ccc",
        "17places": "#004ccc",
        "baidu_datasets": "#004ccc",
        "st_lucia": "#008000",
        "pitts30k": "#008000",
        "Tartan_GNSS_test_rotated": "#800080",
        "Tartan_GNSS_test_notrotated": "#800080",
        "hawkins": "#80471c",
        "laurel_caverns": "#80471c",
        "eiffel": "#297bd8",
        "VPAir": "#800080"
    }
    db_markers = {
        "Oxford": "^",
        "gardens": "p",
        "17places": "P",
        "baidu_datasets": "*",
        "st_lucia": "v",
        "pitts30k": "<",
        "Tartan_GNSS_test_rotated": "_",
        "Tartan_GNSS_test_notrotated": "|",
        "hawkins": "1",
        "laurel_caverns": "2",
        "eiffel": "x",
        "VPAir": "d"
    }
    qu_alphas = 0.5
    # List of datasets being used
    if fit_db_tf_qu:
        use_ds = list(descs_db["database"])
    else:
        use_ds = list(descs_db)
    # Plot figure
    fig = plt.figure()
    for db in use_ds:
        if fit_db_tf_qu:
            plt.scatter(descs_db["database"][db][:, 0], 
                    descs_db["database"][db][:, 1],
                    label=db, c=db_colors[db], marker=db_markers[db])
            plt.scatter(descs_db["queries"][db][:, 0],
                    descs_db["queries"][db][:, 1], alpha=qu_alphas,
                    c=db_colors[db], marker=db_markers[db])
        else:
            plt.scatter(descs_db[db][:, 0], descs_db[db][:, 1], 
                    label=db, c=db_colors[db], marker=db_markers[db])
    plt.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=3)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.tight_layout()
    if cache_fname is not None:
        plt.savefig(f"{cache_fname}_pca.png")
        print(f"Saved figure to {cache_fname}_pca.png")
    if show_plot:
        plt.show()
    else:
        return fig


# %%
if __name__ == "__main__" and ("ipykernel" not in sys.argv[0]):
    cache_file = "/scratch/avneesh.mishra/vl-vpr/cache/dataset_clusters/result_dino_v2_pca.gz"
    assert os.path.isfile(cache_file), f"File {cache_file} does not exist"
    descs_db = joblib.load(cache_file)
    plot_pca(descs_db, show_plot=True)


# %%
cache_file = "/scratch/avneesh.mishra/vl-vpr/cache/dataset_clusters/result_dino_v2_trdbtfq_pca.gz"
descs_db = joblib.load(cache_file)
ret = plot_pca(descs_db, show_plot=False, cache_fname="/scratch/avneesh.mishra/out", fit_db_tf_qu=True)

# %%
