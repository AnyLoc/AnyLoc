# Script to plot the latent space visualizations
"""
    This script is not dependent on library and reads everything from 
    the cache (can run anywhere).
"""
import joblib
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D
from matplotlib.text import TextPath
import numpy as np
import os

def pca_colormap():
    colormap = {
        "Baidu Mall": "#0792c2",
        "Gardens Point": "#0792c2",
        "17 Places": "#0792c2",
        "Pitts-30k": "#008000",
        "St Lucia": "#008000",
        "Oxford": "#008000",
        "Hawkins": "#80471c",
        "Laurel Caverns": "#80471c",
        "Nardo-Air": "#800080",
        "Nardo-Air R": "#800080",
        "VP-Air": "#800080",
        "Mid-Atlantic Ridge": "#4cc3ef",
    }
    return colormap

def pca_markermap():
    markermap = {
        "Baidu Mall": "8",
        "Gardens Point": "h",
        "17 Places": "H",
        "Pitts-30k": "^",
        "St Lucia": ">",
        "Oxford": "<",
        "Hawkins": "+",
        "Laurel Caverns": "x",
        "Nardo-Air": "1",
        "Nardo-Air R": "2",
        "VP-Air": "3",
        "Mid-Atlantic Ridge": "*",
    }
    return markermap

def key_remap(descs_dict: dict):
    keymap = {
        "baidu_datasets": "Baidu Mall",
        "gardens": "Gardens Point",
        "17places": "17 Places",
        "pitts30k": "Pitts-30k",
        "st_lucia": "St Lucia",
        "Oxford": "Oxford",
        "hawkins": "Hawkins",
        "laurel_caverns": "Laurel Caverns",
        "Tartan_GNSS_test_notrotated": "Nardo-Air",
        "Tartan_GNSS_test_rotated": "Nardo-Air R",
        "VPAir": "VP-Air",
        "eiffel": "Mid-Atlantic Ridge",
    }
    remapped_dict = {}
    for key in keymap.keys():
        remapped_dict[keymap[key]] = descs_dict[key]
    return remapped_dict

def plot_pca(dump_file_paths:list, output_path: str):
    """Plot the PCA for Baseline and FoundLoc Baseline"""
    query_marker_alpha = 0.5
    colormap = pca_colormap()
    markermap = pca_markermap()
    fig, ax = plt.subplots(figsize=(3.5, 6), nrows=2, ncols=1)
    # Loop over the dump files
    for i, dump_file in enumerate(dump_file_paths):
        # Load the dump
        dump_file = dump_file_paths[i]
        assert os.path.isfile(dump_file)
        fname = os.path.realpath(os.path.expanduser(dump_file))
        data = joblib.load(fname)
        db_dict = key_remap(data["database"])
        query_dict = key_remap(data["queries"])
        # Plot PCA points
        for key in db_dict.keys():
            ax[i].scatter(db_dict[key][:, 0], db_dict[key][:, 1], c=colormap[key], marker=markermap[key], label=key)
            ax[i].scatter(query_dict[key][:, 0], query_dict[key][:, 1], c=colormap[key], marker=markermap[key], alpha=query_marker_alpha)
            ax[i].set_xticks([])
            ax[i].set_yticks([])
    # Axis Spline Color
    spline_colors = ['orangered', 'darkorange']
    for i in range(2):
        ax[i].spines['bottom'].set_color(spline_colors[i])
        ax[i].spines['top'].set_color(spline_colors[i])
        ax[i].spines['right'].set_color(spline_colors[i])
        ax[i].spines['left'].set_color(spline_colors[i])
        if i == 0:
            ax[i].spines['bottom'].set_linestyle("dashed")
            ax[i].spines['top'].set_linestyle("dashed")
            ax[i].spines['right'].set_linestyle("dashed")
            ax[i].spines['left'].set_linestyle("dashed")
    # Legend indicating baseline name
    legend_elements = [plt.scatter([], [], color='orangered', label='MixVPR', marker='none'),]
    ax[0].legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1), ncol=1, prop={'weight': 'bold', 'size': 10}, labelspacing=2, labelcolor=['orangered'], handlelength=2.5)
    # Dataset legend
    # ds_legend = ax[0].legend(loc='upper right', bbox_to_anchor=(0, 1), ncol=2, prop={'weight': 'bold', 'size': 5}, labelspacing=2.5, labelcolor=[colormap[key] for key in colormap.keys()])
    # Legend indicating FoundLoc baseline name & marker legend
    legend_elements = [plt.scatter([], [], color='darkorange', label='FoundGeM (DINOv2)', marker='none'),]
    legend0 = ax[1].legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1), ncol=1, prop={'weight': 'bold', 'size': 10}, labelspacing=2, labelcolor=['darkorange'])
    legend_elements = [plt.scatter([], [], marker='o', color='grey', label='Database'),
                       plt.scatter([], [], marker='o', color='grey', label='Query', alpha=query_marker_alpha),]
    ax[1].legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(1, 0), ncol=1, prop={'weight': 'bold', 'size': 10}, labelspacing=1)
    ax[1].add_artist(legend0)
    # Save the plot
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight', dpi=400)
    print(f"Saved plot to {output_path}")

def dump_file_data():
    "Dump File paths for PCA cache of Baseline and FoundLoc Baseline"
    dump_fps = ['/ocean/projects/cis220039p/nkeetha/data/vlm/found/pca/result_mixvpr_trdbtfq_pca.gz',
                '/ocean/projects/cis220039p/nkeetha/data/vlm/found/pca/result_dino_v2_trdbtfq_pca.gz']
    return dump_fps

if __name__ == '__main__':
    ## Set plt to seaborn style
    plt.style.use('seaborn-v0_8-white')
    
    ## Latent Space Visualization
    dump_file_paths = dump_file_data()
    output_path = "/ocean/projects/cis220039p/nkeetha/data/vlm/found/pca/orange/splash_latent_viz.svg"
    plot_pca(dump_file_paths, output_path)