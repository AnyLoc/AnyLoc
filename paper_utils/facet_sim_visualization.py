# Script to plot the facet similarity visualization
"""
    This script is not dependent on library and reads everything from 
    the cache (can run anywhere).
"""
import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_facet_sim(dump_file_paths: list, output_path: str, layer_ablation: bool = False, layer_nums: list = None):
    # Define colormap for facets
    colormap = {"key": "tab:pink", "query": "tab:brown", "value": "tab:orange", "token": "tab:purple", "prompt": "red"}
    markermap = {"key": "^", "query": "s", "value": "o", "token": "d"}
    # Define Marker properties
    mp = {"ms": 10, "mew": 2, "mec": 'white', "alpha": 0.7}
    # Init subplots
    if layer_ablation:
        fig, ax = plt.subplots(figsize=(18, 8), nrows=len(dump_file_paths), ncols=6)
    else:
        fig, ax = plt.subplots(figsize=(18, 6.5), nrows=len(dump_file_paths), ncols=6, gridspec_kw={'height_ratios': [1, 1.2, 1.2]})
    fig.subplots_adjust(wspace=0.055, hspace=0.055)
    font_dict = {'size': 14, 'fontweight': 'bold'}
    # Loop over the dump files
    for i, dump_file in enumerate(dump_file_paths):
        # Load the dump
        dump_file = dump_file_paths[i]
        assert os.path.isfile(dump_file)
        fname = os.path.realpath(os.path.expanduser(dump_file))
        data = joblib.load(fname)
        pix_loc = data["pix_loc"] # (x=width=right, y=height=down)
        simg_np: np.ndarray = data["source"]
        timg_np: np.ndarray = data["target"]
        sims = data["similarities"] # Dict[str, np.ndarray] [-1 to 1]
        mlocs = data["max"] # Dict[str, np.ndarray] [y, x] loc
        if i in [1, 2] and not layer_ablation:
            # Resize the images to be non-squashed
            simg_np = cv2.resize(simg_np, (512, 384), interpolation=cv2.INTER_CUBIC)
            timg_np = cv2.resize(timg_np, (512, 384), interpolation=cv2.INTER_CUBIC)
            pix_loc = (pix_loc[0], pix_loc[1]*384/320)
            for sim_key in sims.keys():
                sims[sim_key] = cv2.resize(sims[sim_key], (512, 384), interpolation=cv2.INTER_CUBIC)
                mlocs[sim_key][0] = mlocs[sim_key][0]*384/320
        # Plot the source image
        ax[i, 0].imshow(simg_np)
        # Plot the prompt pixel location
        ax[i, 0].plot(*pix_loc, 'o', c=colormap["prompt"], **mp)
        # Plot the target image
        ax[i, 1].imshow(timg_np)
        # Plot the facet locations on the target image
        for facet in ["Key", "Query", "Value", "Token"]:
            ax[i, 1].plot(mlocs[facet.lower()][1], mlocs[facet.lower()][0], 
                          markermap[facet.lower()], label=facet, c=colormap[facet.lower()], **mp)
        # Set ticks to be off
        ax[i, 0].set_yticklabels([])
        ax[i, 0].set_xticklabels([])
        ax[i, 0].set_xticks([])
        ax[i, 0].set_yticks([])
        ax[i, 1].set_yticklabels([])
        ax[i, 1].set_xticklabels([])
        ax[i, 1].set_xticks([])
        ax[i, 1].set_yticks([])
        for key, spine in ax[i, 0].spines.items():
            spine.set_visible(False)
        for key, spine in ax[i, 1].spines.items():
            spine.set_visible(False)
        # Plot the similarity maps for each facet
        for facet_idx, facet in enumerate(["key", "query", "value", "token"]):
            ax[i, facet_idx+2].imshow((((sims[facet] + 1) / 2) * 255).astype(np.uint8), vmin=0, vmax=255, cmap="jet")
            ax[i, facet_idx+2].axis('off')
        # Set titles at bottom of the last row
        if i == len(dump_file_paths) - 1:
            y_loc = 0
            pad = -23
            ax[i, 0].set_title("Database Image", y=y_loc, pad=pad, fontdict=font_dict)
            ax[i, 1].set_title("Query Image", y=y_loc, pad=pad, fontdict=font_dict)
        if layer_ablation:
            # Add layer names to the left
            ax[i, 0].set_ylabel(f"Layer {layer_nums[i]}", fontdict=font_dict)
    # Add legend aligning with the last row labels
    if layer_ablation:
        ax[-1, 1].legend(loc='upper left', bbox_to_anchor=(1.2, 0), ncol=4, columnspacing=6.25, prop={'weight': 'bold', 'size': 14})
    else:
        ax[-1, 1].legend(loc='upper left', bbox_to_anchor=(1.2, 0), ncol=4, columnspacing=6.05, prop={'weight': 'bold', 'size': 14})
    # Save the plot
    fig.savefig(output_path, bbox_inches='tight', dpi=400)
    print(f"Saved plot to {output_path}")

def dump_file_data_sim():
    "Dump File paths for facet similarity visualization"
    # dump_fps = ['/ocean/projects/cis220039p/nkeetha/data/vlm/found/facet_similarity/facet-sim/sense_baidu-mall/I455Q-457Q_Px410_Py115.gz',
    #             '/ocean/projects/cis220039p/nkeetha/data/vlm/found/facet_similarity/facet-sim/hush-puppies-logo-text_baidu-mall/I173Q-167Q_Px250_Py110.gz',
    #             '/ocean/projects/cis220039p/nkeetha/data/vlm/found/facet_similarity/facet-sim/day-night_Oxford/I1D-1Q_Px555_Py200.gz']
    dump_fps = ['/ocean/projects/cis220039p/nkeetha/data/vlm/found/facet_similarity/facet-sim/sense_baidu-mall/I455Q-457Q_Px410_Py115.gz',
                '/ocean/projects/cis220039p/nkeetha/data/vlm/found/facet_similarity/facet-sim/building_VPAir/I122Q-122D_Px226_Py280.gz',
                '/ocean/projects/cis220039p/nkeetha/data/vlm/found/facet_similarity/facet-sim/bench_viewpoint_change-laurel_caverns/I54Q-56D_Px370_Py160.gz']
    return dump_fps

def dump_file_data_ablation():
    "Dump File paths for facet similarity layer ablation visualization"
    dump_fps = ['/ocean/projects/cis220039p/nkeetha/data/vlm/found/facet_similarity/layers_day-night_Oxford/L7/I1D-1Q_Px555_Py200.gz',
                '/ocean/projects/cis220039p/nkeetha/data/vlm/found/facet_similarity/layers_day-night_Oxford/L15/I1D-1Q_Px555_Py200.gz',
                '/ocean/projects/cis220039p/nkeetha/data/vlm/found/facet_similarity/layers_day-night_Oxford/L31/I1D-1Q_Px555_Py200.gz',
                '/ocean/projects/cis220039p/nkeetha/data/vlm/found/facet_similarity/layers_day-night_Oxford/L39/I1D-1Q_Px555_Py200.gz']
    layer_nums = [7, 15, 31, 39]
    return dump_fps, layer_nums

if __name__ == '__main__':
    ## Set plt to seaborn style
    plt.style.use('seaborn-v0_8-white')
    
    ## Facet Similarity Visualization
    dump_file_paths = dump_file_data_sim()
    output_path = "/ocean/projects/cis220039p/nkeetha/data/vlm/found/facet_similarity/facet_sim.pdf"
    plot_facet_sim(dump_file_paths, output_path)

    ## Facet Similarity Layer Ablation Visualization
    dump_file_paths, layer_nums = dump_file_data_ablation()
    output_path = "/ocean/projects/cis220039p/nkeetha/data/vlm/found/facet_similarity/facet_sim_layer_ablation.pdf"
    plot_facet_sim(dump_file_paths, output_path, layer_ablation=True, layer_nums=layer_nums)