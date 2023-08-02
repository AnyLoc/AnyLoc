# Script to plot the radar chart for performance on diverse datasets
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


def recall_data():
    # Recall@1 VPR Performance
    data_labels = ['Indoor', 'Outdoor', 'Day Vs Night', 'SubT', 'Aerial', 'Underwater', 'VP Shift', 'Opp VP']
    data = {'NetVLAD': [57.73, 65.63, 57.65, 37.02, 28.89, 25.74, 48.52, 31.25],
            'CosPlace': [58.90, 93.71, 75.39, 27.735, 33.22, 20.79, 46.73, 18.49],
            'MixVPR': [73.24, 93.74, 81.78, 27.44, 39.59, 25.74, 55.42, 29.09],
            'CLIP': [52.63, 50.74, 45.47, 34.83, 46.94, 25.74, 49.19, 37.30],
            'DINO': [62.87, 43.69, 52.01, 43.84, 55.43, 27.72, 47.48, 48.48],
            'DINOv2': [60.84, 65.58, 57.70, 34.075, 63.43, 24.75, 57.59, 47.13],
            'FoundGeM (DINOv2)': [67.23, 78.37, 77.57, 56.16, 57.37, 14.85, 55.15, 62.79],
            # 'FoundVLAD (DINO)': [72.89, 76.98, 71.62, 52.725, 47.85, 41.58, 54.01, 46.89],
            # 'FoundVLAD (DINOv2)': [78.01, 91.23, 84.72, 63.43, 62.89, 34.65, 72.41, 61.54],
            'FoundVLAD (DINO)': [73.32, 83.48, 79.11, 52.725, 51.96, 41.58, 54.16, 49.70],
            'FoundVLAD (DINOv2)': [78.58, 94.26, 86.49, 63.43, 76.24, 34.65, 76.54, 67.64],}
    # Rearrange order of values
    for key in data.keys():
        data[key] = [data[key][i] for i in [0, 1, 2, 4, 3, 5, 7, 6]]
    data_labels = [data_labels[i] for i in [0, 1, 2, 4, 3, 5, 7, 6]]
    return data_labels, data

def icon_data():
    # Icon paths
    icon_info = {'Indoor': ['./data/icons/indoor.png', 0.14, (0.5, 1.04)],
                 'Outdoor': ['./data/icons/outdoor.png', 0.18, (0.08, 0.9)],
                 'Day Vs Night': ['./data/icons/day_night.png', 0.15, (-0.075, 0.5)],
                 'SubT': ['./data/icons/subt.png', 0.08, (0.5, -0.05)],
                 'Aerial': ['./data/icons/aerial.png', 0.28, (0.075, 0.1)],
                 'Underwater': ['./data/icons/underwater.png', 0.06, (0.875, 0.095)],
                 'VP Shift': ['./data/icons/viewpoint.png', 0.04, (0.9, 0.87)],
                 'Opp VP': ['./data/icons/opp_viewpoint.png', 0.04, (1.06, 0.5)]}
    return icon_info

def plot_single_radar_chart():
    # Define Args
    baselines = ['MixVPR', 'NetVLAD']
    found_baselines = ['FoundVLAD (DINOv2)']
    methods = found_baselines + baselines
    output_path = '/ocean/projects/cis220039p/nkeetha/data/vlm/found/radar_charts/orange/splash_radar.pdf'

    # Define Radar chart parameters
    N = 8
    theta = radar_factory(N, frame='polygon')

    spoke_labels, data = recall_data()
    icon_info = icon_data()

    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)

    colors = {'NetVLAD': 'chocolate', 'CosPlace': 'blueviolet', 'MixVPR': 'orangered',
              'FoundGeM (DINOv2)': 'chocolate', 'FoundVLAD (DINO)': 'orangered', 'FoundVLAD (DINOv2)': 'darkorange',
              'Found (DINO)': 'orangered', 'Found (DINOv2)': 'darkorange'}
    # Plot the radar chart for each set of methods
    ax.set_rgrids([20, 40, 60, 80], fontsize=14)
    ax.grid(linestyle='--')
    # ax.set_title(method_set, weight='bold', size=14, pad=32, horizontalalignment='center', verticalalignment='center')
    for method_name in methods:
        method_scores = data[method_name]
        if method_name in baselines:
            ax.plot(theta, method_scores, color=colors[method_name], linestyle='dashed')
        else:
            ax.plot(theta, method_scores, color=colors[method_name])
        ax.fill(theta, method_scores, facecolor=colors[method_name], alpha=0.25, label='_nolegend_')
    ax.set_varlabels(['']*N)

    ## Add Icons as axis labels
    for icon_label in icon_info.keys():
        icon_path, zoom, xy_pos = icon_info[icon_label]
        icon = plt.imread(icon_path)
        imagebox = OffsetImage(icon, zoom=zoom)
        ab = AnnotationBbox(imagebox, xy_pos, frameon=False, xycoords='axes fraction', boxcoords="axes fraction")
        ax.add_artist(ab)

    # add legend relative to top-left plot
    ax.legend(methods, loc=(0.63, .94), labelspacing=0.4, fontsize=14, labelcolor=[colors[label] for label in methods], prop=dict(weight='bold'))
    # Legend Position: Bottom Center of Radar Chart
    # ax.legend(methods, loc='upper left', bbox_to_anchor=(0.32, 0.22), labelspacing=0.4, fontsize=10, labelcolor=[colors[label] for label in methods], prop=dict(weight='bold'))
    # Legend Position: Top Right of Radar Chart
    # ax.legend(methods, loc=(0.95, .90), labelspacing=0.4, fontsize=10, labelcolor=[colors[label] for label in methods], prop=dict(weight='bold'))

    # fig.text(0.55, 0.925, 'VPR Performance across Diverse Domains',
    #          horizontalalignment='center', color='black', weight='bold',
    #          size=16)

    # Save the plot
    fig.savefig(output_path, bbox_inches='tight', dpi=400)
    print(f"Saved plot to {output_path}")

def plot_multi_radar_charts():
    # Define Args
    baselines = ['MixVPR', 'CosPlace', 'NetVLAD']
    found_baselines = ['FoundVLAD (DINOv2)', 'FoundGeM (DINOv2)']
    methods = {'Baselines': baselines, 'Found Baselines': found_baselines}
    output_path = '/ocean/projects/cis220039p/nkeetha/data/vlm/found/radar_charts/diverse_performance.pdf'

    # Define Radar chart parameters
    N = 8
    theta = radar_factory(N, frame='polygon')

    spoke_labels, data = recall_data()
    icon_info = icon_data()

    fig, axs = plt.subplots(figsize=(12, 6), nrows=1, ncols=2, subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(wspace=0.275, hspace=0.20, top=0.85, bottom=0.05)

    colors = {'NetVLAD': 'deeppink', 'CosPlace': 'blueviolet', 'MixVPR': 'purple',
              'FoundGeM (DINOv2)': 'chocolate', 'FoundVLAD (DINO)': 'orangered', 'FoundVLAD (DINOv2)': 'darkorange',
              'Found (DINO)': 'orangered', 'Found (DINOv2)': 'darkorange'}
    # Plot the radar chart for each set of methods
    for method_set, ax in zip(methods.keys(), axs.flat):
        ax.set_rgrids([20, 40, 60, 80], fontsize=10)
        ax.grid(linestyle='--')
        ax.set_title(method_set, weight='bold', size=14, pad=32, horizontalalignment='center', verticalalignment='center')
        for method_name in methods[method_set]:
            method_scores = data[method_name]
            ax.plot(theta, method_scores, color=colors[method_name])
            ax.fill(theta, method_scores, facecolor=colors[method_name], alpha=0.25, label='_nolegend_')
        ax.set_varlabels(['']*N)

        ## Add Icons as axis labels
        for icon_label in icon_info.keys():
            icon_path, zoom, xy_pos = icon_info[icon_label]
            icon = plt.imread(icon_path)
            imagebox = OffsetImage(icon, zoom=zoom)
            ab = AnnotationBbox(imagebox, xy_pos, frameon=False, xycoords='axes fraction', boxcoords="axes fraction")
            ax.add_artist(ab)

    # add legend relative to top-left plot
    axs[0].legend(baselines, loc=(0.9625, .83), labelspacing=0.2, fontsize=10, labelcolor=[colors[label] for label in baselines], prop=dict(weight='bold'))
    axs[1].legend(found_baselines, loc=(-0.3125, .95), labelspacing=0.2, fontsize=10, labelcolor=[colors[label] for label in found_baselines], prop=dict(weight='bold'))

    fig.text(0.5125, 0.925, 'VPR Performance across Diverse Domains',
             horizontalalignment='center', color='black', weight='bold',
             size=16)

    # Save the plot
    fig.savefig(output_path, bbox_inches='tight', dpi=400)
    print(f"Saved plot to {output_path}")

if __name__ == '__main__':
    ## Set plt to seaborn style
    plt.style.use('seaborn-v0_8-white')
    
    ## Single radar chart
    plot_single_radar_chart()

    ## Multiple radar charts
    # plot_multi_radar_charts()