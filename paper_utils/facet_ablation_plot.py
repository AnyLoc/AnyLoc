# Script to plot the results of the facet ablation study
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def read_csv(filename, data):
    "Function to read csv file and convert the `R@1` column to numpy arrays"
    df = pd.read_csv(filename)
    for facet, r1 in zip(df['Facet'], df['R@1']):
        data[facet].append(float(r1)*100)
    return data

def plot_facet_ablation(dataset_names, input_csvs, color_dict, model_name, output_path):
    "Plot the results of the facet ablation study as a bar plot and save the plot as pdf to `output_path`"
    fig, ax = plt.subplots(layout='constrained')
    font_dict = {'size': 14, 'fontweight': 'bold'}
    # Define color for each facet bar
    facet_dict = {'Query': 'tab:brown', 'Key': 'tab:pink', 'Value': 'tab:orange', 'Token': 'tab:purple'}

    # Define variables for label locations
    x = np.array([0, 0.5])  # the label locations
    width = 0.1  # the width of the bars
    multiplier = 0

    # Loop over the datasets
    data = {}
    for facet in facet_dict.keys():
        data[facet] = []
    for dataset_name, input_csv in zip(dataset_names, input_csvs):
        # Read the csv file
        data = read_csv(input_csv, data)
    
    # Loop over the data and plot the bar
    for facet, recall in data.items():
        offset = width * multiplier
        bar_spacing = 0.02
        ax.bar(x + offset, recall, width-bar_spacing, label=facet, color=facet_dict[facet])
        multiplier += 1

    # Set the dataset name as the x-tick label using the color dict
    ax.set_xticks(x + 1.5*width, dataset_names, fontsize=14, fontweight='bold')
    # Set tick color
    for ticklabel in plt.gca().get_xticklabels():
        ticklabel.set_color(color_dict[ticklabel.get_text()])
    ax.set_ylabel('Recall@1 (%)', fontdict=font_dict)
    ax.set_title(model_name, fontdict={'size': 16, 'fontweight': 'bold'})
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(axis='y', linestyle='--')
    ax.set_ylim([0, 100])
    # Increase the font size of the y-axis tick labels
    ax.tick_params(axis='y', which='major', labelsize=14)
    # Save the plot
    fig.savefig(output_path, bbox_inches='tight')
    print(f"Saved plot to {output_path}")

def arg_parser():
    parser = argparse.ArgumentParser(description='Plot the results of the facet ablation study')
    parser.add_argument('-d', '--dataset_names', nargs='+', type=str, required=True, help='Names of the dataset')
    parser.add_argument('-i', '--input_csvs', nargs='+', type=str, required=True, help='Path to the input csv file')
    parser.add_argument('-m', '--model_name', type=str, required=True, help='Name of the model')
    parser.add_argument('-o', '--output_path', type=str, required=True, help='Path to save the plot')
    return parser.parse_args()

if __name__ == '__main__':
    args = arg_parser()
    dataset_names = args.dataset_names
    input_csvs = args.input_csvs
    # Set plt to seaborn style
    plt.style.use('seaborn-v0_8-white')
    # Define color dict for ablation datasets
    color_dict = {'Baidu Mall': (0, 0.3, 0.8), 'Oxford': (0, 0.5, 0), 'Pitts-30k': (0, 0.5, 0)}
    plot_facet_ablation(dataset_names, input_csvs, color_dict, args.model_name, args.output_path)