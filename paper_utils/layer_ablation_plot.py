# Script to plot the results of the layer ablation study
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def read_csv(filename):
    "Function to read csv file and convert the `desc_layer` and `R@1` columns to numpy arrays"
    df = pd.read_csv(filename)
    df.sort_values("desc_layer", inplace=True)
    data = {}
    data['ViT Layer Number'] = np.array(df['desc_layer'])
    data['Recall@1'] = np.array(df['R@1'])*100
    return data

def plot_layer_ablation(dataset_names, input_csvs, color_dict, model_name, output_path):
    "Plot the results of the layer ablation study and save the plot as pdf to `output_path`"
    fig, ax = plt.subplots()
    font_dict = {'size': 14, 'fontweight': 'bold'}
    for dataset_name, input_csv in zip(dataset_names, input_csvs):
        # Read the csv file
        data = read_csv(input_csv)
        # Plot the results
        ax.plot(data['ViT Layer Number'], data['Recall@1'], linestyle='--', marker='o', color=color_dict[dataset_name], label=dataset_name)
    ax.set_xlabel('ViT Layer Number', fontdict=font_dict)
    ax.set_ylabel('Recall@1 (%)', fontdict=font_dict)
    ax.set_title(model_name, fontdict={'size': 16, 'fontweight': 'bold'})
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(linestyle='--')
    ax.set_ylim([0, 100])
    # Increase the font size of the tick labels
    ax.tick_params(axis='both', which='major', labelsize=12)
    # Save the plot
    fig.savefig(output_path, bbox_inches='tight')
    print(f"Saved plot to {output_path}")

def arg_parser():
    parser = argparse.ArgumentParser(description='Plot the results of the layer ablation study')
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
    plot_layer_ablation(dataset_names, input_csvs, color_dict, args.model_name, args.output_path)