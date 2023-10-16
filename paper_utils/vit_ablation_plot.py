# Script to plot the results of the ViT ablation study
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
    data['Model'] = df['model_type']
    return data

def plot_vit_ablation(recalls, color_dict, output_path):
    "Plot the results of the layer ablation study and save the plot as pdf to `output_path`"
    fig, ax = plt.subplots()
    font_dict = {'size': 14, 'fontweight': 'bold'}
    for key in recalls.keys():
        # Read the csv file
        data = recalls[key]
        # Plot the results
        ax.plot([21, 86, 300, 1100], data, linestyle='--', marker='o', color=color_dict[key], label=key)
    ax.set_title('DINOv2', fontdict={'size': 16, 'fontweight': 'bold'})
    ax.set_xlabel('Number of ViT Parameters (M)', fontdict=font_dict)
    ax.set_ylabel('Recall@1 (%)', fontdict=font_dict)
    ax.legend(fontsize=12)
    ax.grid(linestyle='--')
    # Replace the xticks with custom labels
    ax.set_xticks([21, 86, 300, 1100])
    ax.set_xticklabels(['21\nS', '86\nB', '300\nL', '1100\nG'])
    # Increase the font size of the tick labels
    ax.tick_params(axis='both', which='major', labelsize=12)
    # Save the plot
    fig.savefig(output_path, bbox_inches='tight')
    print(f"Saved plot to {output_path}")

def load_data():
    "Function to load the data"
    best_layers = [10, 10, 20, 31]
    recalls = {}
    recalls['Baidu Mall'] = []
    recalls['Oxford'] = []
    oxford_csv_path = './data/ablations/vit_and_layer/dinov2_vit_oxford.csv'
    oxford_data = read_csv(oxford_csv_path)
    baidu_csv_path = './data/ablations/vit_and_layer/dinov2_vit_baidu.csv'
    baidu_data = read_csv(baidu_csv_path)
    for vit_idx, size in enumerate(['s', 'b', 'l', 'g']):
        model_name = f'dinov2_vit{size}14'
        oxford_idx = np.where(oxford_data['Model'] == model_name)[0][0]
        baidu_idx = np.where(baidu_data['Model'] == model_name)[0][0]
        recalls['Oxford'].append(oxford_data['Recall@1'][oxford_idx])
        recalls['Baidu Mall'].append(baidu_data['Recall@1'][baidu_idx])
    return recalls

if __name__ == '__main__':
    # Set plt to seaborn style
    plt.style.use('seaborn-v0_8-white')
    # Define color dict for ablation datasets
    color_dict = {'Baidu Mall': (0, 0.3, 0.8), 'Oxford': (0, 0.5, 0)}
    # Load the data
    recalls = load_data()
    # Plot the ViT Ablation Analysis
    output_path = '/ocean/projects/cis220039p/nkeetha/data/vlm/found/ablation_plots/vit_ablation.pdf'
    plot_vit_ablation(recalls, color_dict, output_path)