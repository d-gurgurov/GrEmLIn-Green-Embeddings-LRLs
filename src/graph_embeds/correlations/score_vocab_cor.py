# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

# Function to create dataset
def create_dataset(ppmi_type, task_type):
    if ppmi_type == "single" and task_type == "sid":
        data = data_ppmi_single_sid
    elif ppmi_type == "all" and task_type == "sid":
        data = data_ppmi_all_sid
    elif ppmi_type == "single" and task_type == "sa":
        data = data_ppmi_single_sa
    elif ppmi_type == "all" and task_type == "sa":
        data = data_ppmi_all_sa
    elif ppmi_type == "single" and task_type == "xnli":
        data = data_ppmi_single_xnli
    elif ppmi_type == "all" and task_type == "xnli":
        data = data_ppmi_all_xnli
    elif ppmi_type == "single" and task_type == "simlex":
        data = data_ppmi_single_simlex
    elif ppmi_type == "all" and task_type == "simlex":
        data = data_ppmi_all_simlex
    
    df = pd.DataFrame(data)
    df['Improvement'] = df['G+P'] - df['GloVe']
    return df

data_ppmi_single_sid = {
        'Lang': ['af', 'am', 'az', 'be', 'bn', 'bg', 'ku', 'cy', 'da', 'el', 'eo', 'es', 'gd', 'ga', 'gl', 'gu', 'ht', 'ha', 'he', 'hy', 'is', 'ja', 'kn', 'ka', 'kk', 'km', 'ky', 'lo', 'lt', 'lv', 'ml', 'mr', 'mk', 'my', 'ne', 'pa', 'ps', 'ro', 'sa', 'si', 'sk', 'sl', 'so', 'su', 'sw', 'ta', 'tl', 'ug', 'uk', 'ur', 'uz', 'xh', 'yi', 'ms', 'yo', 'qu', 'wo'],
        'GloVe': [0.454, 0.515, 0.698, 0.580, 0.604, 0.645, 0.110, 0.564, 0.446, 0.531, 0.504, 0.589, 0.230, 0.411, 0.530, 0.544, 0.392, 0.421, 0.759, 0.551, 0.423, 0.511, 0.652, 0.684, 0.664, 0.125, 0.593, 0.183, 0.713, 0.737, 0.651, 0.627, 0.611, 0.228, 0.542, 0.474, 0.351, 0.561, 0.206, 0.678, 0.670, 0.628, 0.363, 0.467, 0.531, 0.710, 0.65, 0.556, 0.682, 0.420, 0.587, 0.388, 0.341, 0.694, 0.199, 0.175, 0.058],
        'G+P': [0.560, 0.472, 0.668, 0.597, 0.617, 0.711, 0.098, 0.608, 0.743, 0.712, 0.567, 0.572, 0.397, 0.532, 0.663, 0.589, 0.496, 0.489, 0.739, 0.609, 0.490, 0.523, 0.581, 0.689, 0.647, 0.117, 0.571, 0.185, 0.775, 0.732, 0.574, 0.608, 0.694, 0.207, 0.563, 0.521, 0.431, 0.686, 0.261, 0.613, 0.667, 0.715, 0.403, 0.446, 0.553, 0.651, 0.707, 0.583, 0.722, 0.627, 0.529, 0.320, 0.384, 0.738, 0.211, 0.167, 0.139],
        'Common_Vocab': [9177, 1105, 7215, 7623, 3962, 92436, 3762, 7774, 38095, 19710, 59476, 14815, 6415, 13871, 29654, 3198, 1557, 671, 16032, 14951, 27007, 2607, 2181, 17869, 8292, 2654, 2234, 269010, 12485, 17450, 4092, 3211, 21692, 3189, 2650, 2282, 847, 25704, 3336, 943, 14694, 45153, 533, 1236, 6425, 4596, 12563, 764, 16397, 4662, 3229, 1650, 5177, 34022, 558, 2056, 999]
    }

data_ppmi_all_sid = {
        'Lang': ['af', 'am', 'az', 'be', 'bn', 'bg', 'ku', 'cy', 'da', 'el', 'eo', 'es', 'gd', 'ga', 'gl', 'gu', 'ht', 'ha', 'he', 'hy', 'is', 'ja', 'kn', 'ka', 'kk', 'km', 'ky', 'lo', 'lt', 'lv', 'ml', 'mr', 'mk', 'my', 'ne', 'pa', 'ps', 'ro', 'sa', 'si', 'sk', 'sl', 'so', 'su', 'sw', 'ta', 'tl', 'ug', 'uk', 'ur', 'uz', 'xh', 'yi', 'ms', 'yo', 'qu', 'wo'],
        'GloVe': [0.454, 0.515, 0.698, 0.580, 0.604, 0.645, 0.110, 0.564, 0.446, 0.531, 0.504, 0.589, 0.230, 0.411, 0.530, 0.544, 0.392, 0.421, 0.759, 0.551, 0.423, 0.511, 0.652, 0.684, 0.664, 0.125, 0.593, 0.183, 0.713, 0.737, 0.651, 0.627, 0.611, 0.228, 0.542, 0.474, 0.351, 0.561, 0.206, 0.678, 0.670, 0.628, 0.363, 0.467, 0.531, 0.710, 0.650, 0.556, 0.682, 0.420, 0.587, 0.388, 0.341, 0.694, 0.199, 0.175, 0.058],
        'G+P': [0.590, 0.555, 0.711, 0.621, 0.681, 0.723, 0.095, 0.694, 0.717, 0.702, 0.588, 0.605, 0.418, 0.547, 0.699, 0.631, 0.523, 0.546, 0.784, 0.644, 0.534, 0.541, 0.658, 0.689, 0.690, 0.109, 0.590, 0.180, 0.797, 0.742, 0.608, 0.676, 0.719, 0.163, 0.605, 0.565, 0.493, 0.704, 0.251, 0.695, 0.725, 0.734, 0.459, 0.526, 0.637, 0.703, 0.709, 0.622, 0.745, 0.643, 0.592, 0.341, 0.453, 0.769, 0.264, 0.153, 0.122],
        'Common_Vocab': [85270, 14217, 80761, 73750, 38221, 368232, 32499, 57522, 450290, 197647, 161634, 163666, 24430, 65169, 215868, 24575, 13304, 33824, 153731, 60756, 143567, 41471, 24783, 96066, 64494, 34014, 29915, 373012, 200404, 183088, 38864, 33552, 93121, 24319, 21479, 16068, 15904, 366809, 12101, 27536, 93121, 734, 459, 467, 531, 710, 650, 556, 682, 420, 587, 388, 341, 694, 199, 175, 58],
    }

data_ppmi_single_sa = {
    "Lang": ["am", "su", "sw", "si", "ka", "ne", "ug", "yo", "ur", "mk", "mr", "bn", "te", "uz", "az", "bg", "sl", "lv", "sk", "ro", "he", "cy", "da"],
    "G+P": [0.86, 0.822, 0.701, 0.85, 0.87, 0.674, 0.811, 0.709, 0.746, 0.711, 0.905, 0.881, 0.808, 0.806, 0.746, 0.801, 0.779, 0.787, 0.806, 0.85, 0.824, 0.789, 0.908],
    "GloVe": [0.881, 0.798, 0.68, 0.848, 0.861, 0.643, 0.746, 0.721, 0.676, 0.716, 0.903, 0.875, 0.806, 0.808, 0.744, 0.786, 0.749, 0.783, 0.756, 0.805, 0.788, 0.77, 0.863],
    "Common_Vocab": [1105, 1236, 6425, 943, 17869, 2650, 764, 558, 4662, 21692, 3211, 3962, 12563, 3229, 7215, 92436, 45153, 17450, 14694, 25704, 16032, 7774, 38095],
    }

data_ppmi_all_sa = {
    "Lang": ["am", "su", "sw", "si", "ka", "ne", "ug", "yo", "ur", "mk", "mr", "bn", "te", "uz", "az", "bg", "sl", "lv", "sk", "ro", "he", "cy", "da"],
    "GloVe": [0.881, 0.798, 0.68, 0.848, 0.861, 0.643, 0.746, 0.721, 0.676, 0.716, 0.903, 0.875, 0.806, 0.808, 0.744, 0.786, 0.749, 0.783, 0.756, 0.805, 0.788, 0.77, 0.863],
    "G+P": [0.88, 0.812, 0.714, 0.857, 0.861, 0.688, 0.811, 0.738, 0.745, 0.7, 0.902, 0.878, 0.817, 0.806, 0.745, 0.805, 0.788, 0.787, 0.805, 0.847, 0.822, 0.801, 0.903],
    "Common_Vocab": [14217, 26068, 59906, 27536, 96066, 21479, 4798, 5254, 44530, 93121, 33552, 38221, 42653, 37704, 80761, 368232, 229429, 183088, 268576, 366809, 153731, 57522, 450290]
    }

data_ppmi_single_xnli = {
    'Lang': ['bg', 'el', 'sw', 'th', 'ur'],
    'GloVe': [0.441, 0.456, 0.438, 0.284, 0.44],
    'G+P': [0.481, 0.496, 0.466, 0.292, 0.473],
    'Common_Vocab': [92436, 19710, 6425, 45975, 4662],
    }

data_ppmi_all_xnli = {
    'Lang': ['bg', 'el', 'sw', 'th', 'ur'],
    'GloVe': [0.441, 0.456, 0.438, 0.284, 0.44],
    'G+P': [0.477, 0.488, 0.468, 0.3, 0.471],
    'Common_Vocab': [368232, 197647, 59906, 238502, 44530]
    }

data_ppmi_single_simlex = {
    'Lang': ['et', 'he', 'cy', 'sw'],
    'GloVe': [0.341, 0.336, 0.276, 0.24],
    'G+P': [0.452, 0.436, 0.366, 0.319],
    'Common_Vocab': [14815, 16032, 7774, 6425],
    }

data_ppmi_all_simlex = {
    'Lang': ['bg', 'el', 'sw', 'th'],
    'GloVe': [0.341, 0.336, 0.276, 0.24],
    'G+P': [0.422, 0.429, 0.357, 0.324],
    'Common_Vocab': [163666, 153731, 57522, 59906]
    }

# Set up the figure
fig, axs = plt.subplots(2, 4, figsize=(24, 12))
plt.subplots_adjust(hspace=0.3, wspace=0.2)

# Color palette
colors = sns.color_palette("viridis", 2)

# Function to create scatter plot
def create_scatter(ax, df, title):
    ax.clear() 
    sns.scatterplot(x='Common_Vocab', y='Improvement', data=df, s=60, color=colors[0], alpha=0.7, ax=ax)
    ax.set_title(title, fontsize=14)
    ax.set_xscale('log')
    
    x = df['Common_Vocab']
    y = df['Improvement']
    z = np.polyfit(np.log(x), y, 1)
    p = np.poly1d(z)
    ax.plot(x, p(np.log(x)), color=colors[1], linewidth=2)
    
    pearson_corr, _ = pearsonr(df['Common_Vocab'], df['Improvement'])
    spearman_corr, _ = spearmanr(df['Common_Vocab'], df['Improvement'])
    ax.annotate(f'Pearson r: {pearson_corr:.3f}\nSpearman ρ: {spearman_corr:.3f}', 
                xy=(0.05, 0.95), xycoords='axes fraction', fontsize=8, 
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

# Function to create boxplot
def create_boxplot(ax, df):
    df['Common_Vocab_Quantile'] = pd.qcut(df['Common_Vocab'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    sns.boxplot(x='Common_Vocab_Quantile', y='Improvement', data=df, palette="viridis", ax=ax)
    
    sns.pointplot(x='Common_Vocab_Quantile', y='Improvement', data=df, color='red', markers='D', scale=0.5, ax=ax)
    
    medians = df.groupby('Common_Vocab_Quantile')['Improvement'].median()
    for i, median in enumerate(medians):
        ax.text(i, median, f'{median:.3f}', horizontalalignment='center', size='x-small', color='white', weight='semibold')

# Create plots
tasks = ['SID', 'SA']
configs = ['Single', 'All']

for i, (task, config) in enumerate([(t, c) for t in tasks for c in configs]):
    df = create_dataset(config.lower(), task.lower())
    create_scatter(axs[0, i], df, f'{task} Task - {config}')
    create_boxplot(axs[1, i], df)
    
    # Remove x-axis labels 
    axs[0, i].set_xlabel('')
    axs[1, i].set_xlabel('')
    # Remove y-axis labels
    axs[0, i].set_ylabel('')
    axs[1, i].set_ylabel('')

# Add shared x-axis labels
fig.text(0.5, 0.04, 'Common Vocabulary Size', ha='center', va='center', fontsize=12)

# Add shared y-axis label
fig.text(0.075, 0.5, 'Performance Improvement', ha='center', va='center', rotation='vertical', fontsize=12)

plt.tight_layout()
# Adjust the layout to make room for shared labels
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.98, top=0.94)

plt.savefig('composite_plots.png', dpi=300, bbox_inches='tight')
plt.close()