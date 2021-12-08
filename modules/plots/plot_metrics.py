import matplotlib.pyplot as plt
from modules.plots.core import autolabels, get_bars_positions
import numpy as np
import math
def plot_metrics(
    title,
    bars_data,
    bar_labels,
    bar_colors,
    figure_labels,
    algo_labels,
    bar_width,
    espacamento_entre_barras,
    espaco_entre_grupos,
    label_pad_bar_x,
    label_pad_bar_y,
    algo_label_fontsize,
    label_fontsize,
    legend_fontsize = 14,
    bar_group_font_size = 14,
    subaxes_title_fontsize = 20, 
    fig = None,
    show_legend = True,
    algo_labels_x = -5,
    algo_labels_dis = 0):
    
    number_columns = len(algo_labels)*2
    number_groups = len(bar_labels)

    if not fig:
        fig = plt.figure()
        
    axes = fig.subplots(1, len(figure_labels))
    
    for i in range(len(axes)):
        groups = get_bars_positions(number_groups, number_columns, bar_width, espaco_entre_grupos, espacamento_entre_barras)
        bars = []
        for j  in range(number_groups):
            bar = axes[i].barh(
                groups[j], 
                bars_data[i][j], 
                bar_width,
                color=bar_colors[j], 
                label=bar_labels[j],)

            bars.append(bar)

        axes[i].set_xticks(np.arange(0, 140, 20))
        
        yticks = [r - bar_width - espacamento_entre_barras - 0.3 for r in groups[0]]
        axes[i].set_yticks(yticks)
        axes[i].set_yticklabels(np.full((int(number_columns / 2), 2), ['High', 'Low']).flatten(), fontsize=bar_group_font_size)
        axes[i].set_title(figure_labels[i], fontsize=subaxes_title_fontsize)
        autolabels(axes[i], bars, label_pad_bar_x, label_pad_bar_y, fontsize=label_fontsize)

        for j in range(len(algo_labels)):
            y = yticks[j] + ((yticks[j] + yticks[j + 1]) / 2) + algo_labels_dis
            axes[i].text(algo_labels_x, y, algo_labels[j], rotation='vertical', fontsize=algo_label_fontsize)
    

    if show_legend:
        axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.04), fancybox=True, shadow=True, ncol=math.ceil(len(bar_labels) / 2), fontsize=legend_fontsize)
    
    
    fig.suptitle(title, fontsize=18, y=0.91)

    return fig, axes