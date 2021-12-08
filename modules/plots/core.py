def autolabel(ax, rects, pad_x, pad_y, fontsize):
    for rect in rects:
        width = rect.get_width()
        height = rect.get_height()
        ypos = rect.get_y() + height/2
        # ax.text(width + 6.0, ypos - .23, width, ha='center', va='bottom', rotation=0, fontsize=10) 
        ax.text(width  + pad_x, ypos  + pad_y, width, ha='center', va='bottom', rotation=0, fontsize=fontsize) 


def autolabels(ax, labels, pad_x, pad_y, fontsize=10):
    for label in labels:
        autolabel(ax, label, pad_x, pad_y, fontsize)


def get_bars_positions(
    num_groups, 
    num_columns,
    bar_width, 
    espacamento_entre_grupos, 
    espacamento_entre_barras):

    pos = []
    for i in range(num_groups):
        if i == 0:
            pos.append([(x * espacamento_entre_grupos + espacamento_entre_barras)
                        * -1 for x in range(num_columns)])
            continue

        pos.append([x - bar_width - espacamento_entre_barras for x in pos[-1]])

    return tuple(pos)
