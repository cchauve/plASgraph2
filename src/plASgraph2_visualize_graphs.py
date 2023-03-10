#!/usr/bin/env python
# coding: utf-8

import argparse
from ctypes import sizeof
import inspect
import sys
import os.path
from syslog import LOG_WARNING
from unicodedata import name
import pandas as pd
import numpy as np
import helpers
import create_graph
import networkx as nx


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib


# pairs length threshold, node size in nx, marker size in legend
NODE_SIZE_SETTINGS = [(100, 20, 6), (1000, 40, 9),
                      (10000, 80, 12), (1e10, 160, 15)]
# pairs label, node color
NODE_COLOR_SETTINGS = [("chromosome", "C0"), ("plasmid", "C1"), 
                        ("ambiguous", "black"), ("unlabeled", "gray")]
# how many interquartile ranges from median is start of outliers when coloring nodes
# traditional value 1.5 seems to be too large
IQR_COEFF = 1

def main(file_list, file_prefix, summary_file, selected_columns, output_dir):
    """Input consists of a list of files for a testing or training set (GFA files will be used)
    and a summary csv file with lines for all nodes in these graphs and various named colums. 

    Argument selected_columns contains a list of columns to be used in visualization. 
    The columns are separated by commas, and for each a separate drawing is made. 
    Two columns can be also separated by a colon and then they are used in a single drawing.

    Finally an output directory is given, where output png files will be written.
    """

    columns = selected_columns.split(",")
    assert len(columns) > 0

    # create dir if needed
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # read feature csv file, create index by node_id
    features_df = pd.read_csv(summary_file)
    features_df["id"] = features_df.apply(
        lambda x: helpers.get_node_id(x["sample"], x["contig"]), axis=1)
    features_df.set_index("id", inplace=True)

    graph_files = pd.read_csv(file_list, names=(
        'graph', 'csv', 'sample_id'), header=None)
    for idx, row in graph_files.iterrows():
        graph_file = row['graph']
        sample_id = row['sample_id']
        graph = create_graph.read_single_graph(
            file_prefix, graph_file, sample_id, delete_short=False)
        for column in columns:
            column_filename = column.replace(":", "_")
            output_filename = os.path.join(
                output_dir, f"{sample_id}_{column_filename}.png")
            draw_graph(graph, features_df, column, output_filename, sample_id)


def norm_value_range(x):
    x = np.array(x)
    q1, median, q3 = np.quantile(x, [0.25, 0.50, 0.75])
    iqr = q3 - q1
    low = max(x.min(), median - IQR_COEFF * iqr)
    high = min(x.max(), median + IQR_COEFF * iqr)
    if low == high:
        high = low + 1
    return (low, high)


def norm_values(x):
    x = np.array(x)
    (low, high) = norm_value_range(x)
    xnorm = (x - low) / (high - low)
    return np.clip(xnorm, 0, 1)


def process_node_colors(graph, features_df, column_name, marker):
    feature_values = [features_df.loc[node_id, column_name] for node_id in graph]
    if column_name.endswith('_label'):
        node_colors = [get_label_color(x) for x in feature_values]
        legend_colors = get_label_legend(marker)
    else:
        colormap = matplotlib.cm.get_cmap('viridis', 10)
        node_colors = [colormap(x) for x in norm_values(feature_values)]
        legend_colors = get_color_legend(
            *norm_value_range(feature_values), colormap, marker)
    return (node_colors, legend_colors)

def draw_graph(graph, features_df, column_name, output_filename, sample_id):
    fig, ax = plt.subplots(figsize=(15, 10))

    columns = column_name.split(":")
    assert len(columns) == 1 or len(columns) == 2

    # compute node sizes based on sequence length
    node_sizes = []
    for node_id in graph:
        seq_length = features_df.loc[node_id, 'length']
        node_sizes.append(get_node_size(seq_length))
        
    # for one or two selected table columns, compute colors and legend items
    node_colors = []
    legend_colors = []
    markers = ["s", "o"]
    for (idx, column) in enumerate(columns):
        (node_colors_curr, legend_colors_curr) = process_node_colors(graph, features_df, column, markers[idx])
        node_colors.append(node_colors_curr)
        legend_colors.append(legend_colors_curr)

    pos = nx.kamada_kawai_layout(graph)

    # draw square nodes if two columns required
    if len(columns) == 2:
        large_sizes = [2 * x for x in node_sizes]
        nx.draw_networkx_nodes(graph, pos=pos, node_size=large_sizes, node_shape='s',
                node_color=node_colors[0], alpha=0.8)

    # draw circle nodes
    nx.draw(graph, pos=pos, node_size=node_sizes,
            node_color=node_colors[-1], alpha=0.8, width=0.2)
    
    legend1 = plt.legend(
        handles=get_size_legend(),
        loc="upper left",
        bbox_to_anchor=(1, 0.9),
        title="lengths",
    )
    plt.gca().add_artist(legend1)

    legend2 = plt.legend(
        handles=legend_colors[0],
        loc="upper left",
        bbox_to_anchor=(1, 0.71),
        title=columns[0],
    )
    plt.gca().add_artist(legend2)

    if(len(legend_colors) == 2):
        legend3 = plt.legend(
           handles=legend_colors[1],
            loc="upper left",
            bbox_to_anchor=(1, 0.51),
            title=columns[1],
        )
        plt.gca().add_artist(legend3)   

    plt.title(f"Sample {sample_id}, {column_name}")
    plt.savefig(output_filename, dpi=500, format="png", bbox_inches="tight")
    plt.clf()
    plt.close()


def get_color_legend(low, high, colormap, marker):
    legend_elements = []
    for frac in [0, 0.25, 0.5, 0.75, 0.99]:
        value = frac * high + (1 - frac) * low
        legend_elements.append(Line2D(
            [0],
            [0],
            marker=marker,
            color="w",
            label=f"{value:.2g}",
            markerfacecolor=colormap(frac),
            markersize=12,
            alpha=0.8,
        ))
    return legend_elements


def get_label_legend(marker):
    legend_elements = []
    for (label, color) in NODE_COLOR_SETTINGS:
        legend_elements.append(Line2D(
            [0],
            [0],
            marker=marker,
            color="w",
            label=label,
            markerfacecolor=color,
            markersize=12,
            alpha=0.8,
        ))
    return legend_elements


def get_size_legend():
    legend_elements = []
    for (idx, (threshold, size, marker)) in enumerate(NODE_SIZE_SETTINGS):
        if idx == len(NODE_SIZE_SETTINGS) - 1:
            label = f"> {NODE_SIZE_SETTINGS[idx-1][0]} bp"
        else:
            label = f"<= {threshold} bp"
        legend_elements.append(Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=label,
            markerfacecolor="grey",
            markersize=marker,
            alpha=0.8,
        ))
    return legend_elements


def get_label_color(node_label):
    for (label, color) in NODE_COLOR_SETTINGS:
        if node_label == label:
            return color
    return get_label_color('unknown')


def get_node_size(seq_length):
    for (threshold, size, marker) in NODE_SIZE_SETTINGS:
        if seq_length <= threshold:
            return size
    return NODE_SIZE_SETTINGS[-1, 1]  # for very big, return last size


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=inspect.getdoc(main))
    parser.add_argument(
        "file_list", help="csv file with a list of training or testing samples")
    parser.add_argument(
        "file_prefix", help="common prefix to be used for all filenames listed in file_list, e.g. ../data/")
    parser.add_argument(
        "summary_file", help="csv files with features to be used in visualization")
    parser.add_argument(
        "selected_columns", help="which columns to use to color nodes, separated by commas")
    parser.add_argument(
        "output_dir", help="where to put output files for each sample and selected column")
    args = parser.parse_args()
    main(** vars(args))
