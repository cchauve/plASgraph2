#!/usr/bin/env python
# coding: utf-8

"""This script can visualize the assembly graph and the output of
plASgraph2 or other statistics related to the contigs provided as a
csv file.

It has two subcommands: "gfa" and "set". More precise usage can be
obtained by rnning a subcommand -h option.

In the "gfa" mode it needs as an input a single gfa or gfa.gz file, a
csv file with the data to be displayed, the name of the column from
csv to be visualized using node colors and the name of the output
file.  The "set" mode is for automating creation of figures for many
graphs.

The nodes of the assembly graph (contigs) are shown as circles, with
size corresponding to contig length (we use 4 discrete circle sizes).
Edges are shown as lines. Color of the circle is assigned according to
the value in the selected column of the csv file.

If the name of the selected column is "label" or a longer string
ending in "label", it is assumed that the column contains a
classification of contigs into classes "chromosome", "plasmid",
"ambiguous", "unlabeled" (it needs to use exactly these strings). Any
other string will be cosidered as equivalent to "unlabeled".

If the name of the column does not end with "label", the column should
contain numeric data which are shown on a color scale.  The color
scale is constructed so that values more than 1 interquantile range
(IQR) from the median are considered as outliers and use the extreme
ends of the color scale.

Instead of a single column, you can specified 2 column names
separated with a colon. Each node will be then displayed as a circle
inside a square, with square having the column according to the first
column and the circle having a color according to the secodn column.

The csv file should have column names in the first row. There should a
column named "contig" which contains contig identifiers identical to
contig names in the gfa file. The csv file may contain only a subset
of nodes from GFA, e.g. very short contigs may be omitted. In the
"set" mode, there should also be a column call "sample" with the
sample name, as all samples are read from the same csv files. 

"""

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


# triples (length threshold, node size in nx, marker size in legend)
NODE_SIZE_SETTINGS = [(100, 20, 6), (1000, 40, 9),
                      (10000, 80, 12), (1e10, 160, 15)]
# pairs label, node color
NODE_COLOR_SETTINGS = [("chromosome", "C0"), ("plasmid", "C1"), 
                        ("ambiguous", "black"), ("unlabeled", "gray")]
# how many interquartile ranges from median is start of outliers when coloring nodes
# traditional value 1.5 seems to be too large
IQR_COEFF = 1

def main_set(file_list, file_prefix, data_csv, selected_columns, output_dir):
    """
    The general description of the tool can be obtained by running the script with -h without
    specifying set or gfa subcommand.

    In the "set" subcommand, the script produces potentially many figures in one run. 

    Input consists of a list of files in a csv file, similarly as for plASgraph2_classify in set mode 
    and a data csv file with lines for nodes in these graphs and various named colums. 
    The first column of the file list should cotain matcing IDs with the column name "sample" in data csv.

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
    features_df = read_data_csv(data_csv, default_sample='.')

    graph_files = pd.read_csv(file_list, names=(
        'graph', 'csv', 'sample_id'), header=None)
    for idx, row in graph_files.iterrows():
        graph_file = row['graph']
        sample_id = row['sample_id']
        graph = create_graph.read_single_graph(
            file_prefix, graph_file, sample_id, minimum_contig_length=0)
        for column in columns:
            column_filename = column.replace(":", "_")
            output_filename = os.path.join(
                output_dir, f"{sample_id}_{column_filename}.png")
            draw_graph(graph, features_df, column, output_filename, sample_id)

def main_gfa(
    graph_file : 'gfa or gfa.gz file', 
    data_csv : 'a csv file with columns listing values for contigs from gfa',
    column_name : 'name of one of the columns from data_csv to be displayed (or two columns separated by a comma)',
    output_png_file : 'name of the output .png file'
    ):

    """The general description of the tool can be obtained by running the script with -h without
    specifying set or gfa subcommand.

    In the "gfa" subcommand it creates a visualization of a single gfa or gfa.gz file."""

    features_df = read_data_csv(data_csv, default_sample='.')
    # check that in this case all samples have the same id
    samples = features_df["sample"].unique()
    if len(samples) > 1:
        raise KeyError(f"column named 'sample' contains multiple different sample IDs in {data_csv}, only one allowed in gfa mode") 
    sample_id = samples[0]

    print(f"reading {graph_file}", file=sys.stderr)
    graph = create_graph.read_single_graph('', graph_file, sample_id, minimum_contig_length=0)
    draw_graph(graph, features_df, column_name, output_png_file, sample_id)

def read_data_csv(data_csv, default_sample=None):
    """Read csv into a pandas table, check required columns. 
    If column sample missing and default_sample given, add a new column.
    Also add a new column _id with node id consisting of sample and contig and use it as index"""

    print(f"Reading {data_csv}", file=sys.stderr)
    features_df = pd.read_csv(data_csv)
    # check that contig and sample columns are present, add sample column if desired
    if "contig" not in features_df.columns:
        raise KeyError(f"column named 'contig' missing in {data_csv}")
    if "sample" not in features_df.columns:
        if default_sample is not None:
            features_df["sample"] = default_sample
        else:
            raise KeyError(f"column named 'sample' missing in {data_csv}")

    # create new node id's as a combination of sample and node id
    features_df["_id"] = features_df.apply(
        lambda x: helpers.get_node_id(x["sample"], x["contig"]), axis=1)
    # index dataframe by the new id
    features_df.set_index("_id", inplace=True)
    return features_df
    
def norm_value_range(x):
    x = np.array(x)
    q1, median, q3 = np.nanquantile(x, [0.25, 0.50, 0.75])
    iqr = q3 - q1
    low = max(np.nanmin(x), median - IQR_COEFF * iqr)
    high = min(np.nanmax(x), median + IQR_COEFF * iqr)
    if low == high:
        high = low + 1
    return (low, high)


def norm_values(x):
    x = np.array(x)
    (low, high) = norm_value_range(x)
    xnorm = (x - low) / (high - low)
    return np.clip(xnorm, 0, 1)

def get_nodes_values(graph, features_df, column_name, default):
    if column_name not in features_df.columns:
         raise KeyError(f"column named '{column_name}' missing in data csv file")
    
    result = []
    for node_id in graph:
        if node_id in features_df.index:
            node_value = features_df.loc[node_id, column_name]
        else:
            # short nodes may be not present in summary file
            node_value = default
        result.append(node_value)
    return result
    
def process_node_colors(graph, features_df, column_name, marker):
    feature_values = get_nodes_values(graph, features_df, column_name, np.nan)
    if column_name.endswith('label'):
        node_colors = [get_label_color(x) for x in feature_values]
        legend_colors = get_label_legend(marker)
    else:
        colormap = matplotlib.cm.get_cmap('viridis', 10)
        node_colors = [colormap(x) for x in norm_values(feature_values)]
        legend_colors = get_color_legend(
            *norm_value_range(feature_values), colormap, marker)
    return (node_colors, legend_colors)

def draw_graph(graph, features_df, column_name, output_filename, sample_id):
    #fig, ax = plt.subplots(figsize=(15, 10))

    # split columns and do some checks
    columns = column_name.split(":")
    if len(columns) != 1 and len(columns) != 2:
        raise KeyError(f"Bad columns specification {column_name}")
        
    print(f"creating {output_filename}", file=sys.stderr)

    # compute node sizes based on sequence length
    node_sizes = [get_node_size(graph.nodes[contig_id]["length"]) for contig_id in graph]
        
    # for one or two selected table columns, compute colors and legend items
    node_colors = []
    legend_colors = []
    if len(columns) == 1:
        markers = ["o"]
    else:
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
        bbox_to_anchor=(1, 0.64),
        title=columns[0],
    )
    
    
    if(len(legend_colors) == 2):
        plt.gca().add_artist(legend2)
        legend3 = plt.legend(
            handles=legend_colors[1],
            loc="upper left",
            bbox_to_anchor=(1, 0.35),
            title=columns[1],
        )

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
    return get_label_color('unlabeled')


def get_node_size(seq_length):
    for (threshold, size, marker) in NODE_SIZE_SETTINGS:
        if seq_length <= threshold:
            return size
    return NODE_SIZE_SETTINGS[-1, 1]  # for very big, return last size


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=inspect.getdoc(inspect.getmodule(main_set)), formatter_class=argparse.RawDescriptionHelpFormatter)

    subparser = parser.add_subparsers(dest='command')

    cmd_gfa = subparser.add_parser('gfa', description=inspect.getdoc(main_gfa))
    cmd_gfa.add_argument("graph_file", help="input gfa or gfa.gz file")
    cmd_gfa.add_argument("data_csv", help="a csv file with columns listing values for contigs from gfa")
    cmd_gfa.add_argument("column_name", help="name of one of the columns from data_csv")
    cmd_gfa.add_argument("output_png_file", help="name of the output .png file")

    cmd_set = subparser.add_parser('set', description=inspect.getdoc(main_set))
    cmd_set.add_argument("file_list", help="csv file with a list of samples to process")
    cmd_set.add_argument("file_prefix", help="common prefix to be used for all filenames listed in file_list, e.g. ../data/")
    cmd_set.add_argument("data_csv", help="a csv file  columns listing values for contigs from gfa")
    cmd_set.add_argument("selected_columns", help="which columns to use to color nodes, separated by commas")
    cmd_set.add_argument("output_dir", help="folder where to put output files for each sample and selected column")

    args = parser.parse_args()
        
    arg_dict = vars(args).copy()
    del arg_dict['command']
    if args.command == 'set':
        main_set(**arg_dict)
    elif args.command == 'gfa':
        main_gfa(**arg_dict)
    else:
        parser.print_usage()

