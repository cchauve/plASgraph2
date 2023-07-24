import itertools
import gzip
import re
import pandas as pd
import networkx as nx
import numpy as np
import os
import math
import fileinput


from spektral.data import Dataset
from spektral.data import Graph
from sklearn.preprocessing import MinMaxScaler
from scipy.special import rel_entr

import helpers

# convert networkx graph to a spektral graph
class Networkx_to_Spektral(Dataset):
    def __init__(self, nx_graph, node_order, parameters, **kwargs):
        self.nx_graph = nx_graph
        self.node_order = node_order
        self.parameters = parameters

        super().__init__(**kwargs)

    def extract_features(self, node_id, features):
        return [self.nx_graph.nodes[node_id][f] for f in features]
        
    def extract_y(self):
        label_features = ["plasmid_label", "chrom_label"]
        y = np.array(
            [self.extract_features(node_id, label_features) for node_id in self.node_order]
        )
        # squared hinge loss function needs correct labels -1, 1 rather than 0, 1
        if self.parameters["loss_function"] == "squaredhinge":
            y = y * 2 - 1
        return y


    def read(self):

        features = self.parameters['features']
        x = np.array(
            [self.extract_features(node_id, features) for node_id in self.node_order]
        )

        y = self.extract_y()
        
        a = nx.adjacency_matrix(self.nx_graph, nodelist=self.node_order)
        a.setdiag(0)
        a.eliminate_zeros()

        # return a list of Graph objects
        return [Graph(x=x.astype(float), a=a.astype(float), y=y.astype(float))]


def KL(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.sum(np.where(a != 0, a * np.log(a / b), 0))

def weighted_median(values, weights):
    middle = np.sum(weights) / 2
    cum = np.cumsum(weights)
    for (i, x) in enumerate(cum):
        if x >= middle:
            return values[i]
    assert False

def add_normalized_coverage(graph, current_nodes):
    """Add attribute coverage_norm which is original coverage divided by median weighted by length.
    (only for nodes in current_nodes list)"""

    # similarly to Unicycler's computation
    # from function get_median_read_depth in 
    # https://github.com/rrwick/Unicycler/blob/main/unicycler/assembly_graph.py

    sorted_nodes = sorted(current_nodes, key=lambda x : graph.nodes[x]["coverage"])
    lengths = np.array([graph.nodes[x]["length"] for x in sorted_nodes])
    coverages = np.array([graph.nodes[x]["coverage"] for x in sorted_nodes])
    median = weighted_median(coverages, lengths)
    for node_id in current_nodes:
        graph.nodes[node_id]["coverage_norm"] = graph.nodes[node_id]["coverage"] / median

def get_node_coverage(gfa_arguments, seq_length):
    """Return coverage parsed from dp or estimated from KC tag. 
    The second return value is True for dp and False for KC"""
    # try finding dp tag
    for x in gfa_arguments:
        match =  re.match(r'^dp:f:(.*)$',x)
        if match :
            return (float(match.group(1)), True)
    # try finding KC tag
    for x in gfa_arguments:
        match =  re.match(r'^KC:i:(.*)$',x)
        if match :
            return (float(match.group(1)) / seq_length, False)
    raise AssertionError("depth not found")

def read_graph(graph_file, csv_file, sample_id, graph, minimum_contig_length):
    """Read a single graph from gfa of gfa.gz, compute attributes, add its nodes and edges to nx graph. 
    Label csv file can be set to None. Contigs shorter than minimum_contig_length are contracted."""

    # first pass: read all nodes
    current_nodes = []
    whole_seq = ""  # concatenated contigs
    coverage_types = {True:0, False:0}  # which coverage types for individual nodes

    with fileinput.input(graph_file, openhook=fileinput.hook_compressed, mode='r') as file:
        for line in file: 
            if isinstance(line, bytes):
                line = line.decode("utf-8") # convert byte sequences to strings
            parts = line.strip().split("\t")
            if parts[0] == "S":  # node line
                node_id = helpers.get_node_id(sample_id, parts[1])
                seq = parts[2].upper()
                if not re.match(r'^[A-Z]*$', seq):
                    raise AssertionError(f"Bad sequence in {node_id}")

                whole_seq += "N" + seq  # N is ignored by GC, helps to avoid fake kmers
                current_nodes.append(node_id)
                assert node_id not in graph
                graph.add_node(node_id)
                seq_length = len(seq)

                graph.nodes[node_id]["contig"] = parts[1]
                graph.nodes[node_id]["sample"] = sample_id
                graph.nodes[node_id]["length"] = seq_length
                (coverage, is_dp) = get_node_coverage(parts[3:], seq_length)
                graph.nodes[node_id]["coverage"] = coverage
                coverage_types[is_dp] += 1
                graph.nodes[node_id]["gc"] = helpers.get_gc_content(seq)
                graph.nodes[node_id]["kmer_counts_norm"] = helpers.get_kmer_distribution(seq, scale=True)
    
    # check that only one coverage type seen
    assert coverage_types[True] == 0 or coverage_types[False] == 0

    # second pass: read all edges
    with fileinput.input(graph_file, openhook=fileinput.hook_compressed, mode='r') as file:
        for line in file: 
            if isinstance(line, bytes):
                line = line.decode("utf-8")
            parts = line.strip().split("\t")
            if parts[0] == "L":  # edge line
                graph.add_edge(helpers.get_node_id(sample_id, parts[1]),
                               helpers.get_node_id(sample_id, parts[3]))


    # get graph degrees
    for node_id in current_nodes:
            graph.nodes[node_id]["degree"] = graph.degree[node_id]

    # get gc of whole seq
    gc_of_whole_seq = helpers.get_gc_content(whole_seq)
    # set normalized gc content
    for node_id in current_nodes:
        graph.nodes[node_id]["gc_norm"] =  graph.nodes[node_id]["gc"] - gc_of_whole_seq
    

    # get max length
    max_contig_length = max([graph.nodes[node_id]["length"] for node_id in current_nodes])
    # get normalized contig lengths (divided by max length)
    for node_id in current_nodes:
        #graph.nodes[node_id]["length_norm"] =  graph.nodes[node_id]["length"] / max_contig_length
        graph.nodes[node_id]["length_norm"] =  graph.nodes[node_id]["length"] / 2000000
        graph.nodes[node_id]["loglength"] = math.log(graph.nodes[node_id]["length"]+1)

    add_normalized_coverage(graph, current_nodes)
        
    # get euclidian of pentamer distribution for each node
    all_kmer_counts_norm = np.array(helpers.get_kmer_distribution(whole_seq, scale=True))
    for node_id in current_nodes:
        diff = np.array(graph.nodes[node_id]["kmer_counts_norm"]) - all_kmer_counts_norm
        graph.nodes[node_id]["kmer_dist"] = np.linalg.norm(diff)
        graph.nodes[node_id]["kmer_dot"] = np.dot(np.array(graph.nodes[node_id]["kmer_counts_norm"]),all_kmer_counts_norm)
        graph.nodes[node_id]["kmer_kl"] = KL(np.array(graph.nodes[node_id]["kmer_counts_norm"]),all_kmer_counts_norm)
    
        
    # read and add node labels
    if csv_file is not None:
        df_labels = pd.read_csv(csv_file)
        df_labels["id"] = df_labels["contig"].map(lambda x : helpers.get_node_id(sample_id, x))
        df_labels.set_index("id", inplace=True)
    else:
        df_labels = pd.DataFrame()

    for node_id in current_nodes:
        label = None
        if node_id in df_labels.index:
            label = df_labels.loc[node_id, "label"]  # textual label

        pair = helpers.label_to_pair(label)  # pair of binary values
        graph.nodes[node_id]["text_label"] = helpers.pair_to_label(pair)
        graph.nodes[node_id]["plasmid_label"] = pair[0]
        graph.nodes[node_id]["chrom_label"] = pair[1]

    if minimum_contig_length > 0:
        delete_short_contigs(graph, current_nodes, minimum_contig_length)
    
def delete_short_contigs(graph, node_list, minimum_contig_length):
    """check length attribute of all contigs in node_list 
    and if some are shorter than minimum_contig_length,
    remove them from the graph and connect new neighbors"""
    for node_id in node_list:
        if graph.nodes[node_id]["length"] < minimum_contig_length:
            neighbors = list(graph.neighbors(node_id))
            all_new_edges = list(itertools.combinations(neighbors, 2))
            for edge in all_new_edges:
                graph.add_edge(edge[0], edge[1])
            graph.remove_node(node_id)


def read_single_graph(file_prefix, gfa_file, sample_id, minimum_contig_length):
    """Read single graph without node labels for testing"""
    graph = nx.Graph()
    graph_file = file_prefix + gfa_file
    read_graph(graph_file, None, sample_id, graph, minimum_contig_length)
    return graph

def read_graph_set(file_prefix, file_list, minimum_contig_length, read_labels=True):
    """Read several graph files to a single graph. 
    Node labels will be read from the csv file for each graph if read_labels is True. 
    Nodes shorter than minimum_contig_length will be deleted from the graph.
    """

    # read data frame with files
    train_files = pd.read_csv(file_list, names=('graph','csv','sample_id'))

    graph = nx.Graph()
    for idx, row in train_files.iterrows():

        graph_file = file_prefix + row['graph']
        if read_labels:
            csv_file = file_prefix + row['csv']
        else:
            csv_file = None
        read_graph(graph_file, csv_file, row['sample_id'], graph, minimum_contig_length)

    return graph
