#!/usr/bin/env python3
# coding: utf-8

"""
Run plASgraph model on a single gfa file or a whole set of files.

The first argument is a command name, followed by arguments of a particular 
command. Commands are 

  set   runs plASgraph on a set of files
  gfa   runs plASgraph on a single gfa file

More details run test.py [command] -h 
"""

import gzip
import os
import pandas as pd
import networkx as nx
import numpy as np
import argparse
import inspect
import architecture

from tensorflow import keras

from spektral.transforms import GCNFilter
from spektral.data.loaders import SingleLoader

import create_graph
import helpers
import config



def main_set(
    test_file_list : 'csv file with a list of testing samples',
    file_prefix : 'common prefix to be used for all filenames listed in test_file_list, e.g. ../data/',         
    model_dir : 'name of an input folder with the trained model',
    output_file : 'name of the output csv file'
    ):

    """Runs a trained model on datasets listed in a file and writes outputs as a csv file.

    The test_file_list is a csv with a list of testing data samples. It contains no header and three comma-separated values:
      name of the gfa.gz file from short read assemby, 
      name of the csv file with correct answers (not used by this script, can be arbitrary value)
      id of the sample (string without colons, commas, whitespace, unique within the set)
    """

    test_files = pd.read_csv(test_file_list, names=('graph','csv','sample_id'), header=None)

    # loading the model from a file
    model = keras.models.load_model(model_dir)
    # Creating a dictionary parameters of parameter values from YAML file
    parameters = config.config(os.path.join(model_dir, config.DEFAULT_FILENAME))
        
    # predict
    for idx, row in test_files.iterrows():
        graph_file = row['graph']
        sample_id = row['sample_id']
        prediction_df = test_one(file_prefix, graph_file, model, parameters, sample_id)

        if idx==0:
            prediction_df.to_csv(output_file, header=True, index=False, mode='w')
        else:
            prediction_df.to_csv(output_file, header=False, index=False, mode='a')


def main_gfa(
    graph_file : 'gfa or gfa.gz file',
    model_dir : 'name of an input folder with the trained model',
    output_file : 'name of the output csv file'
    ):

    """Runs a trained model on a single gfa or gfa.gz file."""

    # loading the model from a file
    model = keras.models.load_model(model_dir)
    # Creating a dictionary parameters of parameter values from YAML file
    parameters = config.config(os.path.join(model_dir, config.DEFAULT_FILENAME))
        
    # predict
    prediction_df = test_one('', graph_file, model, parameters, '.')
    # write output to a file
    prediction_df.to_csv(output_file, header=True, index=False, mode='w')


def test_one(file_prefix, graph_file, model, parameters, sample_id):
    
    G = create_graph.read_single_graph(file_prefix, graph_file, sample_id, parameters['minimum_contig_length'])
    node_list = list(G)  # fix order of nodes

    the_graph = create_graph.Networkx_to_Spektral(G, node_list, parameters)
    the_graph.apply(GCNFilter())
   
    # compute predictions
    preds = architecture.apply_to_graph(model, the_graph, parameters)
        
    # prediction to df
    list_of_lists_with_prediction = []
    for index, contig_id in enumerate(node_list):
        contig_short = G.nodes[contig_id]["contig"]
        contig_len = G.nodes[contig_id]["length"]
        plasmid_probability = preds[index][0]
        chromosome_probability = preds[index][1]
        label = helpers.pair_to_label(list(np.around(preds[index])))
        list_of_lists_with_prediction.append(
            [sample_id, contig_short, contig_len, plasmid_probability, chromosome_probability, label]
    )

    prediction_df = pd.DataFrame(
        list_of_lists_with_prediction,
        columns=[
            "sample",
            "contig",
            "length",
            "plasmid_score",
            "chrom_score",
            "label",
        ],
    )

    return prediction_df


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=inspect.getdoc(inspect.getmodule(main_set)), formatter_class=argparse.RawDescriptionHelpFormatter)

    subparser = parser.add_subparsers(dest='command')
    cmd_set = subparser.add_parser('set', description=inspect.getdoc(main_set))
    cmd_set.add_argument("test_file_list", help="csv file with a list of testing samples")
    cmd_set.add_argument("file_prefix", help="common prefix to be used for all filenames listed in test_file_list, e.g. ../data/")
    cmd_set.add_argument("model_dir", help="name of the input folder with the trained model")
    cmd_set.add_argument("output_file", help="name of the output csv file")
    
    cmd_set = subparser.add_parser('gfa', description=inspect.getdoc(main_gfa))
    cmd_set.add_argument("graph_file", help="input gfa or gfa.gz file")
    cmd_set.add_argument("model_dir", help="name of the input folder with the trained model")
    cmd_set.add_argument("output_file", help="name of the output csv file")
    
    args = parser.parse_args()
    arg_dict = vars(args).copy()
    del arg_dict['command']
    if args.command == 'set':
        main_set(**arg_dict)
    elif args.command == 'gfa':
        main_gfa(**arg_dict)
    else:
        parser.print_usage()
                
