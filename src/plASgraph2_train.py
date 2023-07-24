#!/usr/bin/env python3
# coding: utf-8

import itertools
import inspect
import gzip
import pandas as pd
import networkx as nx
import numpy as np
import os
import random
import argparse
import shutil

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers

from spektral.transforms import GCNFilter
from spektral.data.loaders import SingleLoader

import architecture
import create_graph
import config
import thresholds

tf.config.run_functions_eagerly(
    True
)  # https://stackoverflow.com/questions/58352326/running-the-tensorflow-2-0-code-gives-valueerror-tf-function-decorated-functio

def main(config_file : 'YAML configuration file',
         train_file_list : 'csv file with a list of training samples',
         file_prefix : 'common prefix to be used for all filenames listed in train_file_list, e.g. ../data/',
         model_output_dir : 'name of an output folder with the trained model',
         gml_file : 'optional output file for the graph in GML format' =None,
         log_dir : 'optional output folder for various training logs' =None) : 
    """Trains a model for a given dataset and writes out the trained model and optionally also the graph.

    The train_file_list is a csv with a list of testing data samples. It contains no header and three comma-separated values:
      name of the gfa.gz file from short read assemby,
      name of the csv file with correct answers, 
      id of the sample (string without colons, commas, whitespace, unique within the set)   
    """
    
    # Creating a dictionary parameters of parameter values from YAML file
    parameters = config.config(config_file)

    # read GFA and CSV files in the training set
    G = create_graph.read_graph_set(file_prefix, train_file_list, parameters['minimum_contig_length'])
    node_list = list(G)  # fix node order
    
    if gml_file is not None:
        nx.write_gml(G, path=gml_file)

    # generate spektral graph
    all_graphs = create_graph.Networkx_to_Spektral(G, node_list, parameters)

    all_graphs.apply(GCNFilter()) # normalize by degrees, add 1 along diagonal

    if log_dir is not None:
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)

    print(all_graphs[0])

    # sample weights and masking
    number_total_nodes = len(node_list)

    labels = [G.nodes[node_id]["text_label"] for node_id in node_list]
    num_unlabelled = labels.count("unlabeled")
    num_chromosome = labels.count("chromosome")
    num_plasmid = labels.count("plasmid")
    num_ambiguous = labels.count("ambiguous")
    assert number_total_nodes == num_unlabelled + num_chromosome + num_plasmid + num_ambiguous
    
    print(
        "Chromosome contigs:",
        num_chromosome,
        "Plasmid contigs:",
        num_plasmid,
        "Ambiguous contigs:",
        num_ambiguous,
        "Unlabelled contigs:",
        num_unlabelled,
    )

    # for each class, calculate weight. Set unlabelled contigs weight to 0
    # chromosome_weight = (num_unlabelled + num_plasmid + num_ambiguous) / number_total_nodes
    # plasmid_weight = (num_unlabelled + num_chromosome + num_ambiguous) / number_total_nodes
    # ambiguous_weight = (num_unlabelled + num_chromosome + num_plasmid) / number_total_nodes
    chromosome_weight = 1
    plasmid_weight = parameters["plasmid_ambiguous_weight"]
    ambiguous_weight = plasmid_weight
    # plasmid_weight = 1
    # ambiguous_weight = 1

    masks = []

    for node_id in node_list:
        label = G.nodes[node_id]["text_label"]

        if label == "unlabeled":
            masks.append(0)
        elif label == "chromosome":
            masks.append(chromosome_weight)
        elif label == "plasmid":
            masks.append(plasmid_weight)
        elif label == "ambiguous":
            masks.append(ambiguous_weight)


    #set seeds for reproducibility: done when reading the YAML config file now
    #seed_number = 123
    seed_number = parameters["random_seed"]

    os.environ['PYTHONHASHSEED']=str(seed_number)
    random.seed(seed_number)
    np.random.seed(seed_number+1)
    tf.random.set_seed(seed_number+2)

            
    # 80% train 20% validate
    
    #masks_train = masks[0:int(len(masks)*0.8)] + [0]*(int(len(masks)*0.2)+1)
    #masks_validate = [0]*int(len(masks)*0.8) + masks[int(len(masks)*0.8):]

    masks_train = masks.copy()
    masks_validate = masks.copy()
    
    for i in range(len(masks)):
        if random.random() > 0.8:
            masks_train[i] = 0
        else:
            masks_validate[i] = 0

    print(masks_train[0:20])
    print(masks_validate[0:20])
    
    masks_train = np.array(masks_train).astype(float)
    masks_validate = np.array(masks_validate).astype(float)

    print(len(masks_train))
    print(len(masks_validate))

    learning_rate = parameters['learning_rate']

    model = architecture.plasgraph(parameters=parameters)
    if parameters["loss_function"] == "squaredhinge":
        loss_function = tf.keras.losses.SquaredHinge(reduction="sum")
    elif parameters["loss_function"] == "crossentropy":
        loss_function = tf.keras.losses.BinaryCrossentropy(reduction="sum")
    elif parameters["loss_function"] == "mse":
        loss_function = tf.keras.losses.MeanSquaredError(reduction="sum")
    else:
        raise ValueError(f"Bad loss function {parameters['loss_function']}")

    model.compile(optimizer=Adam(learning_rate), loss=loss_function, weighted_metrics=[])

    loader_tr = SingleLoader(all_graphs, sample_weights=masks_train)
    loader_va = SingleLoader(all_graphs, sample_weights=masks_validate)
    
    history = model.fit(
        loader_tr.load(),
        steps_per_epoch=loader_tr.steps_per_epoch,
        validation_data=loader_va.load(),
        validation_steps=loader_va.steps_per_epoch,
        epochs=parameters['epochs'],
        callbacks=[EarlyStopping(
            patience=parameters['early_stopping_patience'], 
            monitor='val_loss', 
            mode='min', verbose=1, restore_best_weights=True
        )]
    )
    
    # print losses to log files
    if log_dir is not None:
        with open(os.path.join(log_dir, "val_loss.csv"), "wt") as file:
            for (i, value) in enumerate(history.history['val_loss']):
                print(f"{i},{value}", file=file)
        with open(os.path.join(log_dir, "train_loss.csv"), "wt") as file:
            for (i, value) in enumerate(history.history['loss']):
                print(f"{i},{value}", file=file)
    
    if parameters['set_thresholds']:
        thresholds.set_thresholds(model, all_graphs, masks_validate, parameters, log_dir)

    model.save(model_output_dir)
    #shutil.copyfile(config_file, os.path.join(model_output_dir, os.path.basename(config_file)))
    parameters.write_yaml(os.path.join(model_output_dir, config.DEFAULT_FILENAME))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=inspect.getdoc(main))
    parser.add_argument("config_file", help="YAML configuration file")
    parser.add_argument("train_file_list", help="csv file with a list of training samples")
    parser.add_argument("file_prefix", help="common prefix to be used for all filenames listed in train_file_list, e.g. ../data/")
    parser.add_argument("model_output_dir", help="name of the output folder with the trained model")
    parser.add_argument("-g", dest="gml_file", help="optional output file for the graph in GML format", default = None)
    parser.add_argument("-l", dest="log_dir", help="optional output folder for various training logs", default = None)
    args = parser.parse_args()
    main(** vars(args))

    
