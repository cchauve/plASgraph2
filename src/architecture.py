import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow.python.training.tracking.data_structures import NoDependency

from spektral.layers import GCNConv
from spektral.layers.convolutional import gcn_conv
from spektral.transforms import LayerPreprocess
from spektral.transforms import GCNFilter
from spektral.data import Dataset
from spektral.data import Graph
from spektral.data.loaders import SingleLoader

import numpy as np
import thresholds

class plasgraph(tf.keras.Model):
    def __init__(
        self,
        parameters,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._parameters = parameters
        # individual parameters can be now accessed using self['param_name']
            
        reg = tf.keras.regularizers.l2(self['l2_reg'])
        
        # Input layers
        self.preproc = layers.Dense(self['n_channels_preproc'], self['preproc_activation'])
        self._fully_connected_input_1 = layers.Dense(
            self['n_channels'], activation=self['fully_connected_activation']
        )
        self._fully_connected_input_2 = layers.Dense(
            self['n_channels'], activation=self['fully_connected_activation']
        )
        
        # GNN iterations layers
        self._gnn_dropout_pre_gcn = NoDependency([
            layers.Dropout(self['dropout_rate']) 
            for gnn_layer in range(self['n_gnn_layers'])
        ])
        self._gnn_dropout_pre_fully_connected = NoDependency([
            layers.Dropout(self['dropout_rate']) 
            for gnn_layer in range(self['n_gnn_layers'])
        ])
        if self['tie_gnn_layers']:
            self.gnn_gcn_layer = gcn_conv.GCNConv(
                channels=self['n_channels'], activation=self['gcn_activation'], 
                kernel_regularizer=reg, use_bias=True
            )
            self.gnn_fully_connected_layer = layers.Dense(
                self['n_channels'], activation=self['fully_connected_activation']
            )
            self._gnn_gcn = NoDependency([self.gnn_gcn_layer] * self['n_gnn_layers'])
            self._gnn_fully_connected = NoDependency(
                [self.gnn_fully_connected_layer] * self['n_gnn_layers']
            )
            
        else:
            gcn_list = []  # temporary list of GCN layers
            dense_list = []  # temporary list of fully connected layers
            for layer_idx in range(self['n_gnn_layers']):
                # create names of instance variables for the layers
                gcn_name = f"gnn_gcn_{layer_idx}"
                dense_name = f"gnn_dense_{layer_idx}"
                # create layers and store as instance variables (needed to save the model)
                setattr(self, gcn_name, gcn_conv.GCNConv(
                    channels=self['n_channels'], activation=self['gcn_activation'], 
                    kernel_regularizer=reg, use_bias=True
                ))
                setattr(self, dense_name, layers.Dense(
                    self['n_channels'], activation=self['fully_connected_activation']
                ))
                # add layers to lists
                gcn_list.append(getattr(self, gcn_name))
                dense_list.append(getattr(self, dense_name))
            # store lists in instance variables which are not saved
            self._gnn_gcn = NoDependency(gcn_list)
            self._gnn_fully_connected = NoDependency(dense_list)            

        # Last layers
        self._fully_connected_last_1 = layers.Dense(
            self['n_channels'], activation=self['fully_connected_activation']
        )
        self._fully_connected_last_2 = layers.Dense(
            self['n_labels'], activation=self['output_activation']
        )
        self._dropout_last_1 = layers.Dropout(self['dropout_rate'])
        self._dropout_last_2 = layers.Dropout(self['dropout_rate'])

    def __getitem__(self, key):
        return self._parameters[key]

    def call(self, inputs):
        x, a = inputs

        # Input layer
        x = self.preproc(x)
        node_identity = self._fully_connected_input_1(x)
        x = self._fully_connected_input_2(x)

        # GNN layers
        for gnn_layer in range(self['n_gnn_layers']):
            x = self._gnn_dropout_pre_gcn[gnn_layer](x)
            x = self._gnn_gcn[gnn_layer]([x, a])
            merged = layers.concatenate([node_identity, x])
            x = self._gnn_dropout_pre_fully_connected[gnn_layer](merged)
            x = self._gnn_fully_connected[gnn_layer](x)            
        
        # Last Layer
        merged = layers.concatenate([node_identity, x])
        x = self._dropout_last_1(merged)
        x = self._fully_connected_last_1(x)
        x = self._dropout_last_2(x)
        x = self._fully_connected_last_2(x)

        return x

def apply_to_graph(model, graph, parameters, apply_thresholds = True):
    loader = SingleLoader(graph)
    preds = model.predict(loader.load(), steps=loader.steps_per_epoch)
    
    # hinge uses [-1,1] scale of predictons, move them to [0,1] scale
    if parameters["loss_function"] == "squaredhinge":
        preds = (preds + 1) / 2.0
    # if any values outside of [0,1], clip them
    preds = np.clip(preds, 0, 1)

    if apply_thresholds:
        # apply thresholds 
        preds = thresholds.apply_thresholds(preds, parameters)
        # and just in case of any rounding errors, one more round of clipping
        preds = np.clip(preds, 0, 1)

    return preds
