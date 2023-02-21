"""Module for changing setting optimal threshold wrt F1 measure on validation set 
and applying them to new data"""

import numpy as np

import os
import architecture

def set_thresholds(model, graph, weights, parameters, log_dir=None):
    """Set thresholds during training, store in parameters"""
    y_pred = architecture.apply_to_graph(model, graph, parameters, apply_thresholds=False)
    y_true = graph.extract_y()
    # get the first item from dataset iterator
    #(x, y_true, weights) = next(iter(dataset))
    # compute predictions for features x
    #y_pred = model(x, training=False)
    
    # separately handle plasmids and chromosomes
    plasmid_scores = score_thresholds(y_true[:,0], y_pred[:,0], weights)
    store_best(plasmid_scores, parameters, 'plasmid_threshold', log_dir)
    
    chromosome_scores = score_thresholds(y_true[:,1], y_pred[:,1], weights)
    store_best(chromosome_scores, parameters, 'chromosome_threshold', log_dir)


def apply_thresholds(y, parameters):
    """Apply thresholds during testing, return transformed scores so that 0.5 corresponds to threshold"""
    columns = []
    for (column_idx, which_parameter) in [(0, 'plasmid_threshold'), (1, 'chromosome_threshold')]:
        threshold = parameters[which_parameter]
        orig_column = y[:, column_idx]
        # apply the scaling function with different parameters for small and large numbers
        new_column = np.piecewise(
            orig_column,
            [orig_column < threshold, orig_column >= threshold],
            [lambda x : scale_number(x, 0, threshold, 0, 0.5), lambda x : scale_number(x, threshold, 1, 0.5, 1)]
        )
        columns.append(new_column)

    y_new = np.array(columns).transpose()
    return y_new

def scale_number(x, s1, e1, s2, e2):
    """Scale number x so that interval (s1,e1) is transformed to (s2, e2)"""

    factor = (e2 - s2) / (e1 - s1)
    return (x - s1) * factor + s2

def store_best(scores, parameters, which, log_dir):
    """store the optimal threshold for one output in parameter and if requested, print all thresholds to a log file"""
    # scores is a list of pairs threshold, F1 score
    if len(scores) > 0:
        # find index of maximum in scores[*][1]
        maxindex = max(range(len(scores)), key = lambda i : scores[i][1])
        # corrsponding item in scores[*][0] is the threshold
        threshold = scores[maxindex][0]
    else:
        # is input array empty, use default 0.5
        threshold = 0.5
    # store the found threshold
    parameters[which] = float(threshold)

    if log_dir is not None:
        # store thresholds and F1 scores
        filename = os.path.join(log_dir, which + ".csv")
        with open(filename, 'wt') as file:
            print(f"{which},f1", file=file)
            for x in scores:
                print(",".join(str(value) for value in x), file=file)

def score_thresholds(y_true, y_pred, weights):
    """Compute F1 score of all thresholds for one output (plasmid or chromosome)"""
    # compute vector weight and check that all are the same
    length = y_true.shape[0]
    assert tuple(y_true.shape) == (length,)
    assert tuple(y_pred.shape) == (length,)
    assert tuple(weights.shape) == (length,)
    # get data points with non-zero weight
    pairs = []
    for i in range(length):
        if weights[i] > 0:
            pairs.append((y_true[i], y_pred[i]))
    pairs.sort(key=lambda x : x[1], reverse=True)
    
    # count all positives in true labels
    pos = 0
    for pair in pairs:
        if pair[0] > 0.5:
            pos += 1

    scores = []
    tp = 0
    for i in range(len(pairs)):
        # increase true positives if true label is 
        if pairs[i][0] > 0.5:
            tp += 1
        if i > 0 and pairs[i][1] < pairs[i-1][1]:
            recall = tp / pos
            precision = tp / (i+1)
            f1 = 2 * precision * recall / (precision + recall)
            threshold = (pairs[i-1][1] + pairs[i][1]) / 2
            scores.append((threshold, f1))
    
    return scores
