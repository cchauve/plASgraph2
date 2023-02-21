import itertools
from Bio.Seq import Seq
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def get_node_id(sample_id, contig_id):
    return f"{sample_id}:{contig_id}"

def label_to_pair(label):
    if label == "chromosome":
        return [0, 1]
    elif label == "plasmid":
        return [1, 0]
    elif label == "ambiguous":
        return [1, 1]
    elif (label == "unlabelled" or label == "no_label" 
          or label == "unlabeled" or label is None):
        return [0,0]
    else:
        raise AssertionError("bad label {label}")

def pair_to_label(pair):
    if pair == [0, 1]:
        return "chromosome"
    elif pair == [1, 0]:
        return "plasmid"
    elif pair == [1, 1]:
        return "ambiguous"
    elif pair == [0, 0]:
        return "unlabeled"
    else:
        raise AssertionError("bad pair {pair}")
        
def prepare_kmer_lists(kmer_length):
    k_mers = ["".join(x) for x in itertools.product("ACGT", repeat=kmer_length)]

    fwd_kmers = []
    fwd_kmer_set = set()
    rev_kmer_set = set()
    
    for k_mer in k_mers:
        if not ((k_mer in fwd_kmer_set) or (k_mer in rev_kmer_set)):
            fwd_kmers.append(k_mer)
            fwd_kmer_set.add(k_mer)
            rev_kmer_set.add(str(Seq(k_mer).reverse_complement()))

    return (k_mers, fwd_kmers)


def get_kmer_distribution(sequence, kmer_length=5, scale=False):
    assert kmer_length % 2 == 1
    (k_mers, fwd_kmers) = prepare_kmer_lists(kmer_length)
    
    dict_kmer_count = {}
    for k_mer in k_mers:
        dict_kmer_count[k_mer] = 0.01 # pseudocounts

    for i in range(len(sequence) + 1 - kmer_length):
        kmer = sequence[i : i + kmer_length]
        if kmer in dict_kmer_count:
            dict_kmer_count[kmer] += 1

    k_mer_counts = [
        dict_kmer_count[k_mer] + dict_kmer_count[str(Seq(k_mer).reverse_complement())]
        for k_mer in fwd_kmers
    ]

    if scale:
        ksum = sum(k_mer_counts)
        k_mer_counts = [ x / ksum for x in k_mer_counts ]
        
        #scaler = MinMaxScaler()
        #k_mer_counts = scaler.fit_transform(np.array(k_mer_counts).reshape(-1, 1))
        #k_mer_counts = list(k_mer_counts.flatten())

    return k_mer_counts


def get_gc_content(seq):
    number_gc = 0
    number_acgt = 0
    for base in seq:
        if base in "GC":
            number_gc += 1
        if base in "ACGT":
            number_acgt += 1

    if number_acgt > 0:
        gc_content = round(number_gc / number_acgt, 4)
    else:
        gc_content = 0.5
    return gc_content
