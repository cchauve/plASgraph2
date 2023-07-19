# plASgraph2 - Classifying Plasmid Contigs From Bacterial Assembly Graphs Using Graph Neural Networks

## Overview

Identification of plasmids and plasmid genes from sequencing data is an important question regarding antimicrobial resistance spread and other One-Health issues. PlASgraph2 is a deep-learning tool that classifies contigs from a short-read assembly as originating either from a plasmid, the chromosome or being ambiguous (i.e. could originate from both, e.g. in the case of a shared repeated contig). 

PlASgraph2 is built on a graph neural network (GNN) and analysis the **assembly graph** (provided in <a href="http://gfa-spec.github.io/GFA-spec/">GFA format</a>) generated by an assembler such as <a href="https://github.com/rrwick/Unicycler">Unicycler</a> or <a href="https://github.com/ncbi/SKESA">SKESA</a>. 

<p align="center">
  <img src="/doc/plASgraph2_architecture.png" alt="drawing" width="600"/>
</p>

This distribution of PlASgraph2 is provided with a model trained on data from the ESKAPEE group of bacterial pathogens (*Enterococcus faecium*, *Staphylococcus aureus*, *Klebsiella pneumoniae*, *Acinetobacter baumannii*, *Pseudomonas aeruginosa*, *Enterobacter spp.*, and *Escherichia coli*). PlASgraph2 is species-agnostic, so the provided trained model can be applied to analyse data from other pathogen species. Alternativly, plASgraph can be trained on a new training dataset (see section **Training** below).

## Installation
PlASgraph2 can be installed from this repository 

~~~
git clone https://github.com/cchauve/plASgraph2.git
~~~

PlASgraph2 is written in python 3 and has been developed and tested with the following modules.
  - Python 3.8.10
  - NetworkX  2.8.3
  - Pandas  1.4.1
  - NumPy  1.22.2
  - Scikit-learn  1.0.2
  - Scipy 1.8.0
  - Biopython  1.79
  - Matplotlib  3.5.1
  - TensorFlow  2.8.0
  - Spektral  1.1.0
  - PyYAML 6.0  
 
All modules can be installed using pip (https://docs.python.org/3.8/installing/index.html) and we strongly recommand to run plASgraph2 using a dedicated python virtual environment (see https://docs.python.org/3.8/library/venv.html).
    
## Training

Training a plASgraph2 model requires (1) assembly graphs in gzipped GFA format for the training samples and (2) a labeling of the training samples contigs as either *plasmid*, *chromosome*, *ambiguous* (contigs that appear in both a plasmid and the chromosome) or *unlabeled* (typically very short contigs).

The training input consists of two files:
- a *configuration file* in <a href="https://yaml.org/">YAML</a> format, that specifies training parameters
(default file: [model/config_default.yaml](./model/config_default.yaml));
- a *CSV samples file*, with no header line, that contains one line per sample, specifying (1) the path to the gzipped GFA assembly file for the sample, (2) the path for a contig labels CSV file, and (3) a sample name (example: [model/eskapee-train.csv](./model/eskapee-train.csv), taken from the github repo that contains all training data used to train plASgraph2 models, [plasgraph2-datasets](https://github.com/fmfi-compbio/plasgraph2-datasets)).

Files path in the CSV training file are assumed to be relative, with the prefix of the path for each file being provided as a command-line parameter (see example of command-line below). This assumption implies that all GFA and CSV training files are located in the same directory (although they can be located in different subdirectories).

For example, to re-train plASgraph2 with the training data in `example/ESKAPEE_train.csv` one would run the command
```
python ./src/plASgraph2_train.py training_config.yaml example/ESKAPEE_train.csv training_data_dir output_model_dir > model_train.log 2> model_train.err
```

The first training sample in `example/ESKAPEE_train.csv` is described by the line 
```boostrom/abau-SAMEA12292436/short.gfa.gz,boostrom/abau-SAMEA12292436/short.gfa.csv,boostrom_abau-SAMEA12292436-u```
where:
- the gzipped GFA assembly graph file is the file `short.gfa.gz` in the subdirectory `boostrom/abau-SAMEA12292436/` of the data directory `training_data_dir`;
- the contig labels file is the file `short.gfa.csv` in the subdirectory `boostrom/abau-SAMEA12292436/` of the data directory `training_data_dir`;
- the sample name is `boostrom_abau-SAMEA12292436-u`.

**TO DO:** Describe the format of the contigs labels CSV file. Are all the fields (*contig,plasmid_score,chrom_score,label,length,chr_coverage,pl_coverage,un_coverage,hybrid_mapsto*) necessary? 

**QUESTION.** Is-there some preprocessing required for the GFA files depending if they were generated by SKESA or UniCycler?

The result is created in the directory `output_model_dir`, while files `model_train.log`, `model_train.err` record the log and possible errors that occured duing training. The model is provided in the file `output_model_dir/saved_model.pb`.

Additional options `-g` and `-l` allow respectively to generate the GNN as a file in GML format and to generate additional log files.

The directory `model/ESKAPEE_model/` contains the model trained on ESKAPEE pathogens, from data listed in the file `model/ESKAPEE_train.csv`, where each sample was assembled using Unicycler (files `*.short.gfa.gz`) and SKESA (files `*.skesa.gfa.gz`).

**TO DO.** We should deposit the assemblies in a repository (zenodo, gitLFS?).

**QUESTION.** Are-there other technical details we should mention (GPU?)?

## Classification

The input for plASgraph2 consists in a trained model and either a single assembly graph from a single bacterial sample in gzipped <a href="http://gfa-spec.github.io/GFA-spec/">GFA (.gfa) format</a> or a CSV file with a list of gzipped GFA files to analyze.

PlASgraph2 has been trained and tested on assembly graphs generated by the assemblers <a href="https://github.com/rrwick/Unicycler">Unicycler</a> and <a href="https://github.com/ncbi/SKESA">SKESA</a>.

Given a single gzipped GFA file `assembly_graph.gfa.gz`, located in directory `data_dir` and a model located in directory `model_dir`, the contigs of the sample can be classified using the command

```
python ./src/plASgraph2_classify.py gfa assembly_graph.gfa.gz data_dir model_dir output.csv
```

The result is written in a file `output.csv` that contains one line per contig, recording its length, plasmid score, chromosome score and final label.

To classify contigs of several samples at once, the input file is a CSV file `input.csv`, with one line per sample, the first field being the name of the gzippeed assembly graph file, the second is currently not used, and the last field is the name of the sample. 
All assembly graphs files listed in the file are assumed to be located in the same directory `data_dir`. 
The samples contigs can then be classified using the command

```
python ./src/plASgraph2_classify.py set input.csv data_dir model_dir output.csv
```

As in the previous case, `output.csv` is a CSV file containing the results for all contigs of all samples.

The directory `example` contains an example that has been generated by the command

```
python ../src/plASgraph2_classify.py set SAMN15148288_input.csv ./ ../model/ESKAPEE_model/ SAMN15148288_output.csv
```

**TO DO.** Describe the format of the output CSV file.

**TO DO.** Describe options to the classificatio sript if any.

**TO DO.** Uniformize the CSV formats to have all files with a header line and the same field names.

## Citation
Janik Sielemann, Katharina Sielemann, Broňa Brejová, Tomas Vinar, Cedric Chauve; "plASgraph2: Using Graph Neural Networks to Detect Plasmid Contigs from an Assembly Graph", in preparation, 2023.
