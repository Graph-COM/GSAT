# GSAT
The official implementation of Graph Stochastic Attention (GSAT) for our paper: [Interpretable and Generalizable Graph Learning via Stochastic Attention Mechanism](https://arxiv.org/abs/2201.12987).

## Introduction
Commonly used attention mechanisms do not impose any constraints during training, and thus may lack interpretability. GSAT is a novel attention mechanism to build interpretable graph learning models. It injects stochasticity to learn attention, where a higher attention weight means a higher probability of the corresponding edge being kept during training. Such a mechanism will push the model to learn higher attention weights for edges that are important for prediction accuracy, which provides interpretability. To further improve the interpretability for graph learning tasks and avoid trivial solutions, we derive regularization terms for GSAT based on the information bottleneck (IB) principle. As a by-product, IB also helps model generalization. Figure 1 shows the architecture of GSAT.

<p align="center"><img src="./data/arch.png" width=85% height=85%></p>
<p align="center"><em>Figure 1.</em> The architecture of GSAT.</p>

# Installation
We have tested our code on `Python 3.9` with `PyTorch 1.10.0`, `PyG 2.0.3` and `CUDA 11.3`. Please follow the following steps to create a virtual environment and install the required packages.

Create a virtual environment:
```
conda create --name gsat python=3.9
conda activate gsat
```

Install dependencies:
```
conda install -y pytorch=1.10.0 torchvision cudatoolkit=11.3 -c pytorch
pip install torch-scatter==2.0.9 torch-sparse==0.6.12 torch-cluster==1.5.9 torch-spline-conv==1.2.1 torch-geometric==2.0.3 -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
pip install -r requirements.txt
```

# Run Examples
We provide examples with minimal code to run GSAT in `./example/example.ipynb`. We have tested the provided examples on `Ba-2Motifs (GIN)`, `Mutag (GIN)`  and `OGBG-Molhiv (PNA)`.

It should be able to run on other datasets as well, but we have not tested them. To reproduce results for other datasets, please follow the instructions in the following section.

# Reproduce Results
We provide the source code to reproduce the results in our paper.

To pre-train a classifier:
```
cd ./src
python pretrain_clf.py --dataset [dataset_name] --backbone [model_name] --cuda [GPU_id]
```

To train GSAT:
```
cd ./src
python run_gsat.py --dataset [dataset_name] --backbone [model_name] --cuda [GPU_id]
```

`dataset_name` can be choosen from `ba_2motifs`, `mutag`, `mnist`, `Graph-SST2`, `spmotif_0.5`, `spmotif_0.7`, `spmotif_0.9`, `ogbg_molhiv`, `ogbg_moltox21`, `ogbg_molbace`, `ogbg_molbbbp`, `ogbg_molclintox`, `ogbg_molsider`.

`model_name` can be choosen from `GIN`, `PNA`.

`GPU_id` is the id of the GPU to use. To use CPU, please set it to `-1`.


## Training Logs
Standard output provides basic training logs, while more detailed logs and interpretation visualizations can be found on tensorboard:
```
tensorboard --logdir=./data/[dataset_name]/logs
```

## Hyperparameter Settings
All settings can be found in `./src/configs`.




# Instructions for Acquiring Datasets
- Ba_2Motifs
    - Raw data files can be downloaded automatically, provided by [PGExplainer](https://arxiv.org/abs/2011.04573) and [DIG](https://github.com/divelab/DIG).

- Spurious-Motif
    - Raw data files can be generated automatically, provide by [DIR](https://openreview.net/forum?id=hGXij5rfiHw).

- OGBG-Mol
    - Raw data files can be downloaded automatically, provided by [OGBG](https://ogb.stanford.edu/).

- Mutag
    - Raw data files need to be downloaded [here](https://github.com/flyingdoog/PGExplainer/tree/master/dataset), provided by [PGExplainer](https://arxiv.org/abs/2011.04573).
    - Unzip `Mutagenicity.zip` and `Mutagenicity.pkl.zip`.
    - Put the raw data files in `./data/mutag/raw`.

- Graph-SST2
    - Raw data files need to be downloaded [here](https://drive.google.com/drive/folders/1dt0aGMBvCEUYzaG00TYu1D03GPO7305z), provided by [DIG](https://github.com/divelab/DIG).
    - Unzip the downloaded `Graph-SST2.zip`.
    - Put the raw data files in `./data/Graph-SST2/raw`.

- MNIST-75sp
    - Raw data files need to be generated following the instruction [here](https://github.com/bknyaz/graph_attention_pool/blob/master/scripts/mnist_75sp.sh).
    - Put the generated files in `./data/mnist/raw`.

# FAQ
#### Does GSAT encourage sparsity?
No, GSAT doesn't encourage generating sparse subgraphs. We find `r = 0.7` (Eq.9 in our paper) can generally work well for all datasets in our experiments, which means during training roughly `70%` of edges will be kept (kind of still dense). This is because GSAT doesn't try to provide interpretability by finding a small/sparse subgraph of the original input graph (this is what previous works normally do). Instead, it provides interpretability by pushing the critical edges to have relatively lower stochasticity during training.
