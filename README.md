# Graph Stochastic Attention (GSAT)
The official implementation of Graph Stochastic Attention (GSAT) for our paper: [Interpretable and Generalizable Graph Learning via Stochastic Attention Mechanism](https://arxiv.org/abs/2201.12987), to appear in ICML 2022.

## Introduction
Commonly used attention mechanisms do not impose any constraints during training (besides normalization), and thus may lack interpretability. GSAT is a novel attention mechanism for building interpretable graph learning models. It injects stochasticity to learn attention, where a higher attention weight means a higher probability of the corresponding edge being kept during training. Such a mechanism will push the model to learn higher attention weights for edges that are important for prediction accuracy, which provides interpretability. To further improve the interpretability for graph learning tasks and avoid trivial solutions, we derive regularization terms for GSAT based on the information bottleneck (IB) principle. As a by-product, IB also helps model generalization. Fig. 1 shows the architecture of GSAT.

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
conda install -y pytorch==1.10.0 torchvision cudatoolkit=11.3 -c pytorch
pip install torch-scatter==2.0.9 torch-sparse==0.6.12 torch-cluster==1.5.9 torch-spline-conv==1.2.1 torch-geometric==2.0.3 -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
pip install -r requirements.txt
```

In case a lower CUDA version is required, please use the following command to install dependencies:
```
conda install -y pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=10.2 -c pytorch
pip install torch-scatter==2.0.9 torch-sparse==0.6.12 torch-cluster==1.5.9 torch-spline-conv==1.2.1 torch-geometric==2.0.3 -f https://data.pyg.org/whl/torch-1.9.0+cu102.html
pip install -r requirements.txt
```


# Run Examples
We provide examples with minimal code to run GSAT in `./example/example.ipynb`. We have tested the provided examples on `Ba-2Motifs (GIN)`, `Mutag (GIN)`  and `OGBG-Molhiv (PNA)`. Yet, to implement GSAT* one needs to load a pre-trained model first in the provided example.

It should be able to run on other datasets as well, but some hard-coded hyperparameters might need to be changed accordingly. To reproduce results for other datasets, please follow the instructions in the following section.

# Reproduce Results
We provide the source code to reproduce the results in our paper. The results of GSAT can be reproduced by running `run_gsat.py`. To reproduce GSAT*, one needs to run `pretrain_clf.py` first and change the configuration file accordingly (`from_scratch: false`).

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
No, GSAT doesn't encourage generating sparse subgraphs. We find `r = 0.7` (Eq.(9) in our paper) can generally work well for all datasets in our experiments, which means during training roughly `70%` of edges will be kept (kind of still dense). This is because GSAT doesn't try to provide interpretability by finding a small/sparse subgraph of the original input graph (this is what previous works normally do). Instead, it provides interpretability by pushing the critical edges to have relatively lower stochasticity during training.

#### How to choose the value of `r`?
A grid search in `[0.5, 0.6, 0.7, 0.8, 0.9]` is recommended, but `r = 0.7` is a good starting point. Note that in practice we would decay the value of `r` gradually during training from `0.9` to the chosen value.

#### `p` or `α` in Eq.(9)?
Recall in Fig. 1, `p` is the probability of dropping an edge, while `α` is the sampled result from `Bern(p)`. In our provided implementation, as an empirical choice, `α` is used to implement Eq.(9). We find that when `α` is used it may provide more regularization and makes the model more robust to hyperparameters. Nonetheless, using `p` can achieve the same performance, but it needs some more tuning.

#### Can you show an example of how GSAT works?
Below we show an example from the `ba_2motifs` dataset, which is to distinguish five-node cycle motifs (left) and house motifs (right).
To make good predictions (minimize the cross-entropy loss), GSAT will push the attention weights of those critical edges to be relatively large (ideally close to `1`). Otherwise, those critical edges may be dropped too frequently and thus result in a large cross-entropy loss. Meanwhile, to minimize the regularization loss (the KL divergence term in Eq.(9) of the paper), GSAT will push the attention weights of other non-critical edges to be close to `r`, which is set to be `0.7` in the example. This mechanism of injecting stochasticity makes the learned attention weights from GSAT directly interpretable, since the more critical an edge is, the larger its attention weight will be (the less likely it can be dropped). Note that `ba_2motifs` satisfies our Thm. 4.1 with no noise, and GSAT achieves perfect interpretation performance on it.

<p align="center"><img src="./data/example.png" width=85% height=85%></p>
<p align="center"><em>Figure 2.</em> An example of the learned attention weights.</p>
