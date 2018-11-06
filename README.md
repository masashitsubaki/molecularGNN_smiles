# GNN_molecules

This is a code for molecular property prediction (MPP) using a graph neural network (GNN) based on r-radius subgraph (or called fingerprint) embeddings. This GNN is proposed in our paper "[Compound-protein Interaction Prediction with End-to-end Learning of Neural Networks for Graphs and Sequences (Bioinformatics, 2018)](https://academic.oup.com/bioinformatics/advance-article-abstract/doi/10.1093/bioinformatics/bty535/5050020?redirectedFrom=PDF)," which aims to predict compound-protein interactions. In this implementation, we provide our GNN to predict various molecular properties such as drug efficacy and photovoltaic efficiency.

The basic idea of a GNN is as follows:

<div align="center">
<p><img src="basic_GNN.jpeg" width="500" /></p>
</div>

Our GNN is based on the r-radius (or fingerprint) embeddings as follows:

<div align="center">
<p><img src="our_GNN.jpeg" width="500" /></p>
</div>


## How to cite

```
@article{tsubaki2018compound,
  title={Compound-protein Interaction Prediction with End-to-end Learning of Neural Networks for Graphs and Sequences},
  author={Tsubaki, Masashi and Tomii, Kentaro and Sese, Jun},
  journal={Bioinformatics},
  year={2018}
}
```


## Requirements

- PyTorch (version 0.4.0)
- scikit-learn
- RDKit


## Usage

We provide two major scripts:

- code/classification or regression/preprocess_data.py creates the input tensor data of molecules for processing with PyTorch from the original data (see dataset/classification or regression/original/smiles_property.txt).
- code/classification or regression/run_training.py trains our neural network using the above preprocessed to predict molecular properties.

(i) Create the tensor data of molecules and their properties with the following command:
```
cd code/classification (or cd code/regression)
bash preprocess_data.sh
```

(ii) Using the preprocessed data, train our neural network with the following command:
```
bash run_training.sh
```

The training result and trained model are saved in the output directory (after training, see output/result and output/model).

(iii) You can change the hyperparameters in preprocess_data.sh (e.g., radius) and run_training.sh (e.g., dimensionality, batch size, the number of layers, and learning rate), and try to learn various models!


## Training of our neural network using your molecular property dataset
In this repository, we provide a dataset of classification (see dataset/classification/HIV)
and regression (see dataset/regression/photovoltaic).
If you prepare dataset with the same format as "smiles_property.txt" in a new directory,
you can train our neural network using your dataset by the above two commands (i) and (ii).
