
# MSS-PAE: Saving Autoencoder-based Outlier Detection from Unexpected Reconstruction

This repo contains source codes for the paper [**MSS-PAE: Saving Autoencoder-based Outlier Detection from Unexpected Reconstruction**](https://www.sciencedirect.com/science/article/abs/pii/S003132032500127X).

  

## Software Requirement

* Python 3.7
* numpy 1.19.5
* pandas 1.3.3
* scikit-learn 0.22
* scipy 1.7.1
* torch 1.9.1
  

## Get started

* Prepare the dataset.
    * Place the dataset in *./dataset/original/* \
        For example:
        ```
        ├── original/
            └── Optdigits/
                ├── Optdigits_data.txt
                └── Optdigits_label.txt
        ```
        **The dataset name should be consistent everywhere in this project.**

    * Open *./data_pre.py* \
        Edit "data_dir", "target_dir", and "dataset" according to your setting in the previous step.

    * Run *python ./data_pre.py*

* Do main experiments.
    * Open *./run_main.sh* \
        Edit the experimental settings. \
        --gpu: Training on which GPU. \
        --dataset: Name of the dataset \
        --data_dir: Directory of data \
        --epochs: Number of training epochs \
        --result_dir: Directory to dump results \
        --net: Which network to use: AE; PAE \
        --alpha: Hyper-paramter alpha for PAE \
        --beta: Hyper-paramter beta for PAE \
        --inits: How many random initial states for step 2.

    * Run *bash ./run_main.sh*

## Dataset
All datasets used can be found in [OutlierNet](http://www.OutlierNet.com).


## Citation
```
@article{TAN2025111467,
    title = {MSS-PAE: Saving Autoencoder-based Outlier Detection from Unexpected Reconstruction},
    journal = {Pattern Recognition},
    volume = {163},
    pages = {111467},
    year = {2025},
    issn = {0031-3203},
    doi = {https://doi.org/10.1016/j.patcog.2025.111467},
    url = {https://www.sciencedirect.com/science/article/pii/S003132032500127X}
}
```
