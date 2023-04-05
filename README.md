
# Improving Autoencoder-based Outlier Detection with Adjustable Probabilistic Reconstruction Error and Mean-shift Outlier Scoring

This repo contains source codes for the paper [**Improving Autoencoder-based Outlier Detection with Adjustable Probabilistic Reconstruction Error and Mean-shift Outlier Scoring**](http://arxiv.org/abs/2304.00709) submitted to IEEE Transactions on Neural Networks and Learning Systems (TNNLS).

  

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
All datasets used can be found in [OutlierNet](https://www.OutlierNet.com).


## Citation
```
@misc{tan2023improving,
    title={Improving Autoencoder-based Outlier Detection with Adjustable Probabilistic Reconstruction Error and Mean-shift Outlier Scoring},
    author={Xu Tan and Jiawei Yang and Junqi Chen and Sylwan Rahardja and Susanto Rahardja},
    year={2023},
    eprint={2304.00709},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
