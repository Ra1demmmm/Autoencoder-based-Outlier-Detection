#!/usr/bin/env bash

gpu=(0)
dataset='Optdigits'
data_dir='./dataset/processed'
epochs=5000
net='PAE'
alpha=0.5
beta=2
inits=20

result_dir='./result/pae/'${dataset[i]}
python ./main.py \
--gpu ${gpu[i]} \
--dataset $dataset \
--data_dir $data_dir \
--epochs $epochs \
--result_dir $result_dir \
--net $net \
--alpha $alpha \
--beta $beta \
--inits $inits


