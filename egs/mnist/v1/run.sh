#!/bin/bash


# This is somewhere to put the data; it can just be a local path if you
# don't want to give it a permanent home.
data=/export/corpora5/NIST/MNIST

local/download_data_if_needed.sh $data || exit 1;

local/format_data.sh $data

mkdir -p data/train_train data/train_heldout
heldout_size=1000
for f in feats.scp labels; do
  head -n $heldout_size data/train/$f > data/train_heldout/$f
  tail -n +$[$heldout_size+1] data/train/$f > data/train_train/$f
done

for x in train train_heldout train_train t10k; do
  utils/validate_data_dir.sh data/$x || exit 1;
done

dir=exp/nnet2
steps/nnet2/train_tanh.sh data/train_train data/train_heldout $dir
steps/nnet2/test.sh data/t10k/ $dir

