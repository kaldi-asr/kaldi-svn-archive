#!/bin/bash


# This is somewhere to put the data; it can just be a local path if you
# don't want to give it a permanent home.
data=/export/corpora5/NIST/MNIST
export train_cmd="queue.pl -l arch=*64,mem_free=2G,ram_free=2G"  
dnn_gpu_parallel_opts=(--minibatch-size 512 --max-change 40 --num-jobs-nnet 4 --num-threads 1 --parallel-opts "-l gpu=1" ) 

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

# train and test using tanh network. Result: %WER 1.97 [ 197 / 10000, 0 ins, 0 del, 197 sub ]
dir=exp/nnet2
steps/nnet2/train_tanh.sh data/train_train data/train_heldout $dir
steps/nnet2/test.sh data/t10k/ $dir

# train and test using pnorm network. Result: %WER 1.37 [ 137 / 10000, 0 ins, 0 del, 137 sub ]
dir=exp/nnet2_pnorm 
steps/nnet2/train_pnorm.sh --cmd "$train_cmd" --num-hidden-layers 2 --pnorm-input-dim 1200 --pnorm-output-dim 600 --stage -5 \
 "${dnn_gpu_parallel_opts[@]}" data/train_train data/train_heldout $dir 
steps/nnet2/test.sh data/t10k/ $dir

# train and test using ensemble trained pnorm network. Result: %WER 1.26 [ 126 / 10000, 0 ins, 0 del, 126 sub ]
dir=exp/nnet2_ensemble_pnorm
steps/nnet2/train_pnorm_ensemble.sh --cmd "$train_cmd" --num-hidden-layers 2 --pnorm-input-dim 1200 --pnorm-output-dim 600 --final-beta 3 --ensemble-size 2 \
 "${dnn_gpu_parallel_opts[@]}" data/train_train data/train_heldout $dir 
steps/nnet2/test.sh data/t10k/ $dir
