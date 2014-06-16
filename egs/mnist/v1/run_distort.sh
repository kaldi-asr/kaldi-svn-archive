#!/bin/bash

# This is not the top-level run.sh as it is in other directories.
# The difference between this script and run.sh is that it uses perturbed training data
# during nnet trainig.


[ -f ./path.sh ] && . ./path.sh; # source the path.

# JHU cluster options
export train_cmd="queue.pl -l arch=*64,mem_free=2G,ram_free=2G"  

[ ! -f ./conf/image.conf ] && echo 'Image trainig configuration does not exist!' && exit 1;
. conf/image.conf || exit 1;

local/download_data_if_needed.sh $data || exit 1;

local/format_data.sh $data

mkdir -p data/train_train data/train_heldout
for f in feats.scp labels; do
  head -n $heldout_size data/train/$f > data/train_heldout/$f
  tail -n +$[$heldout_size+1] data/train/$f > data/train_train/$f
done

for x in train train_heldout train_train t10k; do
  utils/validate_data_dir.sh data/$x || exit 1;
done

# train and test using tanh network. Result: %WER between (1.15, 1.5), [ 115-150 / 10000, 0 ins, 0 del, 115-150 sub ]
dir=exp_distort/nnet2
if [ ! -f $dir/.done ]; then
  steps/nnet2/train_tanh.sh --use-distortion $use_distortion --num-epochs $num_epochs \
    --num-epochs-extra $num_epochs_extra data/train_train data/train_heldout $dir
  touch $dir/.done
fi
steps/nnet2/test.sh data/t10k/ $dir

# train and test using distorted pnorm network. Result: %WER 0.8 [ 80 / 10000, 0 ins, 0 del, 80 sub ]
dir=exp_distort/nnet2_pnorm 
if [ ! -f $dir/.done ]; then
  steps/nnet2/train_pnorm.sh --cmd "$train_cmd" --use-distortion $use_distortion --num-epochs $num_epochs \
    --num-epochs-extra $num_epochs_extra --num-hidden-layers $num_hidden_layers \
    --pnorm-input-dim $pnorm_input_dim --pnorm-output-dim $pnorm_input_dim \
   "${dnn_gpu_parallel_opts[@]}" data/train_train data/train_heldout $dir 
  touch $dir/.done
fi
steps/nnet2/test.sh data/t10k/ $dir

# train and test using ensemble trained pnorm network. Result: %WER 1.26 [ 126 / 10000, 0 ins, 0 del, 126 sub ]
dir=exp_distort/nnet2_ensemble_pnorm
if [ ! -f $dir/.done ]; then  
  steps/nnet2/train_pnorm_ensemble.sh --cmd "$train_cmd" --num-hidden-layers $ens_num_hidden_layers \
  --pnorm-input-dim $ens_pnorm_input_dim --pnorm-output-dim $ens_pnorm_output_dim \
  --final-beta $final_beta --ensemble-size $ensemble_size \
   "${dnn_gpu_parallel_opts[@]}" data/train_train data/train_heldout $dir 
  touch $dir/.done
fi
steps/nnet2/test.sh data/t10k/ $dir

