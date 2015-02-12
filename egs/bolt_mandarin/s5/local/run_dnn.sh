#!/bin/bash

# DNN training recipe based on swithboard tri6_nnet recipe. (It's tuned on CALLHOME data only for now). It's using steps/nnet2/train_pnorm_fast.sh.

dir=tri6_nnet
train_stage=-10

. ./cmd.sh
. ./path.sh
 ! cuda-compiled && cat <<EOF && exit 1 
 This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA 
 If you want to use GPUs (and have them), go to src/, and configure and make on a machine
 where "nvcc" is installed.
EOF


. utils/parse_options.sh
parallel_opts="-l gpu=1"  # This is suitable for the CLSP network, you'll likely have to change it.

( 
  # using tri5a model to align the training data.
  if [ ! -f exp/tri5a_ali/ali.1.gz ]; then
    steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" \
      data/train data/lang exp/tri5a exp/tri5a_ali || exit 1;
  fi

  # train dnn
  if [ ! -f exp/$dir/final.mdl ]; then
    if [ $USER == xzhang ]; then # this shows how you can split across multiple file-systems.
      utils/create_split_dir.pl /export/b0{1,2,3,4}/xzhang/kaldi-online/egs/bolt/20141103/$dir/egs $dir/egs/storage
    fi

    steps/nnet2/train_pnorm_fast.sh --stage $train_stage \
      --num-jobs-nnet 8 --num-threads 1 \
      --minibatch-size 128 --parallel-opts "$parallel_opts" \
      --mix-up 8000 \
      --initial-learning-rate 0.08 --final-learning-rate 0.008 \
      --samples-per-iter 400000 \
      --num-hidden-layers 4 \
      --stage $train_stage \
      --pnorm-input-dim 3000 \
      --pnorm-output-dim 500 \
      --cmd "$decode_cmd" \
      data/train data/lang exp/tri5a_ali exp/$dir || exit 1;
    touch exp/$dir/.done
  fi

  ./local/run_anydecode.sh bolt_dev &
)
