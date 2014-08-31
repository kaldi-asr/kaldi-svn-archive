#!/bin/bash

# DNN training recipe based on swithboard nnet5d_gpu recipe. (It's tuned on CALLHOME data only for now). It's using steps/nnet2/train_pnorm_fast.sh.

dir=nnet5d_gpu
temp_dir=
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
    if [ ! -z "$temp_dir" ] && [ ! -e exp/$dir/egs ]; then
      mkdir -p exp/$dir
      mkdir -p $temp_dir/$dir/egs
      ln -s $temp_dir/$dir/egs exp/$dir/
    fi

    steps/nnet2/train_pnorm_fast.sh --stage $train_stage \
      --num-jobs-nnet 2 --num-threads 1 \
      --minibatch-size 128 --parallel-opts "$parallel_opts" \
      --mix-up 8000 \
      --initial-learning-rate 0.02 --final-learning-rate 0.004 \
      --samples-per-iter 400000 \
      --num-hidden-layers 3 \
      --pnorm-input-dim 1000 \
      --pnorm-output-dim 200 \
      --cmd "$decode_cmd" \
      data/train data/lang exp/tri5a_ali exp/$dir || exit 1;
  fi

    steps/nnet2/decode.sh --cmd "$decode_cmd" --nj 30 \
      --config conf/decode.config --transform-dir exp/tri5a/decode_dev \
      exp/tri5a/graph data/dev exp/$dir/decode_dev
)

