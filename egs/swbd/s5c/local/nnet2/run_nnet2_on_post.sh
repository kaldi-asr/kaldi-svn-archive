#!/bin/bash

# Copyright 2015  Xiaohui Zhang
# Apache 2.0

# This script shows an example of generating posteriors from a DNN, interpolating the posteriors with hard labels, and then training another DNN on top of the interpolated targets.
has_fisher=true
stage=0
. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

parallel_opts="-l gpu=1"  # This is suitable for the CLSP network, you'll
                          # likely have to change it.

data_post=data/train_nodup  # Data on which we generate posteriors
transform_dir=exp/tri4_ali_nodup
nnet_dir=exp/nnet2_5  # This if the DNN where we get posteriors, not the DNN we will train on top of the posteriors
post_dir=exp/combined_1hard_1nnet_post  # If you don't want to combine posteriors with hard labels, just set post_dir as ${nnet_dir}_post

if [ $stage -le 0 ]; then
  echo ---------------------------------------------------------------------
  echo "$0: Generate posteriors and combine them with hard labels."
  echo ---------------------------------------------------------------------
  ./steps/nnet2/get_post.sh --transform_dir $transform_dir $data_post $nnet_dir ${nnet_dir}_post
  # steps/post/get_sgmm_post.sh --stage 1 --cmd "$train_cmd" --transform_dir $transform_dir $data_post data/lang exp/sgmm2_5_mmi_b0.1 exp/sgmm2_5_mmi_b0.1_post 
  # steps/post/get_pdf_mapping.sh exp/sgmm2_5_mmi_b0.1_post exp/tri4_ali_nodup exp/map_sgmm2_to_tri4
  # steps/post/apply_pdf_mapping.sh --cmd "$train_cmd" exp/map_sgmm2_to_tri4 exp/sgmm2_5_mmi_b0.1_post exp/sgmm2_tri4_post
  # steps/post/combine_pdf_post.sh exp/tri3b_ali exp/sgmm2_tri3b_post exp/nnet4d_new_post exp/tri3b_sgmm2_nnet4d_combined_post
  steps/post/combine_pdf_post.sh --resplit-nj 100 exp/tri4_ali_nodup:1 ${nnet_dir}_post:1 $post_dir
  
  cp $transform_dir/num_jobs $post_dir
  nj=`cat $transform_dir/num_jobs`
  cat $post_dir/post.*.scp | sort -u > $post_dir/post_all.scp  
  for i in `seq 1 $nj`; do
    cat $data_post/split${nj}/$i/feats.scp | cut -f1 -d ' ' > ./uttlist
    grep -F -f ./uttlist $post_dir/post_all.scp > $post_dir/post.$i.scp
  done  
fi

if [ $stage -le 1 ]; then
  echo ---------------------------------------------------------------------
  echo "$0: Train DNN on top of the posteriors"
  echo ---------------------------------------------------------------------
  dir=exp/nnet_on_combined_1hard_1nnet
  if [ ! -f $dir/final.mdl ]; then
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
      # spread the egs over various machines. 
      utils/create_split_dir.pl \
      /export/b0{1,2,3,4}/$USER/kaldi-data/egs/swbd-$(date +'%m_%d_%H_%M')/s5c/$dir/egs/storage $dir/egs/storage
    fi

    steps/nnet2/train_pnorm_accel2.sh --parallel-opts "$parallel_opts" \
      --cmd "$decode_cmd" --stage -10 \
      --cleanup false \
      --postdir $postdir \
      --num-threads 1 --minibatch-size 512 \
      --mix-up 20000 --samples-per-iter 300000 \
      --num-epochs 15 \
      --initial-effective-lrate 0.005 --final-effective-lrate 0.0002 \
      --num-jobs-initial 3 --num-jobs-final 10 --num-hidden-layers 5 \
      --pnorm-input-dim 5000  --pnorm-output-dim 500 data/train_nodup \
      data/lang exp/tri4_ali_nodup $dir || exit 1;
  fi

  steps/nnet2/decode.sh --cmd "$decode_cmd" --nj 30 \
    --config conf/decode.config \
    --transform-dir exp/tri4/decode_eval2000_sw1_tg \
    exp/tri4/graph_sw1_tg data/eval2000 \
    $dir/decode_eval2000_sw1_tg || exit 1;

  if [ $has_fisher ]; then
    steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
      data/lang_sw1_{tg,fsh_fg} data/eval2000 \
      $dir/decode_eval2000_sw1_{tg,fsh_fg} || exit 1;
  fi
fi
