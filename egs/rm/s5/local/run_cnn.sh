#!/bin/bash

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

. ./path.sh ## Source the tools/utils (import the queue.pl)

dev=data-fbank/test
train=data-fbank/train

dev_original=data/test
train_original=data/train

#
# Make the FBANK features
#
# false && \
{
# Dev set
mkdir -p $dev && cp $dev_original/* $dev
steps/make_fbank_pitch.sh --nj 10 --cmd "$train_cmd" \
   $dev $dev/log $dev/data || exit 1;
steps/compute_cmvn_stats.sh $dev $dev/log $dev/data || exit 1;
# Training set
mkdir -p $train && cp $train_original/* $train
steps/make_fbank_pitch.sh --nj 10 --cmd "$train_cmd -tc 10" \
   $train $train/log $train/data || exit 1;
steps/compute_cmvn_stats.sh $train $train/log $train/data || exit 1;
# Split the training set
utils/subset_data_dir_tr_cv.sh --cv-spk-percent 10 $train ${train}_tr90 ${train}_cv10
}

#
# Run the CNN training.
#

dir=exp/cnn4a
gmm=exp/tri3b
ali=${gmm}_ali
# false && \
{
 $cuda_cmd $dir/log/train_nnet.log \
  steps/nnet/train.sh \
    --apply-cmvn true --norm-vars true --delta-order 2 --splice 5 \
    --learn-rate 0.008 --train-opts "--verbose 2" \
    --convolutional true --cnn-init-opts "--pitch-dim 3" \
    ${train}_tr90 ${train}_cv10 data/lang $ali $ali $dir || exit 1;
}

#
# Pre-train stack of RBMs (6 layers, 2048 units) on top of convolutional part of network
#

dir=exp/cnn4a_pretrain-dbn
cnn=exp/cnn4a/final.nnet
nn_depth=6
# false && \
{ 
  $cuda_cmd $dir/log/pretrain_dbn.log \
    steps/nnet/pretrain_dbn.sh --cnn $cnn --nn-depth 6 --rbm-iter 10 \
     --apply-cmvn true --norm-vars true --splice 5 --delta_order 2 \
     --rbm-lrate-low 0.005 --rbm-lrate 0.2 \
     --param-stddev-first 0.026 --param-stddev 0.045 \
     $train $dir || exit 1
}



#
# Train the DNN optimizing cross-entropy.
# This will take quite some time.
#

dir=exp/cnn4a_pretrain-dbn_dnn
feature_transform=exp/cnn4a_pretrain-dbn/final.feature_transform
dbn=exp/cnn4a_pretrain-dbn/${nn_depth}.dbn
feature_transform=exp/cnn4a/final.feature_transform
# false && \
{

  # Train
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train.sh --feature-transform $feature_transform --cnn $cnn --delta-order 2 --splice 5  --dbn $dbn --hid-layers 0 \
    ${train}_tr90 ${train}_cv10 data/lang $ali $ali $dir || exit 1;
}


dir=exp/cnn4a_pretrain-dbn_dnn
# Decode (reuse HCLG graph)
 steps/nnet/decode.sh --nj 20 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.1 \
   $gmm/graph $dev $dir/decode || exit 1;


# Sequence training using sMBR criterion, we do Stochastic-GD 
# with per-utterance updates. We use usually good acwt 0.1
dir=exp/cnn4a_pretrain-dbn_dnn_smbr
srcdir=exp/cnn4a_pretrain-dbn_dnn
acwt=0.1

# First we generate lattices and alignments:
steps/nnet/align.sh --nj 20 --cmd "$train_cmd" \
  $train data/lang $srcdir ${srcdir}_ali || exit 1;
steps/nnet/make_denlats.sh --nj 20 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt $acwt \
  $train data/lang $srcdir ${srcdir}_denlats || exit 1;

# Re-train the DNN by 6 iterations of sMBR 
steps/nnet/train_mpe.sh --cmd "$cuda_cmd" --num-iters 6 --acwt $acwt --do-smbr true \
  $train data/lang $srcdir ${srcdir}_ali ${srcdir}_denlats $dir || exit 1
# Decode
for ITER in 1 2 3 4 5 6; do
  steps/nnet/decode.sh --nj 20 --cmd "$decode_cmd" --config conf/decode_dnn.config \
    --nnet $dir/${ITER}.nnet --acwt $acwt \
    $gmm/graph $dev $dir/decode_it${ITER} || exit 1
done 

echo Success
exit 0




