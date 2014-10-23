#!/bin/bash

# This script assumes you have previously run the Gale example script including
# the optional part local/online/run_online_decoding_nnet2.sh.  It builds a
# neural net for online decoding on top of the network we previously trained on
# Gale, by keeping everything but the last layer of that network and then
# training just the last layer on our data.  We then train the whole thing.

stage=0
set -e

train_stage=-10
use_gpu=true
gale_exp_dir=../../gale_arabic/s5/exp

. cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if $use_gpu; then
  if ! cuda-compiled; then
    cat <<EOF && exit 1 
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA 
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.  Otherwise, call this script with --use-gpu false
EOF
  fi
  parallel_opts="-l gpu=1" 
  num_threads=1
  minibatch_size=512
  dir=exp/nnet2_online_gale/nnet_a
  trainfeats=exp/nnet2_online_gale/gale_activations_train
  # later we'll change the script to download the trained model from kaldi-asr.org.
  srcdir=$gale_exp_dir/nnet2_online/nnet_a_gpu_online
else
  # Use 4 nnet jobs just like run_4d_gpu.sh so the results should be
  # almost the same, but this may be a little bit slow.
  num_threads=16
  minibatch_size=128
  parallel_opts="-pe smp $num_threads" 
  dir=exp/nnet2_online_gale/nnet_a
  trainfeats=exp/nnet2_online_gale/gale_activations_train
  srcdir=$gale_exp_dir/nnet2_online/nnet_a_online
fi
    
date=$(date +'%m_%d_%H_%M')

if [ $stage -le 0 ]; then
  echo "$0: dumping activations from Gale model"
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $trainfeats/feats/storage ]; then
    # this shows how you can split the data across multiple file-systems; it's optional.
    utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/callhome-egyptian-$date/s5/$trainfeats/feats/storage \
       $trainfeats/feats/storage
  fi
  steps/online/nnet2/dump_nnet_activations.sh --cmd "$train_cmd" --nj 30 \
     data/train $srcdir $trainfeats
fi

if [ $stage -le 1 ]; then
  echo "$0: training 0-hidden-layer model on top of Gale activations"
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then    
    utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/callhome-egyptian-$date/s5/$dir/egs/storage $dir/egs/storage
  fi

  steps/nnet2/retrain_fast.sh --stage $train_stage \
    --num-threads "$num_threads" \
    --minibatch-size "$minibatch_size" \
    --parallel-opts "$parallel_opts" \
    --io-opts "-tc 12" \
    --cmd "$decode_cmd" \
    --num-jobs-nnet 4 \
    --mix-up 4000 \
    --initial-learning-rate 0.02 --final-learning-rate 0.004 \
     $trainfeats/data data/lang exp/tri5a $dir 
fi

if [ $stage -le 2 ]; then
  echo "$0: formatting combined model for online decoding."
  steps/online/nnet2/prepare_online_decoding_retrain.sh $srcdir $dir ${dir}_online
fi

if [ $stage -le 3 ]; then
  # do online decoding with the combined model.
  steps/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 11 \
    exp/tri5a/graph data/dev ${dir}_online/decode_dev &
  wait
fi

if [ $stage -le 4 ]; then
  # do online per-utterance decoding with the combined model.
  steps/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 11 \
     --per-utt true \
    exp/tri5a/graph data/dev ${dir}_online/decode_dev_utt &
  wait
fi

## From this point on we try something else: we try training all the layers of
## the model on this dataset.  First we need to create a combined version of the
## model. 
if [ $stage -le 5 ]; then
  steps/nnet2/create_appended_model.sh $srcdir $dir ${dir}_combined_init
  
  # Set the learning rate in this initial value to our guess of a suitable value.
  initial_learning_rate=0.01
  nnet-am-copy --learning-rate=$initial_learning_rate \
    ${dir}_combined_init/final.mdl ${dir}_combined_init/final.mdl
fi

if [ $stage -le 6 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then    
    utils/create_split_dir.pl \
      /export/b0{1,2,3,4}/$USER/kaldi-data/callhome-egyptian-$date/s5/${dir}_combined/egs/storage \
      $dir_combined/egs/storage
  fi

  # This version of the get_egs.sh script does the feature extraction and iVector
  # extraction in a single binary, reading the config, as part of the script.
  steps/online/nnet2/get_egs.sh --cmd "$train_cmd" --num-jobs-nnet 4 \
    data/train exp/tri5a ${dir}_online ${dir}_combined
fi

if [ $stage -le 7 ]; then
  steps/nnet2/train_more.sh --learning-rate-factor 0.1 --cmd "$train_cmd" \
    --num-threads "$num_threads" \
    --minibatch-size "$minibatch_size" \
    --parallel-opts "$parallel_opts" \
     ${dir}_combined_init/final.mdl  ${dir}_combined/egs ${dir}_combined 
fi

if [ $stage -le 8 ]; then
  # Create an online-decoding dir corresponding to what we just trained above.
  # If this setup used PLP features, we'd have to give the option --feature-type plp
  # to the script below.
  steps/online/nnet2/prepare_online_decoding.sh \
    --mfcc-config $srcdir/conf/mfcc.conf \
    --online-pitch-config $srcdir/conf/online_pitch.conf --add-pitch true \
    data/lang $srcdir/ivector_extractor \
    ${dir}_combined ${dir}_combined_online || exit 1;
fi

if [ $stage -le 9 ]; then
  # do the online decoding on top of the retrained _combined_online model, and
  # also the per-utterance version of the online decoding.
  steps/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 11 \
    exp/tri5a/graph data/dev ${dir}_combined_online/decode_dev &
  steps/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 11 \
    --per-utt true exp/tri5a/graph data/dev ${dir}_combined_online/decode_dev_per_utt &
  wait
fi

exit 0;
