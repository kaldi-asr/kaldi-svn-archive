#!/bin/bash


# example script for online-nnet2 system training and decoding.
# based on the one for fisher-English.

. cmd.sh


stage=0
train_stage=-10
use_gpu=true
set -e
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
  # the _a is in case I want to change the parameters.
  dir=exp/nnet2_online/nnet_a_gpu 
else
  # Use 4 nnet jobs just like run_4d_gpu.sh so the results should be
  # almost the same, but this may be a little bit slow.
  num_threads=16
  minibatch_size=128
  parallel_opts="-pe smp $num_threads" 
  dir=exp/nnet2_online/nnet_a
fi

if [ $stage -le 0 ]; then
  # create some data subsets.
  # mixed is the clean+other data.
  # 30k is 1/10 of the data (around 100 hours), 60k is 1/5th of it (around 200 hours).
  utils/subset_data_dir.sh data/train_960 30000 data/train_mixed_30k
  utils/subset_data_dir.sh data/train_960 60000 data/train_mixed_60k
fi

if [ $stage -le 1 ]; then
  mkdir -p exp/nnet2_online
  # To train a diagonal UBM we don't need very much data, so use a small subset
  # (actually, it's not that small: still around 100 hours).
  steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj 30 --num-frames 700000 \
    data/train_mixed_30k 512 exp/tri6b exp/nnet2_online/diag_ubm
fi

if [ $stage -le 2 ]; then
  # iVector extractors can in general be sensitive to the amount of data, but
  # this one has a fairly small dim (defaults to 100) so we don't use all of it,
  # we use just the 60k subset (about one fifth of the data, or 200 hours).
  steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" --nj 10 \
    data/train_mixed_60k exp/nnet2_online/diag_ubm exp/nnet2_online/extractor || exit 1;
fi

if [ $stage -le 3 ]; then
  ivectordir=exp/nnet2_online/ivectors_train_960
  if [ $USER == dpovey ]; then # this shows how you can split across multiple file-systems.
    utils/create_split_dir.pl /export/b0{1,2,3,4}/dpovey/kaldi-online/egs/fisher_english/s5/$ivectordir $ivectordir/storage
  fi

  # We extract iVectors on all the train data, which will be what we train the
  # system on.  With --utts-per-spk-max 2, the script.  pairs the utterances
  # into twos, and treats each of these pairs as one speaker.  Note that these
  # are extracted 'online'.
  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 60 \
    --utts-per-spk-max 2 data/train_960 exp/nnet2_online/extractor $ivectordir || exit 1;
fi


if [ $stage -le 4 ]; then
  if [[ $(hostname -f) ==  *.clsp.jhu.edu ]]; then 
     # spread the egs over various machines.  will help reduce overload of any
     # one machine.
    utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/librispeech/s5/$dir/egs/storage $dir/egs/storage
  fi
  
  # The size of the system is kept rather smaller than the run_7a_960.sh system:
  # this is because we want it to be small enough that we could plausibly run it
  # in real-time.
  steps/nnet2/train_pnorm_fast.sh --stage $train_stage \
    --samples-per-iter 400000 \
    --num-epochs 6 --num-epochs-extra 2 \
    --splice-width 7 --feat-type raw \
    --online-ivector-dir exp/nnet2_online/ivectors_train_960 \
    --cmvn-opts "--norm-means=false --norm-vars=false" \
    --num-threads "$num_threads" \
    --minibatch-size "$minibatch_size" \
    --parallel-opts "$parallel_opts" \
    --io-opts "-tc 12" \
    --num-jobs-nnet 6 \
    --initial-learning-rate 0.01 --final-learning-rate 0.001 \
    --cmd "$decode_cmd" \
    --pnorm-input-dim 3500 \
    --pnorm-output-dim 350 \
    --num-hidden-layers 4 \
    --mix-up 12000 \
    data/train_960 data/lang exp/tri6b $dir  || exit 1;
fi

if [ $stage -le 5 ]; then
  # dump iVectors for the testing data.
  for test in dev_clean dev_other; do
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 20 \
      data/$test exp/nnet2_online/extractor exp/nnet2_online/ivectors_$test || exit 1;
  done
fi


if [ $stage -le 6 ]; then
  # this does offline decoding that should give about the same results as the
  # real online decoding (the one with --per-utt true)
  for test in dev_clean dev_other; do
    steps/nnet2/decode.sh --nj 30 --cmd "$decode_cmd" --config conf/decode.config \
      --online-ivector-dir exp/nnet2_online/ivectors_${test} \
      exp/tri6b/graph_tgsmall data/$test $dir/decode_${test}_tgsmall || exit 1;
    steps/lmrescore.sh --cmd "$decode_cmd" data/lang_test_{tgsmall,tgmed} \
      data/$test $dir/decode_${test}_{tgsmall,tgmed}  || exit 1;
  done
fi


if [ $stage -le 7 ]; then
  # If this setup used PLP features, we'd have to give the option --feature-type plp
  # to the script below.
  steps/online/nnet2/prepare_online_decoding.sh data/lang exp/nnet2_online/extractor \
    "$dir" ${dir}_online || exit 1;
fi

if [ $stage -le 8 ]; then
  # do the actual online decoding with iVectors, carrying info forward from 
  # previous utterances of the same speaker.
  for test in dev_clean dev_other; do
    steps/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 30 \
      exp/tri6b/graph_tgsmall data/$test ${dir}_online/decode_${test}_tgsmall || exit 1;
    steps/lmrescore.sh --cmd "$decode_cmd" data/lang_test_{tgsmall,tgmed} \
      data/$test ${dir}_online/decode_${test}_{tgsmall,tgmed}  || exit 1;
  done
fi

if [ $stage -le 9 ]; then
  # this version of the decoding treats each utterance separately
  # without carrying forward speaker information.
  for test in dev_clean dev_other; do
    steps/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 30 \
      --per-utt true exp/tri6b/graph_tgsmall data/$test ${dir}_online/decode_${test}_tgsmall_utt || exit 1;
    steps/lmrescore.sh --cmd "$decode_cmd" data/lang_test_{tgsmall,tgmed} \
      data/$test ${dir}_online/decode_${test}_{tgsmall,tgmed}_utt  || exit 1;
  done
fi

if [ $stage -le 10 ]; then
  # this version of the decoding treats each utterance separately
  # without carrying forward speaker information, but looks to the end
  # of the utterance while computing the iVector.
  for test in dev_clean dev_other; do
    steps/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 30 \
      --per-utt true --online false exp/tri6b/graph_tgsmall data/$test \
        ${dir}_online/decode_${test}_tgsmall_utt_offline || exit 1;
    steps/lmrescore.sh --cmd "$decode_cmd" data/lang_test_{tgsmall,tgmed} \
      data/$test ${dir}_online/decode_${test}_{tgsmall,tgmed}_utt_offline  || exit 1;
  done
fi

exit 0;
