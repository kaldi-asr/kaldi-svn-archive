#!/bin/bash

. cmd.sh


stage=0
train_stage=-10
use_gpu=true
mfccdir=param

set -e
. cmd.sh
. ./path.sh
. ./utils/parse_options.sh


# assume use_gpu=true since it would be way too slow otherwise.

if ! cuda-compiled; then
  cat <<EOF && exit 1 
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA 
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi
parallel_opts="-l gpu=1" 
num_threads=1
minibatch_size=512
dir=exp/nnet2_online/nnet_a_gpu 
mkdir -p exp/nnet2_online

if [ $stage -le 0 ]; then
  # this shows how you can split across multiple file-systems.  we'll split the
  # MFCC dir across multiple locations.  You might want to be careful here, if you
  # have multiple copies of Kaldi checked out and run the same recipe, not to let
  # them overwrite each other.
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
    date=$(date +'%m_%d_%H_%M')
    utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/bolt-$date/s5/$mfccdir/storage $mfccdir/storage
  fi
  utils/copy_data_dir.sh data/train data/train_hires
  steps/make_mfcc_pitch_online.sh --nj 70 --mfcc-config conf/mfcc_hires.conf \
      --cmd "$train_cmd" data/train_hires exp/make_hires/train $mfccdir || exit 1;
  steps/compute_cmvn_stats.sh data/train_hires exp/make_hires/train $mfccdir || exit 1;
fi

if [ $stage -le 1 ]; then
  # we'll use the features with just MFCC, no pitch, to train the iVector
  # extractor on.  Check that we're using 40-dim features so the command line is correct.
  ! grep 'num-ceps=20' conf/mfcc_hires.conf >/dev/null && \
     echo "Change the script if you change conf/mfcc_hires.conf" && exit 1;
  steps/select_feats.sh  --nj 5 --cmd "$train_cmd" 0-19 data/train_hires \
      data/train_hires_mfcconly exp/nnet2_online/select_hires_train $mfccdir

  steps/compute_cmvn_stats.sh data/train_hires_mfcconly exp/nnet2_online/select_hires_train $mfccdir

  # Make a subset of about 1/3 of the data.
  utils/subset_data_dir.sh data/train_hires_mfcconly 100000 \
     data/train_hires_mfcconly_100k

  # make a corresponding subset of normal-dimensional-MFCC training data.
  utils/subset_data_dir.sh --utt-list <(awk '{print $1}' data/train_hires_mfcconly_100k/utt2spk) \
    data/train data/train_100k
fi

if [ $stage -le 2 ]; then
  # We need to build a small system just because we need the LDA+MLLT transform
  # to train the diag-UBM on top of.  First align the data of the 100k subset using
  # the tri5a system and normal MFCC features, so we have alignments to build our
  # system on hires MFCCs on top of.

  steps/align_fmllr.sh --nj 50 --cmd "$train_cmd" \
    data/train_100k data/lang exp/tri5a exp/tri5a_ali_100k || exit 1;

  # Build a small LDA+MLLT system on top of the hires MFCC features, just
  # because we need the transform.  We use --num-iters 13 because after we get
  # the transform (12th iter is the last), any further training is pointless.
  steps/train_lda_mllt.sh --cmd "$train_cmd" --num-iters 13 --realign-iters "" \
    --splice-opts "--left-context=3 --right-context=3" \
    5000 10000 data/train_hires_mfcconly_100k data/lang exp/tri5a_ali_100k exp/nnet2_online/tri6a
fi

if [ $stage -le 3 ]; then
  # Train a diagonal UBM.  The input directory exp/nnet2_online/tri6a is only
  # needed for the splice-opts and the LDA+MLLT transform.
  steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj 30 --num-frames 400000 \
    data/train_hires_mfcconly_100k 512 exp/nnet2_online/tri6a exp/nnet2_online/diag_ubm
fi

if [ $stage -le 4 ]; then
  # train an iVector extractor on all the mfcconly data.  Note: although we use
  # only 10 job, each job uses 16 processes in total.
  steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" --nj 10 \
    data/train_hires_mfcconly exp/nnet2_online/diag_ubm exp/nnet2_online/extractor || exit 1;
fi

if [ $stage -le 5 ]; then
  # extract iVectors for the training data.
  ivectordir=exp/nnet2_online/ivectors_train
  if [[ $(hostname -f) == *.clsp.jhu.edu ]]; then # this shows how you can split across multiple file-systems.
    utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/bolt/s5/$ivectordir/storage $ivectordir/storage
  fi

  # having a larger number of speakers is helpful for generalization, and to
  # handle per-utterance decoding well (iVector starts at zero).
  steps/online/nnet2/copy_data_dir.sh --utts-per-spk-max 2 data/train_hires_mfcconly data/train_hires_mfcconly_max2

  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 60 \
    data/train_hires_mfcconly_max2 exp/nnet2_online/extractor $ivectordir || exit 1;
fi


if [ $stage -le 6 ]; then
  if [ $USER == dpovey ]; then # this shows how you can split across multiple file-systems.
    utils/create_split_dir.pl /export/b0{1,2,3,4}/dpovey/kaldi-online/egs/bolt/s5/$dir/egs $dir/egs/storage
  fi

  # Because we have a lot of data here and we don't want the training to take
  # too long, we reduce the number of epochs from the defaults (15 + 5) to (3 +
  # 1).  The option "--io-opts '-tc 12'" is to have more than the default number
  # (5) of jobs dumping the egs to disk; this is OK since we're splitting our
  # data across four filesystems for speed.

  steps/nnet2/train_pnorm_simple.sh --stage $train_stage \
    --num-epochs 8 \
    --samples-per-iter 400000 \
    --splice-width 7 --feat-type raw \
    --online-ivector-dir exp/nnet2_online/ivectors_train \
    --cmvn-opts "--norm-means=false --norm-vars=false" \
    --num-threads "$num_threads" \
    --minibatch-size "$minibatch_size" \
    --parallel-opts "$parallel_opts" \
    --io-opts "-tc 12" \
    --num-jobs-nnet 6 \
    --num-hidden-layers 4 \
    --initial-learning-rate 0.01 --final-learning-rate 0.001 \
    --cmd "$decode_cmd" \
    --pnorm-input-dim 2500 \
    --pnorm-output-dim 250 \
     data/train_hires data/lang exp/tri5a $dir  || exit 1;
fi

if [ $stage -le 7 ]; then
  steps/online/nnet2/prepare_online_decoding.sh --mfcc-config conf/mfcc_hires.conf \
     --add-pitch true data/lang exp/nnet2_online/extractor "$dir" ${dir}_online || exit 1;
fi

if [ $stage -le 8 ]; then
  # do the actual online decoding with iVectors, carrying info forward from 
  # previous utterances of the same speaker.
   steps/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 30 \
      exp/tri5a/graph data/dev ${dir}_online/decode_dev || exit 1;
fi

if [ $stage -le 9 ]; then
  # this version of the decoding treats each utterance separately
  # without carrying forward speaker information.
   steps/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 30 \
     --per-utt true \
      exp/tri5a/graph data/dev ${dir}_online/decode_dev_utt || exit 1;
fi

if [ $stage -le 10 ]; then
  # this version of the decoding treats each utterance separately
  # without carrying forward speaker information, but looks to the end
  # of the utterance while computing the iVector.
   steps/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 30 \
     --per-utt true --online false \
      exp/tri5a/graph data/dev ${dir}_online/decode_dev_utt_offline || exit 1;
fi
 
if [ $stage -le 11 ]; then
  # this version of the decoding does pure offline decoding, with
  # mfcc + online_pitch + ivectors features
#  utils/copy_data_dir.sh data/dev data/dev_hires
#  steps/make_mfcc_pitch_online.sh --nj 70 --mfcc-config conf/mfcc_hires.conf \
#      --cmd "$train_cmd" data/dev_hires exp/make_hires/dev $mfccdir || exit 1;
#  steps/compute_cmvn_stats.sh data/dev_hires exp/make_hires/dev $mfccdir || exit 1;
#  
#  steps/select_feats.sh  --nj 5 --cmd "$train_cmd" 0-19 data/dev_hires \
#      data/dev_hires_mfcconly exp/nnet2_online/select_hires_dev $mfccdir
#
#  steps/compute_cmvn_stats.sh data/dev_hires_mfcconly exp/nnet2_online/select_hires_dev $mfccdir
#  
#  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 30 \
#    data/dev_hires_mfcconly exp/nnet2_online/extractor exp/nnet2_online/ivectors_dev || exit 1;
#
  steps/nnet2/decode.sh --nj 30 --cmd "$decode_cmd" \
          --online-ivector-dir exp/nnet2_online/ivectors_dev \
      exp/tri5a/graph data/dev_hires ${dir}/decode_dev || exit 1;
fi

exit 0;

