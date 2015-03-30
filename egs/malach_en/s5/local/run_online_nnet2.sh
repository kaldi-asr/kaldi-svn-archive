#!/bin/bash

. ./cmd.sh
. ./path.sh

train_stage=-999

set  -e -o pipefail

mfcc=mfcc_hires
for data_set in train dev ; do
  [ -f data/${data_set}_hires/.done ] && continue
  # Create MFCCs for the eval set
  utils/copy_data_dir.sh data/${data_set} data/${data_set}_hires
  steps/make_mfcc.sh --cmd "$train_cmd" --nj 32 --mfcc-config conf/${mfcc}.conf \
    data/${data_set}_hires exp/make_${mfcc}/${data_set} param/
  steps/compute_cmvn_stats.sh data/${data_set}_hires exp/make_${mfcc}/train param/
  utils/fix_data_dir.sh data/${data_set}_hires  # remove segments with problems
  touch data/${data_set}_hires/.done
done

if [ ! -f exp/nnet2_online/tri4/.done ]; then
  steps/train_lda_mllt.sh --cmd "$decode_cmd" --num-iters 13 --realign-iters ""\
    --splice-opts " --left-context=3 --right-context=3" 5000 10000 \
    data/train_hires data/lang   exp/tri3_ali exp/nnet2_online/tri4
 touch exp/nnet2_online/tri4/.done
fi

if [ ! -f exp/nnet2_online/diag_ubm/.done ] ; then
  steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj 32 \
    --num-frames 700000 data/train_hires 512 exp/nnet2_online/tri4 \
    exp/nnet2_online/diag_ubm
  touch exp/nnet2_online/diag_ubm/.done 
fi

if [ ! -f exp/nnet2_online/extractor/.done ] ; then
  steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" --nj 10 \
    data/train_hires exp/nnet2_online/diag_ubm exp/nnet2_online/extractor
  touch exp/nnet2_online/extractor/.done 
fi

if [ ! -f data/train_hires_max2/.done ] ; then
  steps/online/nnet2/copy_data_dir.sh --utts-per-spk-max 2 \
    data/train_hires data/train_hires_max2
  touch data/train_hires_max2/.done
fi

if [ ! -f exp/nnet2_online/ivectors_train_hires/.done ] ; then
  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 32 \
    data/train_hires_max2 exp/nnet2_online/extractor \
    exp/nnet2_online/ivectors_train_hires
  touch exp/nnet2_online/ivectors_train_hires/.done 
fi


