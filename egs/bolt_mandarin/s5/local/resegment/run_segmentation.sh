#!/bin/bash

# Copyright 2014  Vimal Manohar, Johns Hopkins University (Author: Jan Trmal)
# Apache 2.0

. path.sh
. cmd.sh
. ./conf.sh

# This is not necessarily the top-level run.sh as it is in other directories.   see README.txt first.

[ -f local.conf ] && . ./local.conf

. ./utils/parse_options.sh

set -o pipefail  #Exit if any of the commands in the pipeline will 
                 #return non-zero return code
set -u           #Fail on an undefined variable

#Later in the script we assume the run-1-main.sh was run (because we are using exp/tri4)
#So let's make it mandatory, instead of doing the work on our own.

[ ! -d data/train ] && echo "The source training data directory is not ready. Use the data preparation steps in run.sh" && exit 1

mkdir -p exp/make_seg/data/train_whole
cp data/train/wav.scp exp/make_seg/data/train_whole || exit 1

local/resegment/combine_segments.pl utt2spk data/train/segments \
  < data/train/utt2spk | sort -k 1,1 | head -n 100 > exp/make_seg/data/train_whole/utt2spk
local/resegment/combine_segments.pl text data/train/segments \
  < data/train/text | sort -k 1,1 > exp/make_seg/data/train_whole/text

utils/fix_data_dir.sh exp/make_seg/data/train_whole
 
mfccdir=param
x=train_whole
steps/make_mfcc_pitch.sh --nj $train_nj --cmd "$train_cmd" exp/make_seg/data/$x exp/make_mfcc/$x $mfccdir || exit 1;
steps/compute_cmvn_stats.sh exp/make_seg/data/$x exp/make_mfcc/$x $mfccdir || exit 1;

utils/fix_data_dir.sh exp/make_seg/data/$x

echo ---------------------------------------------------------------------
echo "Training segmentation model in exp/tri4b_whole"
echo ---------------------------------------------------------------------

local/resegment/train_segmentation.sh \
  --boost-sil 1.0 --nj $train_nj --cmd "$decode_cmd" \
  exp/tri3a exp/make_seg/data/$x data/lang exp/tri4b_$x || exit 1

echo ---------------------------------------------------------------------
echo "Finished successfully on" `date`
echo ---------------------------------------------------------------------

exit 0

