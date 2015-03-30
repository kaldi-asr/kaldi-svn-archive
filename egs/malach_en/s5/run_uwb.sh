#!/bin/bash
#/export/corpora/LDC/LDC2012S05/data/
#/export/corpora/LDC/LDC2012S05/data/
train_audio_path=/export/corpora/LDC/LDC2012S05/data
train_trs_path=/export/a13/jtrmal/malach_uwb_sets/train/
dev_audio_path=/export/corpora/LDC/LDC2012S05/data/mp2/
dev_audio_path=/export/a13/jtrmal/malach_uwb_sets/test/
dev_trs_path=/export/a13/jtrmal/malach_uwb_sets/test/


. ./path.sh
. ./cmd.sh

set -e -o pipefail

if [[ `hostname -f` == *clsp.jhu.edu ]] ; then
  w=`whoami`
  segpath=`pwd | sed 's/.*'$who'//g'`
  utils/create_split_dir.pl  /export/b0{1,2,3,4,5,6,7}/$w/${segpath}/storage \
    param/storage 2>/dev/null
fi
if [ ! -f data/.done ] ; then
  local/malach_prepare_data.sh \
    --train-audio $train_audio_path --dev-audio $dev_audio_path \
    --train-transcripts $train_trs_path --dev-transcripts $dev_trs_path


  local/malach_filter_transcripts_uwb.sh data/local/dev
  local/malach_filter_transcripts_uwb.sh data/local/train

  local/malach_create_lexicon.sh --nonspeech conf/nonspeech-uwb.txt \
    --sil "<SILENCE>" --oov "<UNK>" \
    data/local/train/transcriptions_for_lm.txt data/local/dict

  #local/malach_create_test_lexicon.sh --nonspeech conf/nonspeech-uwb.txt \
  #  --sil "<SILENCE>" --oov "<UNK>" \
  #  data/local/train/transcriptions_for_lm.txt data/local/dict

  local/malach_create_lang.sh --oov "<UNK>" \
    data/local/dict data/local/lang data/lang

  #local/malach_create_lang.sh --oov "<UNK>" \
  #  data/local/dict_test data/local/lang_test data/lang_test



  cp -r data/lang data/lang_test
  local/malach_create_lang_test.sh --oov "<UNK>" data/local/train/transcriptions_for_lm.txt \
    data/local/srilm data/lang_test


  local/malach_create_kaldi_files.sh --map-oov false \
    data/local/dev  data/lang  data/dev
  local/malach_create_kaldi_files.sh --map-oov true  \
    data/local/train data/lang data/train

  touch data/.done
fi

for dataset in train dev ; do
  if [ ! -f data/$dataset/.done ] ; then
    #steps/make_plp_pitch.sh --nj 32 --cmd "$train_cmd" data/$dataset exp/make_param/$dataset param
    steps/make_mfcc_pitch.sh --nj 32 --cmd "$train_cmd" data/$dataset exp/make_param/$dataset param
    utils/fix_data_dir.sh data/$dataset/
    steps/compute_cmvn_stats.sh data/$dataset exp/make_param/$dataset param/
    utils/fix_data_dir.sh data/$dataset
    utils/validate_data_dir.sh data/$dataset
    touch data/$dataset/.done
  fi
done

if [ ! -f data/train_sub1/.done ] ; then
  utils/subset_data_dir.sh  data/train 10000 data/train_sub1
  touch data/train_sub1/.done 
fi

if [ ! -f data/train_sub2/.done ] ; then
  utils/subset_data_dir.sh  data/train 150000 data/train_sub2
  touch data/train_sub2/.done 
fi

# Acoustic model parameters
numLeavesTri1=1000
numGaussTri1=10000

numLeavesTri2=1000
numGaussTri2=20000

numLeavesTri3=6000
numGaussTri3=75000

numLeavesMLLT=8000
numGaussMLLT=100000

numLeavesSAT=10000
numGaussSAT=150000

numGaussUBM=800
numLeavesSGMM=10000
numGaussSGMM=80000


if [ ! -f exp/mono/.done ]; then
  steps/train_mono.sh --nj 16 --cmd "$train_cmd" data/train_sub1 data/lang exp/mono
  touch exp/mono/.done
fi

(
  [ -f exp/mono/decode_dev/.done ] && exit 0

  utils/mkgraph.sh --mono data/lang_test exp/mono/ exp/mono/graph

  steps/decode.sh  --cmd "$decode_cmd" \
    --parallel-opts "-pe smp 4" --num-threads 4 --nj 32 \
    exp/mono/graph/ data/dev/ exp/mono/decode_dev

  touch exp/mono/decode_dev/.done
) &

if [ ! -f exp/tri1/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Starting (small) triphone training in exp/tri1 on" `date`
  echo ---------------------------------------------------------------------
  steps/align_si.sh \
    --nj 32  --cmd "$train_cmd" \
    data/train_sub2 data/lang exp/mono exp/mono_ali
  steps/train_deltas.sh \
    --cmd "$train_cmd" $numLeavesTri1 $numGaussTri1 \
    data/train_sub2 data/lang exp/mono_ali exp/tri1
  touch exp/tri1/.done
fi


(
  [ -f exp/tri1/decode_dev/.done ] && exit 0

  utils/mkgraph.sh data/lang_test exp/tri1/ exp/tri1/graph

  steps/decode_si.sh  --cmd "$decode_cmd" \
    --parallel-opts "-pe smp 4" --num-threads 4 --nj 32 \
    exp/tri1/graph/ data/dev/ exp/tri1/decode_dev

  touch exp/tri1/decode_dev/.done
) &

echo ---------------------------------------------------------------------
echo "Starting (medium) triphone training in exp/tri2 on" `date`
echo ---------------------------------------------------------------------
if [ ! -f exp/tri2/.done ]; then
  steps/align_si.sh \
    --nj 32 --cmd "$train_cmd" \
    data/train data/lang exp/tri1 exp/tri1_ali
  steps/train_deltas.sh \
    --cmd "$train_cmd" $numLeavesTri2 $numGaussTri2 \
    data/train data/lang exp/tri1_ali exp/tri2
  touch exp/tri2/.done
fi

(
  [ -f exp/tri2/decode_dev/.done ] && exit 0

  utils/mkgraph.sh data/lang_test exp/tri2/ exp/tri2/graph

  steps/decode_si.sh  --cmd "$decode_cmd" \
    --parallel-opts "-pe smp 4" --num-threads 4 --nj 32 \
    exp/tri2/graph/ data/dev/ exp/tri2/decode_dev

  touch exp/tri2/decode_dev/.done
) &


echo ---------------------------------------------------------------------
echo "Starting (full) triphone training in exp/tri3 on" `date`
echo ---------------------------------------------------------------------
if [ ! -f exp/tri3/.done ]; then
  steps/align_si.sh \
    --nj 32 --cmd "$train_cmd" \
    data/train data/lang exp/tri2 exp/tri2_ali
  steps/train_deltas.sh \
    --cmd "$train_cmd" $numLeavesTri3 $numGaussTri3 \
    data/train data/lang exp/tri2_ali exp/tri3
  touch exp/tri3/.done
fi


(
  [ -f exp/tri3/decode_dev/.done ] && exit 0

  utils/mkgraph.sh data/lang_test exp/tri3/ exp/tri3/graph

  steps/decode_si.sh  --cmd "$decode_cmd" \
    --parallel-opts "-pe smp 4" --num-threads 4 --nj 32 \
    exp/tri3/graph/ data/dev/ exp/tri3/decode_dev

  touch exp/tri3/decode_dev/.done
) &



echo ---------------------------------------------------------------------
echo "Starting (lda_mllt) triphone training in exp/tri4 on" `date`
echo ---------------------------------------------------------------------
if [ ! -f exp/tri4/.done ]; then
  steps/align_si.sh \
    --nj 32 --cmd "$train_cmd" \
    data/train data/lang exp/tri3 exp/tri3_ali

  steps/train_lda_mllt.sh \
    --cmd "$train_cmd" \
    $numLeavesMLLT $numGaussMLLT data/train data/lang exp/tri3_ali exp/tri4
  touch exp/tri4/.done
fi

(
  [ -f exp/tri4/decode_dev/.done ] && exit 0

  utils/mkgraph.sh data/lang_test exp/tri4/ exp/tri4/graph

  steps/decode.sh  --cmd "$decode_cmd" \
    --parallel-opts "-pe smp 4" --num-threads 4 --nj 32 \
    exp/tri4/graph/ data/dev/ exp/tri4/decode_dev

  touch exp/tri4/decode_dev/.done
) &

echo ---------------------------------------------------------------------
echo "Starting (SAT) triphone training in exp/tri5 on" `date`
echo ---------------------------------------------------------------------

if [ ! -f exp/tri5/.done ]; then
  steps/align_si.sh \
    --nj 32 --cmd "$train_cmd" \
    data/train data/lang exp/tri4 exp/tri4_ali

  steps/train_sat.sh \
    --cmd "$train_cmd" \
    $numLeavesSAT $numGaussSAT data/train data/lang exp/tri4_ali exp/tri5
  touch exp/tri5/.done
fi

utils/mkgraph.sh data/lang_test  exp/tri5/ exp/tri5/graph
  (
    [ -f exp/tri5/decode_dev_fmllr_extra_A/.done ] && exit 0
    
    steps/decode_fmllr_extra.sh  --cmd "$decode_cmd" \
      --parallel-opts "-pe smp 4" --num-threads 4 --nj 32 \
      exp/tri5/graph/ data/dev_A/ exp/tri5/decode_dev_fmllr_extra_A


    touch exp/tri5/decode_dev_fmllr_extra_A/.done
  ) &
  (
  [ -f exp/tri5/decode_dev_fmllr_extra_B/.done ] && exit 0
    
    steps/decode_fmllr_extra.sh  --cmd "$decode_cmd" \
      --parallel-opts "-pe smp 4" --num-threads 4 --nj 32 \
      exp/tri5/graph/ data/dev_B/ exp/tri5/decode_dev_fmllr_extra_B


    touch exp/tri5/decode_dev_fmllr_extra_B/.done
  ) &

wait

