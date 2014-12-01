#!/bin/bash
# Copyright 2014  Xiaohui Zhang, Jan Trmal
# Apache 2.0.

. ./path.sh
. ./cmd.sh
. ./conf.sh
stage=1
# resegment=false
# mkdir -p $corpora
#corpora=./corpora/
# audio files are in corpora and eval_audio
# corpora=corpora_eval/LDC2014E57
corpora=corpora/LDC2014E57/
eval_audio=/export/corpora5/LDC/LDC2014R65
# text files are in eval_sets
# eval_sets=./tmp_xml/
eval_sets=/export/a09/jtrmal/bolt/CTS/LDC2014R64/
target=./data/local/

. utils/parse_options.sh

set -e 
set -o pipefail
set -x

evaldir=data/local/eval
[ ! -x $corpora ] && echo "Corpora directory $corpora must exist!" && exit 1;

. utils/parse_options.sh
mkdir -p $evaldir
( 
  find -L $eval_audio  \( -iname "*sph" -o -iname "*.flac" \) -iname "MA_*"
  find -L $corpora -ipath "*LDC*" \( -iname "*sph" -o -iname "*.flac" \) -iname "MA_*") > $evaldir/audio.txt

if [ $stage -le 0 ]; then
  for evalset in `find $eval_sets/ -maxdepth 1 -mindepth 1 -type d ` ; do
    echo $evalset
    evlname=`basename ${evalset%%_without_transcriptions}`
    find -L $evalset  \( -iname "*scr" -o -name "*.xml" \) -iname "MA_*"\
      > $evaldir/${evlname}.list
  
    local/pair_data_files.pl $evaldir/audio.txt $evaldir/${evlname}.list  $evaldir
    mv $evaldir/map.txt $evaldir/${evlname}.map.txt 
    
    cat $evaldir/${evlname}.map.txt | cut -f 1 > $evaldir/${evlname}.audio.list
    cat $evaldir/${evlname}.map.txt | cut -f 2 > $evaldir/${evlname}.texts.list
  
    mkdir -p data/$evlname
    cat $evaldir/${evlname}.texts.list |\
    	perl /export/a09/jtrmal/bolt/callhome_azr15_no_gale2/local/bolt_data_convert.pl --empty-trans --numeric-channel-id $evaldir/$evlname.audio.list data/$evlname 
      	# perl local/bolt_data_convert.pl $evaldir/$evlname.audio.list data/$evlname
      # text and stm files only contain segmentation infomation. 
      utils/utt2spk_to_spk2utt.pl < data/$evlname/utt2spk >data/$evlname/spk2utt
      cp data/$evlname/transcripts.txt data/$evlname/text
      local/prepare_stm.pl --fragmentMarkers "-" --hesitationToken "<HES>" --oovToken "<UNK>" data/$evlname
      cat data/$evlname/stm | cut -f1-5 -d' ' | awk '{split($3, a, "-");split(a[2],b,""); c=b[1]; s=b[2];if(s=="")s="0";sub(/A/,"1",c);sub(/B/,"2",c); printf tolower($1)"_"s; printf " "c" "; printf ("%s_%d%d "$4" "$5" <O>\n", tolower(a[1]),s,c);}' > tmp1
      mv tmp1 data/$evlname/stm
      cp conf/glm data/$evlname
      utils/fix_data_dir.sh data/$evlname
  done
fi

if [ $stage -le 1 ]; then
  for dataset in PROGRESS VALIDATION; do
    echo "Extracting mfcc features for ${dataset} data"
    mfccdir=param
    steps/make_mfcc_pitch.sh --nj 64 --cmd "$train_cmd" data/$dataset exp/make_mfcc/${dataset}.original $mfccdir || exit 1;
    utils/fix_data_dir.sh data/$dataset
    steps/compute_cmvn_stats.sh data/${dataset} exp/make_mfcc/${dataset}.original $mfccdir || exit 1;
    utils/fix_data_dir.sh data/$dataset
    utils/validate_data_dir.sh --no-text  data/$dataset
  done
fi

#   # for segments longer than thr (in seconds), re-segment them using automatic segmentations.
#   if $resegment && [ $stage -le 2 ]; then
#     thr=10
#     echo "Doing resegmentation for ${dataset} data. Threshold is $thr"
#     cp -r data/${dataset} data/${dataset}.long
#     cat data/${dataset}/segments | awk -v thr=$thr '{if($4-$3 > thr)print $0}' > data/${dataset}.long/segments
#     utils/fix_data_dir.sh data/${dataset}.long
#     
#     mkdir -p data/${dataset}.short
#     cp data/${dataset}/{reco2file_and_channel,segments,spk2utt,utt2spk,wav.scp} data/${dataset}.short
#     cat data/${dataset}/segments | awk -v thr=$thr '{if($4-$3 <= thr)print $0}' > data/${dataset}.short/segments
#     utils/fix_data_dir.sh data/${dataset}.short
#     
#     # steps/make_phone_graph.sh data/lang exp/tri3a exp/tri3a
#     rm -rf exp/tri3a/decode_${dataset}.long tmp
#     ./local/resegment/generate_subsegments.sh --nj 1 data/${dataset}.long exp/tri3a tmp data/${dataset}.long.seg
#     mv data/${dataset} data/${dataset}.bk
#     utils/combine_data.sh data/${dataset} data/${dataset}.short data/${dataset}.long.seg
#   fi
# 
#   # re-compute mfcc for re-segmented audio files
#   if $resegment && [ $stage -le 3 ]; then
#     echo "Extracting mfcc features for fixed ${dataset} data"
#     mfccdir=param
#     steps/make_mfcc_pitch.sh --nj 32 --cmd "$train_cmd" data/$dataset exp/make_mfcc/${dataset}.original $mfccdir || exit 1;
#     steps/compute_cmvn_stats.sh data/${dataset} exp/make_mfcc/${dataset}.original $mfccdir || exit 1;
#   fi

