#!/bin/bash
# Copyright 2014  Xiaohui Zhang, Jan Trmal
# Apache 2.0.

. ./path.sh
. ./cmd.sh
. ./conf.sh

set -e 
set -o pipefail
set -x
stage=0
corpora=./corpora_eval
resegment=true
# mkdir -p $corpora
target=./data/local/

. utils/parse_options.sh
if true; then
  find -L $corpora -ipath "*LDC*" \( -iname "*sph" -o -name "*.flac" \) -iname "MA_*" > $corpora/audio.txt
  # Test here, needs modification!
  find -L $corpora -ipath "*LDC*" \( -iname "*scr" -o -name "*.xml" -o -iname "*.txt" \) -iname "MA_*">  $corpora/texts.txt
  
  local/pair_data_files.pl $corpora/audio.txt $corpora/texts.txt  $corpora
  # mv $corpora/map.txt $corpora/map.eval.txt
  # Test here, needs modification!
  grep -i -F -f conf/list.dev $corpora/map.txt > $corpora/map.eval.txt
fi

mkdir -p $target
for dataset in eval; do
  if [ $stage -le 0 ]; then
    output=$target/$dataset.xml
    mkdir -p $output
    echo $output
    cat $corpora/map.${dataset}.txt | grep -F ".su.xml" | cut -f 1 > $output/audio.list
    cat $corpora/map.${dataset}.txt | grep -F ".su.xml" | cut -f 2 > $output/texts.list
    cat $output/texts.list |\
    	perl local/bolt_data_convert.pl $output/audio.list $output/
    output=$target/$dataset.scr
    mkdir -p $output
    echo $output
    (set +o pipefail; cat $corpora/map.${dataset}.txt | grep -v -F ".su.xml" | cut -f 1 )> $output/audio.list
    (set +o pipefail; cat $corpora/map.${dataset}.txt | grep -v -F ".su.xml" | cut -f 2 )> $output/texts.list
  
    cat $output/texts.list |\
    	perl local/callhome_data_convert.pl $output/audio.list $output/
  
    utils/combine_data.sh --extra-files "transcripts.txt" $target/$dataset $target/$dataset.scr $target/$dataset.xml 
    # text and stm files only contain segmentation infomation. 
    cp $target/$dataset/transcripts.txt $target/$dataset/text
    local/prepare_stm.pl --fragmentMarkers "-" --hesitationToken "<HES>" --oovToken "<UNK>" $target/$dataset
  
    cat $target/$dataset/stm | cut -f1-5 -d' ' | awk '{split($3, a, "-");split(a[2],b,""); c=b[1]; s=b[2];if(s=="")s="0";sub(/A/,"1",c);sub(/B/,"2",c); printf tolower($1)"_"s; printf " "c" "; printf ("%s_%d%d "$4" "$5" <O>\n", tolower(a[1]),s,c);}' > tmp1
    mv tmp1 $target/$dataset/stm

    cp conf/glm data/local/${dataset}
    cp -R data/local/eval data/eval
  fi

  if [ $stage -le 1 ]; then
    echo "Extracting mfcc features for ${dataset} data"
    mfccdir=param
    steps/make_mfcc_pitch.sh --nj 32 --cmd "$train_cmd" data/$dataset exp/make_mfcc/${dataset}.original $mfccdir || exit 1;
    steps/compute_cmvn_stats.sh data/${dataset} exp/make_mfcc/${dataset}.original $mfccdir || exit 1;
  fi

  # for segments longer than thr (in seconds), re-segment them using automatic segmentations.
  if $resegment && [ $stage -le 2 ]; then
    thr=10
    echo "Doing resegmentation for ${dataset} data. Threshold is $thr"
    cp -r data/${dataset} data/${dataset}.long
    cat data/${dataset}/segments | awk -v thr=$thr '{if($4-$3 > thr)print $0}' > data/${dataset}.long/segments
    utils/fix_data_dir.sh data/${dataset}.long
    
    mkdir -p data/${dataset}.short
    cp data/${dataset}/{reco2file_and_channel,segments,spk2utt,utt2spk,wav.scp} data/${dataset}.short
    cat data/${dataset}/segments | awk -v thr=$thr '{if($4-$3 <= thr)print $0}' > data/${dataset}.short/segments
    utils/fix_data_dir.sh data/${dataset}.short
    
    # steps/make_phone_graph.sh data/lang exp/tri3a exp/tri3a
    rm -rf exp/tri3a/decode_${dataset}.long tmp
    ./local/resegment/generate_subsegments.sh --nj 1 data/${dataset}.long exp/tri3a tmp data/${dataset}.long.seg
    mv data/${dataset} data/${dataset}.bk
    utils/combine_data.sh data/${dataset} data/${dataset}.short data/${dataset}.long.seg
  fi

  # re-compute mfcc for re-segmented audio files
  if $resegment && [ $stage -le 3 ]; then
    echo "Extracting mfcc features for fixed ${dataset} data"
    mfccdir=param
    steps/make_mfcc_pitch.sh --nj 32 --cmd "$train_cmd" data/$dataset exp/make_mfcc/${dataset}.original $mfccdir || exit 1;
    steps/compute_cmvn_stats.sh data/${dataset} exp/make_mfcc/${dataset}.original $mfccdir || exit 1;
  fi
done

# after this, the next command will remove the small number of utterances
