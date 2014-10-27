#!/bin/bash

. ./path.sh

set -e 
set -o pipefail

corpora=./corpora/
target=./data/local/
if false; then
find -L $corpora -ipath "*LDC*" \( -iname "*sph" -o -name "*.flac" \) -iname "AR_*" > $corpora/audio.txt
find -L $corpora -ipath "*LDC*" \( -iname "*scr" -o -name "*.xml" \) -iname "AR_*">  $corpora/texts.txt

local/pair_data_files.pl $corpora/audio.txt $corpora/texts.txt  $corpora

grep -i -F -f conf/list.dev corpora/map.txt > $corpora/map.dev.txt
grep -i -F -f conf/list.tune corpora/map.txt > $corpora/map.tune.txt
grep -i -F -f conf/list.test corpora/map.txt > $corpora/map.test.txt
grep -i -v -F -f <(cat conf/list.dev conf/list.tune conf/list.test) $corpora/map.txt > $corpora/map.train.txt

mkdir -p $target
for dataset in dev tune train test ; do
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
  #uconv -f utf-8 -t utf-8 -x "Arabic-Latin" $datadir/transcripts.txt > $datadir/transcripts.transliterated.txt

  utils/combine_data.sh --extra-files "transcripts.txt" $target/$dataset $target/$dataset.scr $target/$dataset.xml 
  ./IBM/ibm_normalize.sh $target/$dataset/transcripts.txt $target/$dataset/text
done

#./IBM/ibm_normalize.sh data/local/train/transcripts.txt data/local/train/text
#./IBM/ibm_normalize.sh data/local/dev/transcripts.txt data/local/dev/text
#./IBM/ibm_normalize.sh data/local/tune/transcripts.txt data/local/tune/text

aux_corpora=""
#for dir in `find -L $corpora -maxdepth 1 -mindepth 1 \( -not -ipath "*LDC*" \)  -type d ` ; do
#  echo $dir
#  corpus_id=`basename $dir`
#  aux_corpora+=" $target/$corpus_id"
#  mkdir -p data/local/$corpus_id
#  local/parse_ibm_data.pl $dir/*db*.txt $dir/audio $dir/txt data/local/$corpus_id
#done

(
aux_corpora=""
set -x 
rm -rf data/train
utils/combine_data.sh data/train data/local/train $aux_corpora
cp -R data/local/dev data/dev
cp -R data/local/tune/ data/tune
cp -R data/local/test/ data/test
)

echo "Now lexicon preparation"
exit 0;

if $graphemic_lexicon ; then
  rm -rf data/local/lang
  mkdir -p data/local/dict
  rm -f data/local/dict/lexiconp.txt 
  cat <(echo -e "<UNK>\t<unk>") IBM/training/train.lexicon |\
    sed 's/(01)//g' |\
    sed 's/\[wb\]//g' |\
    sed 's/ _bw//g' | \
    sed 's/^\([^ \t][^\t ]*\) */\1\t/g' |\
    sed 's/\t  *\t* */\t/g' |\
    sed 's/  */ /g' |\
    sed 's/ *$//g ' |\
    sed 's/\t*$//g ' |\
    sed 's/\t\t*/\t/g' |\
    grep -v "</?s>" |\
    cat -> data/local/dict/lexicon.txt
    
    cat data/local/dict/lexicon.txt | awk '{print $1;}' > data/local/dict/words.txt

    cat data/local/dict/lexicon.txt | grep -v -F "%" | grep -v "SIL" | grep -v "UNK"  | cut -f 2- | sed 's/\s/\n/g' | sort -u > data/local/dict/nonsilence_phones.txt
    cat data/local/dict/lexicon.txt | grep -E "%|~SIL|</?s>|UNK" | cut  -f 2- | sed 's/ /\n/g' | sort -u > data/local/dict/silence_phones.txt
    #echo "~OOV" > data/local/dict/oov.txt
    echo "SIL" > data/local/dict/optional_silence.txt
else
  eca_lexicon=/export/corpora/LDC/LDC99L22
  lex_file=`find $eca_lexicon -iname "ar_lex.v07"`
  ./IBM/ibm_normalize_lexicon.sh $lex_file  >(cat - ) | local/callhome_prepare_dict_ibm.sh
fi
fi


cat IBM/scoring.Tune/tune.mbw.stm  | sed 's/[A-Z][A-Z]*_LDC2014E86_ar_/AR_/g' |sed 's/^\([A-Z][A-Z_0-9]*\)_[AB][0-9]* /\1 /g'  > data/tune/stm
cat IBM/scoring.Tune/dev.mbw.stm   | sed 's/[A-Z][A-Z]*_LDC2014E86_ar_/AR_/g' |sed 's/^\([A-Z][A-Z_0-9]*\)_[AB][0-9]* /\1 /g'   > data/dev/stm
cat IBM/scoring.Test//test.mbw.stm | sed 's/[A-Z][A-Z]*_LDC2014E86_ar_/AR_/g' |sed 's/^\([A-Z][A-Z_0-9]*\)_[AB][0-9]* /\1 /g'   > data/test/stm
cp ./IBM/scoring.Tune/ar2009.glm  data/tune/glm
cp -f data/tune/glm data/dev/glm
cp -f data/tune/glm data/test/glm

