#!/bin/bash

. ./path.sh

corpora=./corpora/
target=./data/local/
eca_lexicon=

. utils/parse_options.sh

set -e 
set -o pipefail


[ ! -x $corpora ] && echo "Corpora directory $corpora must exist!" && exit 1;

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
  	perl local/bolt_data_convert.pl --numeric-channel-id $output/audio.list $output/

  output=$target/$dataset.scr
  mkdir -p $output
  echo $output
  (set +o pipefail; cat $corpora/map.${dataset}.txt | grep -v -F ".su.xml" | cut -f 1 )> $output/audio.list
  (set +o pipefail; cat $corpora/map.${dataset}.txt | grep -v -F ".su.xml" | cut -f 2 )> $output/texts.list
  cat $output/texts.list |\
  	perl local/callhome_data_convert.pl --numeric-channel-id  $output/audio.list $output/
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
set -x
if [ -z  "$eca_lexicon" ] ; then
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
    cat -> data/local/dict/lexicon-arz.txt

    cat data/train/text | sed 's/ /\n/g' | grep '_en' |  sort -u > data/local/dict/words-en-oov.txt
    local/generate_english_graphemes.sh ../callhome_azr15_no_gale_phonetic2/exp/g2p/en_to_arz_graphemes/ data/local/dict/words-en-oov.txt data/local/dict/lexicon-en.txt
    
    cat data/local/dict/lexicon-arz.txt data/local/dict/lexicon-en.txt > data/local/dict/lexicon.txt
    
    cat data/local/dict/lexicon.txt | awk '{print $1;}' > data/local/dict/words.txt

    cat data/local/dict/lexicon.txt | grep -v -F "%" | grep -v "SIL" | grep -v "UNK"  | cut -f 2- | sed 's/\s/\n/g' | sort -u > data/local/dict/nonsilence_phones.txt
    cat data/local/dict/lexicon.txt | grep -E "%|~SIL|</?s>|<UNK>" | cut  -f 2- | sed 's/ /\n/g' | sort -u > data/local/dict/silence_phones.txt
    #echo "~OOV" > data/local/dict/oov.txt
    echo "SIL" > data/local/dict/optional_silence.txt
else
  set -x
  lex_file=`find $eca_lexicon -iname "ar_lex.v07"`
  ./IBM/ibm_normalize_lexicon.sh $lex_file  >(cat - ) | pv | local/callhome_prepare_dict_ibm.sh
fi

cat IBM/scoring/tune.mbw.stm  |\
  local/convert_ibm_stm.sh | sort +0 -1 +1 -2 +3nb -4 > data/tune/stm
cat IBM/scoring/dev.mbw.stm |\
  local/convert_ibm_stm.sh | sort +0 -1 +1 -2 +3nb -4 > data/dev/stm
cat IBM/scoring/test.mbw.stm |\
  local/convert_ibm_stm.sh | sort +0 -1 +1 -2 +3nb -4 > data/test/stm

cat IBM/scoring/tune.stm  |\
  local/convert_ibm_stm.sh | sort +0 -1 +1 -2 +3nb -4 > data/tune/stm.utf8
cat IBM/scoring/dev.stm |\
  local/convert_ibm_stm.sh | sort +0 -1 +1 -2 +3nb -4 > data/dev/stm.utf8
cat IBM/scoring/test.stm |\
  local/convert_ibm_stm.sh | sort +0 -1 +1 -2 +3nb -4 > data/test/stm.utf8

cp IBM/scoring/ar2009.utf8.glm  data/tune/glm.utf8
cp -f data/tune/glm.utf8 data/dev/glm.utf8
cp -f data/tune/glm.utf8 data/test/glm.utf8

cp IBM/scoring/ar2009.glm  data/tune/glm
cp -f data/tune/glm data/dev/glm
cp -f data/tune/glm data/test/glm

