#!/bin/bash

. ./path.sh

set -e 
set -o pipefail


corpora=./corpora
mkdir -p $corpora
ln -s /export/a12/xzhang/kaldi-bolt/egs/bolt/20141103/corpora/LDC* $corpora

target=./data/local/
if true; then
find -L $corpora -ipath "*LDC*" \( -iname "*sph" -o -name "*.flac" \) -iname "MA_*" > $corpora/audio.txt
find -L $corpora -ipath "*LDC*" \( -iname "*scr" -o -name "*.xml" -o -iname "*.txt" \) -iname "MA_*">  $corpora/texts.txt

local/pair_data_files.pl $corpora/audio.txt $corpora/texts.txt  $corpora

grep -i -F -f conf/list.dev corpora/map.txt > $corpora/map.dev.txt
grep -i -F -f conf/list.tune corpora/map.txt > $corpora/map.tune.txt
grep -i -F -f conf/list.test corpora/map.txt > $corpora/map.test.txt
grep -i -v -F -f <(cat conf/list.dev conf/list.tune conf/list.test) $corpora/map.txt > $corpora/map.train.txt
fi
mkdir -p $target
for dataset in dev tune test train ; do
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
  ./local/bolt_mandarin_normalize.sh $target/$dataset/transcripts.txt $target/$dataset/text
  # local/prepare_stm.pl --fragmentMarkers "-" --hesitationToken "<HES>" --oovToken "<UNK>" $target/$dataset

  utils/fix_data_dir.sh $target/$dataset
done

(
set -x 
rm -rf data/bolt_dev
rm -rf data/bolt_tune
rm -rf data/bolt_test
cp -R data/local/dev data/bolt_dev
cp -R data/local/tune data/bolt_tune
cp -R data/local/test data/bolt_test
mv data/local/train data/local/train.new
)
