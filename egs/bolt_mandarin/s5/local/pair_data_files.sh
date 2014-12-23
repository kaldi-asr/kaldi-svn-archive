#!/bin/bash

# Copyright 2014 Xiaohui Zhang, Jan Trmal
# Apache 2.0

. ./path.sh

set -e 
set -o pipefail
set -x

corpora=./corpora
rm -rf $corpora
mkdir -p $corpora

# In this script, we put soft links of training data (Part II) and Dev data (bolt_dev/tune/test) under corpora/ and get the data prepared.
# Training data (Part I) are specified in conf.sh and prepared via local/{callhome,hkust,hub5,rt04f}_data_prep.sh
# The reason why there are two subsets of training data is that, The correspondance between Audio/Text files is very clear 
# for Part I, while for Part II it's quite complicated (Only selected conversations in each corpus are counted in, 
# which are specifed in conf/list.train.part2). 

for dir in `cat conf/list.train.part2 | cut -f1 -d' '`; do
mkdir -p $corpora/$dir
for i in $(find -L /export/corpora/LDC/$dir -iname "ma_*" | grep "`cat conf/list.train.part2 | grep "$dir " | cut -f2 -d' '`");do ln -s $i corpora/$dir; done;
done

target=./data/local/

if true; then
find -L $corpora -ipath "*LDC*" \( -iname "*sph" -o -name "*.flac" \) -iname "MA_*" > $corpora/audio.txt
find -L $corpora -ipath "*LDC*" \( -iname "*scr" -o -name "*.xml" -o -iname "*.txt" \) -iname "MA_*">  $corpora/texts.txt

local/pair_data_files.pl $corpora/audio.txt $corpora/texts.txt  $corpora

grep -i -F -f conf/list.dev corpora/map.txt > $corpora/map.dev.txt
grep -i -F -f conf/list.tune corpora/map.txt > $corpora/map.tune.txt
grep -i -F -f conf/list.test corpora/map.txt > $corpora/map.test.txt
grep -i -v -F -f <(cat conf/list.dev conf/list.tune conf/list.test) $corpora/map.txt > $corpora/map.train.part2.txt
fi

mkdir -p $target
for dataset in dev tune test train.part2 ; do
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
  	perl local/callhome_data_convert.pl $output/audio.list $output/
  #uconv -f utf-8 -t utf-8 -x "Arabic-Latin" $datadir/transcripts.txt > $datadir/transcripts.transliterated.txt

  utils/combine_data.sh --extra-files "transcripts.txt" $target/$dataset $target/$dataset.scr $target/$dataset.xml 
  ./local/bolt_mandarin_normalize.sh $target/$dataset/transcripts.txt $target/$dataset/text

  utils/fix_data_dir.sh $target/$dataset
done

for dataset in dev tune test; do
  cp conf/glm data/local/${dataset}
  # using Cambridge provided stms
  cp conf/dev-${dataset}.stm data/local/${dataset}/stm
  # You can also try to prepare stms yourself, rather than using the stm provided by Cambridge. However different scoring results are expected. 
  # local/prepare_stm.pl --fragmentMarkers "-" --hesitationToken "<HES>" --oovToken "<UNK>" $target/$dataset
  # paste -d' ' <(cat $target/$dataset/stm | cut -f1-5 -d' ' |\
  # awk '{split($3, a, "-");split(a[2],b,""); c=b[1]; s=b[2];if(s=="")s="0";sub(/A/,"1",c);sub(/B/,"2",c); printf tolower($1)"_"s; printf " "c" "; printf ("%s_%d%d "$4" "$5" <O>\n", tolower(a[1]),s,c);}') <(cat $target/$dataset/stm | cut -f6- -d' ')  |\
  # sed 1'i;; CATEGORY "0" "" ""' | sed 1'i;; LABEL "O" "Overall" "All Segments in the test set"' > tmp
  # mv tmp $target/$dataset/stm
done

(
set -x 
cp -R data/local/dev/* data/bolt_dev
cp -R data/local/tune/* data/bolt_tune
cp -R data/local/test/* data/bolt_test
)
