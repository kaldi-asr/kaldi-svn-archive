#!/usr/bin/env bash

# Creates splits of a specific train directory (default 10)
# Trains an LM which holds back data from that split
# Used for cross fold training and decoding

if [ $# -ne 2 ]
then
  echo "Usage : local/create_splits.sh <data-dir> <train-dir-name>"
  echo "Eg. local/create_splits.sh data train"
  exit 1;
fi

dataDir=$1
trainDir=$dataDir/$2

if [ ! -f $trainDir/segments ]
then
  echo "Invalid training dir specified"
  exit 1;
fi

noSplits=10
utils/split_data.sh data/train $noSplits

lang=data/lang_test/split$noSplits
lm=data/lm/split$noSplits
mkdir -p $lang
for elem in data/train/split$noSplits/* ; do
  echo $elem
  mkdir -p $lang/`basename $elem`
  mkdir -p $lm/`basename $elem` 

  seq=`basename $elem`
  otherdirs=`ls -1 data/train/split$noSplits/ | grep -v  "^${seq}\$" | paste -s -d','`
  othertext=data/train/split$noSplits/{$otherdirs}/text
  eval cat $othertext | sort -u > $elem/text.other
  local/split_train_lms.sh --lexicon data/local/dict/lexicon.txt $elem $lm/`basename $elem`
done

for elem in $lm/* ; do
  echo $elem
  mkdir -p $lang/`basename $elem`
  local/arpa2G.sh  $elem/3gram-mincount/lm_unpruned.gz  data/lang $lang/`basename $elem`
done
