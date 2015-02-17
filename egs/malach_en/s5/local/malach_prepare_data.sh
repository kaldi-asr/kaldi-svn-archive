#!/bin/bash

# Copyright 2015  Johns Hopkins University (Author: Jan Trmal).  Apache 2.0.

# Begin configuration section.
train_corpus_audio=/export/corpora/LDC/LDC2012S05
train_corpus_text=./transcriptions/
dev_corpus=/export/a13/jtrmal/Malach_EN_test
audio_sampling_rate=16000
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


set -e
set -o pipefail
set -u


if false; then
mkdir -p data/local
#We are using two corpora dirs, as the original UWB dataset is (mostly) not part
#of the LDC Malach EN release
#It would be great if someone would come with a dev set that is fully contained
#in the LDC Malach EN release
mkdir -p data/local/dev_uwb
find $dev_corpus -maxdepth 1 -type f  -name "*.trs"| sort -u > data/local/dev_uwb/files.lst
find $dev_corpus -maxdepth 1 -type f  -name "*.wav"| sort -u > data/local/dev_uwb/audio.lst
cat data/local/dev_uwb/files.lst |\
  ./local/malach_prepare_text_data.pl --crosstalks --max-text-len -1 --nounks > data/local/dev_uwb/transcriptions.raw

paste -d ' '\
  <(cut -f 1 -d ' ' data/local/dev_uwb/transcriptions.raw)  \
  <(cut -f 2- -d ' '  data/local/dev_uwb/transcriptions.raw | \
    sed  -f conf/unify.sed | sed  -f conf/uhs.sed | sed -f conf/doubles.sed ) \
  > data/local/dev_uwb/transcriptions.txt

local/malach_create_kaldi_files.pl data/local/dev_uwb/audio.lst \
  data/local/dev_uwb/transcriptions.txt data/local/dev_uwb

#Let's assume the LDC devset is defined by conf/dev.list
mkdir -p data/local/dev
find -L $train_corpus_text -type f -name "*.trs"| sort -u | \
  grep -F -f <(cat conf/dev.list | xargs -n1 basename) \
  > data/local/dev/files.lst
find -L $train_corpus_audio -type f -name "*.mp2"| sort -u | \
  grep -F -f <(cat conf/dev.list | xargs -n1 basename) \
  > data/local/dev/audio.lst
cat data/local/dev/files.lst |\
  ./local/malach_prepare_text_data.pl --crosstalks --max-text-len -1 --nounks > data/local/dev/transcriptions.raw

paste -d ' ' \
  <(cut -f 1 -d ' ' data/local/dev/transcriptions.raw)  \
  <(cut -f 2- -d ' '  data/local/dev/transcriptions.raw | \
    sed  -f conf/unify.sed | sed  -f conf/uhs.sed | sed -f conf/doubles.sed ) \
  > data/local/dev/transcriptions.txt

local/malach_create_kaldi_files.pl data/local/dev/audio.lst \
  data/local/dev/transcriptions.txt data/local/dev

#For train set, we remove all files that are already in dev and dev_uwb
mkdir -p data/local/train
find -L $train_corpus_text -type f -name "*.trs"| sort -u | \
  grep -v -F -f <(cat data/local/dev_uwb/files.lst | xargs -n1 basename) |\
  grep -v -F -f <(cat data/local/dev/files.lst | xargs -n1 basename) \
  > data/local/train/files.lst
find -L $train_corpus_audio -type f -name "*.mp2"| sort -u | \
  grep -F -f <(cat data/local/train/files.lst| xargs -n1 -IX basename X .trs) \
  > data/local/train/audio.lst

cat data/local/train/files.lst | \
  ./local/malach_prepare_text_data.pl --max-text-len 110 --nowarn-inline-speakers > data/local/train/transcriptions.raw

fi
paste -d ' ' \
  <(cut -f 1 -d ' ' data/local/train/transcriptions.raw)  \
  <(cut -f 2- -d ' '  data/local/train/transcriptions.raw | \
    sed  -f conf/unify.sed | \
    sed  -f conf/uhs.sed | \
    sed -f conf/doubles.sed |\
    sed -f conf/kill.sed  ) \
  > data/local/train/transcriptions.txt

cat data/local/train/files.lst | \
  ./local/malach_prepare_text_data.pl --crosstalks --max-text-len -1 --nowarn-inline-speakers > data/local/train/transcriptions_for_lm.raw

paste -d ' '  \
  <(cut -f 1 -d ' ' data/local/train/transcriptions_for_lm.raw)  \
  <(cut -f 2- -d ' '  data/local/train/transcriptions_for_lm.raw | \
    sed  -f conf/unify.sed | sed  -f conf/uhs.sed | sed -f conf/doubles.sed ) \
  > data/local/train/transcriptions_for_lm.txt


