#!/bin/bash

# Copyright 2015  Johns Hopkins University (Author: Jan Trmal).  Apache 2.0.

# Begin configuration section.
dev_audio=/export/corpora/LDC/LDC2012S05
train_audio=/export/corpora/LDC/LDC2012S05
train_transcripts=./transcriptions/
dev_transcripts=/export/a13/jtrmal/malach_uwb_sets/test/
audio_sampling_rate=16000
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


set -e
set -o pipefail
set -u

echo >&2 "Processing dev data: "
echo >&2 "  Transcripts from: $dev_transcripts"
echo >&2 "  Audio from      : $dev_audio"
mkdir -p data/local
#We are using two corpora dirs, as the original UWB dataset is (mostly) not part
#of the LDC Malach EN release
#It would be great if someone would come with a dev set that is fully contained
#in the LDC Malach EN release
mkdir -p data/local/dev
find $dev_transcripts -maxdepth 1 -type f  -name "*.trs"| sort -u > data/local/dev/files.lst
find $dev_audio -maxdepth 1 -type f  -name "*.mp2"| sort -u |\
  grep -F -f <(cat data/local/dev/files.lst| xargs -n1 -IX basename X .trs) \
  > data/local/dev/audio.lst

n_audio=`cat data/local/dev/audio.lst | wc -l`
n_text=`cat data/local/dev/files.lst | wc -l`

echo >&2 "  Number of recording files in data/local/dev/audio.lst : $n_audio"
echo >&2 "  Number of transcript files in data/local/dev/files.lst: $n_text"

echo >&2 "  Generating transcript... (this can take a while)..."
cat data/local/dev/files.lst |\
  ./local/malach_prepare_text_data.pl --crosstalks --max-text-len -1 --nounks \
  > data/local/dev/transcriptions.raw 2> data/local/dev/transcriptions.log

echo >&2
echo >&2 "Processing train data: "
echo >&2 "  Transcripts from: $train_transcripts"
echo >&2 "  Audio from      : $train_audio"
#For train set, we remove all files that are already in dev and dev
mkdir -p data/local/train 

find -L $train_transcripts -type f -name "*.trs" | \
  grep -v -F -f <(cat data/local/dev/files.lst | xargs -n1 basename) |\
  sort -u > data/local/train/files.lst

find -L $train_audio -type f -name "*.mp2"| sort -u | \
  grep -F -f <(cat data/local/train/files.lst| xargs -n1 -IX basename X .trs) \
  > data/local/train/audio.lst

n_audio=`cat data/local/train/audio.lst | wc -l`
n_text=`cat data/local/train/files.lst | wc -l`

echo >&2 "  Number of recording files in data/local/dev/audio.lst : $n_audio"
echo >&2 "  Number of transcript files in data/local/dev/files.lst: $n_text"

echo >&2 "  Generating transcript for training... (this can take a while)..."
cat data/local/train/files.lst | \
  ./local/malach_prepare_text_data.pl --max-text-len 110 \
  --nowarn-inline-speakers \
  > data/local/train/transcriptions.raw \
  2> data/local/train/transcriptions.log


echo >&2 "  Generating transcript for LM... (this can take a while)..."
cat data/local/train/files.lst | \
  ./local/malach_prepare_text_data.pl --crosstalks --max-text-len -1 \
  --nowarn-inline-speakers > data/local/train/transcriptions_for_lm.raw \
  2> data/local/train/transcriptions_for_lm.log

echo "All done..."

