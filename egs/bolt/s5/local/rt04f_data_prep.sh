#!/bin/bash
# Copyright 2013  Johns Hopkins University (author: Daniel Povey)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.


corpus_id=rt04f

[ -f ./path.sh ] &&  . ./path.sh

. ./utils/parse_options.sh

if [ $# != 5 ]; then
   echo "Usage: $0 /path/to/speech_audio /path/to/transcription /path/to/destination"
   exit 1;
fi

AUDIO_TRAIN_DIR=$1
TRANS_TRAIN_DIR=$2
AUDIO_DEV_DIR=$3
TRANS_DEV_DIR=$4
DESTINATION=$5

train_dir=$DESTINATION/train.${corpus_id}
devtest_dir=$DESTINATION/devtest.${corpus_id}

mkdir -p $train_dir
mkdir -p $devtest_dir

# Data directory check
if [ ! -d $AUDIO_DIR ] || [ ! -d $TRANS_DIR ]; then
  echo "Usage: $0 /path/to/speech_audio /path/to/transcription /path/to/destination"
  exit 1;
fi

# Find sph audio file for train dev resp.
find -L $AUDIO_TRAIN_DIR -iname "*.sph" > $train_dir/sph.flist
find -L $AUDIO_DEV_DIR -iname "*.sph"  > $devtest_dir/sph.flist

n=`cat $train_dir/sph.flist $devtest_dir/sph.flist | wc -l`
[ $n -ne 275 ] && \
  echo Warning: expected 275 data files, found $n


for dataset in train ; do
  eval dest_dir=\$${dataset}_dir
  echo -e "\n----Converting the $corpus_id $dataset dataset into kaldi directory in $dest_dir"
  find -L $TRANS_TRAIN_DIR -iname "*.txt"   |\
    perl local/callhome_data_convert.pl $dest_dir/sph.flist $dest_dir || exit 1

  cat $dest_dir/utt2spk | utils/utt2spk_to_spk2utt.pl > $dest_dir/spk2utt || exit 1
done
echo "All $corpus_id datasets converted..."
# We prepare the full kaldi directory (with the exception of the text file)
#d irectly in one pass through the data

for dataset in dev ; do
  eval dest_dir=\$${dataset}test_dir
  echo -e "\n----Converting the $corpus_id $dataset dataset into kaldi directory in $dest_dir"
  find -L $TRANS_DEV_DIR -iname "*.txt"  |\
    perl local/callhome_data_convert.pl $dest_dir/sph.flist $dest_dir || exit 1

  cat $dest_dir/utt2spk | utils/utt2spk_to_spk2utt.pl > $dest_dir/spk2utt || exit 1
done
echo "All $corpus_id datasets converted..."

# TODO: This should be moved to kaldi-tools directory
# Transcripts normalization and segmentation (this needs external tools).
# Download and configure word segmentation tool.
pyver=`python --version 2>&1 | sed -e 's:.*\([2-3]\.[0-9]\+\).*:\1:g'`
export PYTHONPATH=$PYTHONPATH:`pwd`/tools/mmseg-1.3.0/lib/python${pyver}/site-packages
if [ ! -d tools/mmseg-1.3.0/lib/python${pyver}/site-packages ]; then
  echo "--- Downloading mmseg-1.3.0 ..."
  echo "NOTE: it assumes that you have Python, Setuptools installed on your system!"
  wget -P tools http://pypi.python.org/packages/source/m/mmseg/mmseg-1.3.0.tar.gz
  tar xf tools/mmseg-1.3.0.tar.gz -C tools
  cd tools/mmseg-1.3.0
  mkdir -p lib/python${pyver}/site-packages
    python setup.py build
  python setup.py install --prefix=.
  cd ../..
  if [ ! -d tools/mmseg-1.3.0/lib/python${pyver}/site-packages ]; then
    echo "mmseg is not found - installation failed?"
    exit 1
  fi
fi
#  sed -e 's:<foreign language="\([a-z0-9][a-z0-9]*\)"> *\([^< ][^< ]*\) *\([^< ][^< ]*\) *</foreign>:<\1_\2_\3>:gi' |\
#  sed -e 's:<foreign language="\([a-z0-9][a-z0-9]*\)"> *\([^< ][^< ]*\) *\([^< ][^< ]*\) *\([^< ][^< ]*\) *</foreign>:<\1_\2_\3\_4>:gi' |\
#  sed -e 's:<foreign language="\([a-z0-9][a-z0-9]*\)"> *\([^< ][^< ] *\) *</foreign>:<\1_\2>:gi' |\
# TODO: The text filtering should be improved
echo -e "\n----Preparing audio training transcripts in $train_dir"
cat $train_dir/transcripts.txt |\
  sed -e 's/\([.,!?]\)/ \1 /g' |\
  sed -e 's/\[[a-zA-Z]*_noise/[noise/g' |\
  sed -e 's/{[a-zA-Z]*_noise/{noise/g' |\
  sed -e 's/((\([^)]\{1,\}\)))/\1/g' |\
  sed -e 's:<foreign language="[a-z][a-z]*"> *</foreign>:(()):gi' |\
  sed -e 's:<foreign language="[a-z][a-z]*"> *(()) *</foreign>:(()):gi' |\
  sed -e 's:<foreign language="English"> *\([^<][^<]*\) *</foreign>: \1 :gi' |\
  sed -e 's:<foreign language="\([a-z0-9][a-z0-9]*\)"> *\([^< ][^< ]*\) *</foreign>:<\1_\2>:gi' |\
  perl -pe 's:<noise>(.*?)</noise>:\1:i' |\
  tee $train_dir/transcripts_filtered.txt |\
  sed '/^\s*$/d' |\
  uconv -f utf-8 -t utf-8 -x "Any-Upper" |\
  local/callhome_normalize.pl |\
  python local/callhome_mmseg_segment.py |\
  awk '{if (NF > 1) print $0;}' | sort -u > $train_dir/text

local/prepare_stm.pl --fragmentMarkers "-" --hesitationToken "<HES>" --oovToken "<UNK>" $train_dir
utils/fix_data_dir.sh $train_dir

for dataset in $devtest_dir ; do
  echo -e "\n----Preparing audio (reference) transcripts in $dataset"
  cat $dataset/transcripts.txt |\
    sed -e 's/\([.,!?]\)/ \1 /g' |\
    sed -e 's/\[[a-zA-Z]*_noise/[noise/g' |\
    sed -e 's/{[a-zA-Z]*_noise/{noise/g' |\
    sed -e 's/((\([^)]\{1\}\)))/\1/g' |\
    sed -e 's:<foreign language="[a-z][a-z]*"> *</foreign>:(()):gi' |\
    sed -e 's:<foreign language="[a-z][a-z]*"> *(()) *</foreign>:(()):gi' |\
    sed -e 's:<foreign language="English"> *\([^<][^<]*\) *</foreign>: \1 :gi' |\
    sed -e 's:<foreign language="\([a-z0-9][a-z0-9]*\)"> *\([^< ][^< ]*\) *</foreign>:<\1_\2>:gi' |\
    uconv -f utf-8 -t utf-8 -x "Any-Upper" |\
    local/callhome_normalize.pl |\
    python local/callhome_mmseg_segment.py |\
    awk '{if (NF > 1) print $0;}' | sort -u > $dataset/text
  utils/fix_data_dir.sh $dataset

  local/prepare_stm.pl --fragmentMarkers "-" --hesitationToken "<HES>" --oovToken "<UNK>" $dataset
done

echo -e "\n\n$corpus_id data preparation succeeded..."
