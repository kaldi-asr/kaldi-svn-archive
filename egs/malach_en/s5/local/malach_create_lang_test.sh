#!/bin/bash

# Copyright 2015  Johns Hopkins University (Author: Jan Trmal).  Apache 2.0.

# Begin configuration section.
oov="<UNK>"
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

text=$1
tmp=$2
lang=$3

set -e
set -o pipefail
set -u

mkdir -p $tmp
cat $lang/words.txt | awk '{print $1}' > $tmp/words.txt

local/train_lms_srilm.sh --oov-symbol "$oov" \
  --words-file $tmp/words.txt --train-text $text \
  $tmp $tmp

local/arpa2G.sh $tmp/lm.gz $lang $lang

