#!/bin/bash

# Copyright 2015  Johns Hopkins University (Author: Jan Trmal).  Apache 2.0.

# Begin configuration section.
oov="<UNK>"
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

dict=$1
tmp=$2
lang=$3

set -e
set -o pipefail
set -u

rm -rf $lang 2>/dev/null || true 
rm -rf $tmp 2>/dev/null || true 

utils/prepare_lang.sh --share-silence-phones true  $dict "$oov" \
  $tmp $lang


