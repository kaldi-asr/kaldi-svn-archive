#!/bin/bash 

# Copyright 2015  Johns Hopkins University (Author: Jan Trmal).  Apache 2.0.
# Begin configuration section.
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

set -e
set -o pipefail
set -u

dir=$1

if [ -f $dir/transcriptions.raw ] ; then
  transcript=$dir/transcriptions.raw
  paste -d ' ' \
    <(cut -f 1  -d ' ' $transcript)  \
    <(cut -f 2- -d ' ' $transcript | sed -f conf/uwb.sed )  \
    > $dir/transcriptions.txt

fi
if [ -f $dir/transcriptions_for_lm.raw ] ; then
#paste -d ' '  \
#  <(cut -f 1 -d ' ' data/local/train/transcriptions_for_lm.raw)  \
#  <(cut -f 2- -d ' '  data/local/train/transcriptions_for_lm.raw | \
#    sed  -f conf/unify.sed | sed  -f conf/uhs.sed | sed -f conf/doubles.sed ) \
#  > data/local/train/transcriptions_for_lm.txt
  transcript=$dir/transcriptions_for_lm.raw
  paste -d ' ' \
    <(cut -f 1  -d ' ' $transcript)  \
    <(cut -f 2- -d ' ' $transcript | sed -f conf/uwb.sed  )  \
    > $dir/transcriptions_for_lm.txt
fi


