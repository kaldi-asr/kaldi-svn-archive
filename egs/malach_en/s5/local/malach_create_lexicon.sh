#!/bin/bash

# Copyright 2015  Johns Hopkins University (Author: Jan Trmal).  Apache 2.0.

# Begin configuration section.
transcript=data/local/train/transcriptions_for_lm.txt 
datadir=data/local/lexicon
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


set -e
set -o pipefail
set -u



mkdir -p $datadir

#Generate words with counts (as we will be able to track easily how much
#of the text we did cover
cut -f 2- -d ' ' $transcript |  sed 's/ /\n/g' | \
  sort | uniq -c |sort -k1nr -k2  > $datadir/wordlist.txt

#now, lets install cmudict
( 
  cd  $datadir
  test -f cmudict-0.7b || \
    wget "http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b"
)

set -x 
local/malach_find_pronunciation.pl $datadir/cmudict-0.7b $datadir/wordlist.txt\
  $datadir/lexicon-iv.txt $datadir/wordlist-oov.txt 

