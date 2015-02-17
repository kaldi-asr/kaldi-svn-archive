#!/bin/bash

# Copyright 2015  Johns Hopkins University (Author: Jan Trmal).  Apache 2.0.

# Begin configuration section.
transcript=data/local/train/transcriptions.txt 
datadir=data/local/lang
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
cut -f 2- -d ' ' $transcript |  sed 's/ /\n/g' | sed '/^\s*$/d' | \
  sort | uniq -c |sort -k1nr -k2 | grep -v -F '<s>'  > $datadir/wordlist.txt

#now, lets install cmudict
( 
  cd  $datadir
  test -f cmudict-0.7b || \
    wget "http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b"
)

local/malach_find_pronunciation.pl $datadir/cmudict-0.7b conf/nonspeech.txt \
  $datadir/wordlist.txt $datadir/lexicon-iv.txt $datadir/wordlist-oov.txt 

#Next, we have to prepare the non-speech part of the lexicon
#We are grepping the "<unintelligible>" away, so that it gets mapped
#to <OOV>, which has kinda the same meaning

echo "<UNK>" | \
cat - conf/nonspeech.txt | \
grep -v "<unintelligible>"  |\
awk '{ print $1, $1}' > $datadir/lexicon-ns.txt

cat $datadir/lexicon-iv.txt | sort -u | cat $datadir/lexicon-ns.txt - \
  > $datadir/lexicon.txt
rm $datadir/lexiconp.txt 2>/dev/null || true

cut -f 2 -d ' ' $datadir/lexicon-ns.txt | sed '/^\s*$/d' \
  > $datadir/silence_phones.txt
cut -f 2- -d ' ' $datadir/lexicon.txt | sed 's/ /\n/g' | sort -u | \
  grep -v -F -f $datadir/silence_phones.txt | sed '/^\s*$/d' \
  > $datadir/nonsilence_phones.txt
echo "<pause>" > $datadir/optional_silence.txt
cat $datadir/silence_phones.txt | paste -s -d ' ' > $datadir/extra_questions.txt
cat $datadir/silence_phones.txt | grep hes | paste -s -d ' ' >> $datadir/extra_questions.txt

#Remap the $data/text
cat  $transcript | \
  local/map_oov.pl --symbol "<UNK>"  $datadir/lexicon.txt \
  > `dirname $transcript`/transcriptions_mapped.txt

local/malach_create_kaldi_files.pl data/local/train/audio.lst \
  data/local/train/transcriptions_mapped.txt data/local/train

mkdir -p data/local/lang
cut -f 1 -d ' ' $datadir/lexicon.txt  > data/local/lang/words.txt

rm -rf data/lang 2>/dev/null || true 
utils/prepare_lang.sh --share-silence-phones true  data/local/lang/ "<UNK>" data/local/lang/tmp data/lang
#local/train_lms_srilm.sh --oov-symbol "<UNK>" --train-text $transcript \
#  data/local data/local/lang
local/arpa2G.sh  data/local/lang/lm.gz  data/lang/ data/lang

