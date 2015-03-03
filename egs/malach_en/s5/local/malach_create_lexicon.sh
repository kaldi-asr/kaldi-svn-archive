#!/bin/bash

# Copyright 2015  Johns Hopkins University (Author: Jan Trmal).  Apache 2.0.

# Begin configuration section.
nonspeech=conf/nonspeech.txt
oov="<UNK>"
sil="<SILENCE>"
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


set -e
set -o pipefail
set -u

src=$1
dest=$2
tmp=$2/tmp

mkdir -p $tmp

#Generate words with counts (as we will be able to track easily how much
#of the text we did cover
cut -f 2- -d ' ' $src |  sed 's/ /\n/g' | sed '/^\s*$/d' | \
  sort | uniq -c |sort -k1nr -k2 | grep -v -F '</*s>'  > $dest/wordlist.txt

#now, lets install cmudict
( 
  cd  $tmp
  test -f cmudict-0.7b || \
    wget "http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b"
)

local/malach_find_pronunciation.pl $tmp/cmudict-0.7b  $nonspeech \
  $dest/wordlist.txt $dest/lexicon-iv.txt $dest/wordlist-oov.txt 

#Next, we have to prepare the non-speech part of the lexicon
#We are grepping the "<unintelligible>" away, so that it gets mapped
#to <OOV>, which has kinda the same meaning
echo -e "$oov\n$sil" | \
cat - $nonspeech | sort -u | \
awk '{ print $1, tolower($1)}'| sort -u > $dest/lexicon-ns.txt

cat $dest/lexicon-iv.txt | sort -u | cat $dest/lexicon-ns.txt - | \
 sed '/^\s*$/d'  > $dest/lexicon.txt
rm $dest/lexiconp.txt 2>/dev/null || true

cut -f 2 -d ' ' $dest/lexicon-ns.txt | sed '/^\s*$/d' | sort -u \
  > $dest/silence_phones.txt
cut -f 2- -d ' ' $dest/lexicon.txt | sed 's/ /\n/g' | sort -u | \
  grep -v -F -f $dest/silence_phones.txt | sed '/^\s*$/d' \
  > $dest/nonsilence_phones.txt
grep -F "$sil"  $dest/lexicon-ns.txt | head -n 1 | \
  cut -f 2 -d ' ' > $dest/optional_silence.txt
cat $dest/silence_phones.txt | paste -s -d ' ' > $dest/extra_questions.txt

echo "Lexicon (and associate files) prepared..."
