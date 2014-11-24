#!/bin/bash

# To be run from one level above this directory
# Generate the text for the LM training

lexicon=data/local/dict/lexicon_bolt.txt 

. utils/parse_options.sh

if [ $# -lt 1 ]; then
  echo "Specify the location of the split files"
  exit 1;
fi

split=$1
lmdir=$2
text=$split/text.other

for f in "$text" "$lexicon"; do
  [ ! -f $x ] && echo "$0: No such file $f" && exit 1;
done

# This script takes no arguments.  It assumes you have already run
# fisher_data_prep.sh and fisher_prepare_dict.sh
# It takes as input the files
#data/train_all/text
#data/local/dict/lexicon.txt

rm -rf $lmdir

mkdir -p $lmdir
export LC_ALL=C # You'll get errors about things being not sorted, if you
# have a different locale.
(
. ./path.sh
# First make sure the kaldi_lm toolkit is installed.
 cd $KALDI_ROOT/tools || exit 1;
 if [ -d kaldi_lm ]; then
   echo Not installing the kaldi_lm toolkit since it is already there.
 else
   echo Downloading and installing the kaldi_lm tools
   if [ ! -f kaldi_lm.tar.gz ]; then
     wget http://www.danielpovey.com/files/kaldi/kaldi_lm.tar.gz || exit 1;
   fi
   tar -xvzf kaldi_lm.tar.gz || exit 1;
   cd kaldi_lm
   make || exit 1;
   echo Done making the kaldi_lm tools
 fi
) || exit 1;
#PATH=$KALDI_ROOT/tools/kaldi_lm:$PATH


mkdir -p $lmdir


cleantext=$lmdir/text.no_oov

cat $text |  awk -v lex=$lexicon 'BEGIN{while((getline<lex) >0){ seen[$1]=1; } } 
  {printf("%s ", $1); for(n=2; n<=NF;n++) {  if (seen[$n]) { printf("%s ", $n); } else {printf("<UNK> ");} } printf("\n");}' \
  > $text.filtered || exit 1;
cat $text |  awk -v lex=$lexicon 'BEGIN{while((getline<lex) >0){ seen[$1]=1; } } 
  {for(n=2; n<=NF;n++) {  if (seen[$n]) { printf("%s ", $n); } else {printf("<UNK> ");} } printf("\n");}' \
  > $cleantext || exit 1;


cat $cleantext | awk '{for(n=2;n<=NF;n++) print $n; }' | sort | uniq -c | \
   sort -nr > $lmdir/word.counts || exit 1;

# Get counts from acoustic training transcripts, and add  one-count
# for each word in the lexicon (but not silence, we don't want it
# in the LM-- we'll add it optionally later).


cat $cleantext | awk '{for(n=2;n<=NF;n++) print $n; }' | \
  cat - <(grep -w -v '<SIL>' $lexicon | awk '{print $1}' | sort -u ) | \
   sort | uniq -c | sort -nr > $lmdir/unigram.counts || exit 1;

# note: we probably won't really make use of <unk> as there aren't any OOVs
cat $lmdir/unigram.counts  | awk '{print $2}' | get_word_map.pl "<s>" "</s>" "<UNK>" > $lmdir/word_map \
   || exit 1;

# note: ignore 1st field of train.txt, it's the utterance-id.
cat $cleantext | awk -v wmap=$lmdir/word_map 'BEGIN{while((getline<wmap)>0)map[$1]=$2;}
  { for(n=2;n<=NF;n++) { printf map[$n]; if(n<NF){ printf " "; } else { print ""; }}}' | gzip -c >$lmdir/train.gz \
   || exit 1;

train_lm.sh --arpa --lmtype 3gram-mincount $lmdir || exit 1;

# Perplexity over 88307.000000 words (excluding 691.000000 OOVs) is 71.241332

# note: output is
# data/local/lm/3gram-mincount/lm_unpruned.gz 


exit 0
