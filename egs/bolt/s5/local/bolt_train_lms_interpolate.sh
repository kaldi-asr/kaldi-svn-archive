#!/bin/bash

# To be run from one directory above this script.
# Train an individual LM for each corpus
# Interpolate them based on the heldout set from callhome
. ./path.sh

text_callhome=data/local/train.callhome/text
text_hkust=data/local/train.hkust/text
text_hub5=data/local/train.hub5/text
text_rt04f=data/local/train.rt04f/text
lexicon=data/local/dict/lexicon.txt 

for f in "$text_callhome" "$text_hkust" "$text_hub5" "$text_rt04f" "$lexicon"; do
  [ ! -f $x ] && echo "$0: No such file $f" && exit 1;
done

export LC_ALL=C # You'll get errors about things being not sorted, if you
# have a different locale.

dir=data/local/lm
sdir=$dir/srilm # use srilm toolkit for LM building
mkdir -p $dir
mkdir -p $sdir

# Ensure srilm is installed
#export PATH=$PATH:$KALDI_ROOT/tools/kaldi_lm
export PATH=$PATH:$KALDI_ROOT/tools/srilm/lm/bin/i686-m64:$KALDI_ROOT/tools/srilm/bin/i686-m64

heldout_sent=3000
cat $text_callhome | head -$heldout_sent > $sdir/heldout
gawk 'NR==FNR{utts[$1]; next;} !($1 in utts)' \
  $sdir/heldout $text_callhome > $sdir/train_callhome

cleantrain_callhome=$sdir/train_callhome.no_oov
cleantrain_hkust=$sdir/train_hkust.no_oov
cleantrain_hub5=$sdir/train_hub5.no_oov
cleantrain_rt04f=$sdir/train_rt04f.no_oov
cleanheldout=$sdir/heldout.no_oov

echo "Preparing clean LM training data for each corpus as well as clean heldout set"
cut -f 2- -d ' ' $sdir/train_callhome | awk -v lex=$lexicon 'BEGIN{while((getline<lex) >0){ seen[$1]=1; } } 
  {for(n=1; n<=NF;n++) {  if (seen[$n]) { printf("%s ", $n); } else {printf("<UNK> ");} } printf("\n");}' \
  > $cleantrain_callhome || exit 1;

cut -f 2- -d ' ' $text_hkust | awk -v lex=$lexicon 'BEGIN{while((getline<lex) >0){ seen[$1]=1; } } 
  {for(n=1; n<=NF;n++) {  if (seen[$n]) { printf("%s ", $n); } else {printf("<UNK> ");} } printf("\n");}' \
  > $cleantrain_hkust || exit 1;

cut -f 2- -d ' ' $text_hub5 | awk -v lex=$lexicon 'BEGIN{while((getline<lex) >0){ seen[$1]=1; } } 
  {for(n=1; n<=NF;n++) {  if (seen[$n]) { printf("%s ", $n); } else {printf("<UNK> ");} } printf("\n");}' \
  > $cleantrain_hub5 || exit 1;

cut -f 2- -d ' ' $text_rt04f | awk -v lex=$lexicon 'BEGIN{while((getline<lex) >0){ seen[$1]=1; } } 
  {for(n=1; n<=NF;n++) {  if (seen[$n]) { printf("%s ", $n); } else {printf("<UNK> ");} } printf("\n");}' \
      > $cleantrain_rt04f || exit 1;

cut -f 2- -d ' ' $sdir/heldout | awk -v lex=$lexicon 'BEGIN{while((getline<lex) >0){ seen[$1]=1; } } 
  {for(n=1; n<=NF;n++) {  if (seen[$n]) { printf("%s ", $n); } else {printf("<UNK> ");} } printf("\n");}' \
  > $cleanheldout || exit 1;

corpus=( callhome hkust hub5 rt04f )
echo "Train individual LM for each corpus"
for c in ${corpus[@]} ; do
  cleantrain=cleantrain_$c
  #cat ${!cleantrain} | awk '{for(n=1;n<=NF;n++) print $n; }' | \
  #  cat - <(grep -w -v '!SIL' $lexicon | awk '{print $1}') | \
  #  sort | uniq -c | sort -nr > $dir/unigram_$c.counts || exit 1;
  #cat $dir/unigram_$c.counts  | awk '{print $2}' | get_word_map.pl "<s>" "</s>" "<UNK>" > $dir/word_${c}_map \
  # || exit 1;
  #cat $dir/word_${c}_map | awk '{print $1}' | cat - <(echo "<s>"; echo "</s>" ) > $sdir/wordlist_${c}
  cat ${!cleantrain} | awk '{for(n=1;n<=NF;n++) print $n; }' | \
  cat - <(grep -w -v '!SIL' $lexicon | awk '{print $1}') | \
  sort -u | cat - <(echo "<s>"; echo "</s>"; echo "<UNK>") > $sdir/wordlist_$c || exit 1;
  ngram-count -text ${!cleantrain} -order 3 -limit-vocab -vocab $sdir/wordlist_$c -unk \
  -map-unk "<UNK>" -kndiscount -interpolate -lm $sdir/srilm.$c.o3g.kn.gz
done

echo "Calculate perplexity of each LM on the heldout set"
for c in ${corpus[@]} ; do
  ngram -debug 2 -order 3 -unk -lm $sdir/srilm.$c.o3g.kn.gz -ppl $cleanheldout > $sdir/$c.lm.ppl ;
done

echo "Determine the optimum mixture weight for LM interpolation"
compute-best-mix $sdir/*.lm.ppl > $sdir/best-mix.ppl 

# Extract lambda values from best-mix.ppl
lambdas=`cat $sdir/best-mix.ppl | sed 's=(==g' | sed 's=)==g' | awk '{for(i=NF-3;i<=NF;i++)printf("%s ",$i);print""}' | head -1`
LAMBDAS=(0.25 0.25 0.25 0.25)
let i=0
for l in $lambdas ; do
  LAMBDAS[$i]=$l
  let i=i+1
done

echo "LM models interpolation, stored as $sdir/mixed_lm.gz"
ngram -order 3 -unk -map-unk "<UNK>" \
  -lm      $sdir/srilm.${corpus[0]}.o3g.kn.gz -lambda  ${LAMBDAS[0]} \
  -mix-lm  $sdir/srilm.${corpus[1]}.o3g.kn.gz \
  -mix-lm2 $sdir/srilm.${corpus[2]}.o3g.kn.gz -mix-lambda2 ${LAMBDAS[2]} \
  -mix-lm3 $sdir/srilm.${corpus[3]}.o3g.kn.gz -mix-lambda3 ${LAMBDAS[3]} \
  -write-lm $sdir/mixed_lm.gz

echo "Test perplexity of the mixed LM on heldout set"
# Test perplexity of the mixed LM on heldout set
ngram -lm $sdir/mixed_lm.gz -ppl $cleanheldout

echo "Successfully generate a mixed language model from different corpus"
