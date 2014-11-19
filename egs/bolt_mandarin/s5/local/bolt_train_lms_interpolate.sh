#!/bin/bash

# To be run from one directory above this script.
# Train an individual LM for each corpus
# Interpolate them based on the heldout set from callhome
# Copyright  2014 Johns Hopkins University (Minhua Wu)
. ./path.sh

text_callhome=data/local/train.callhome/text
text_hkust=data/local/train.hkust/text
text_hub5=data/local/train.hub5/text
text_rt04f=data/local/train.rt04f/text
text_uw=data/local/train.uw/text
text_add=data/local/train.add/text
lexicon=data/local/dict/lexicon.txt 
words_filter=data/local/dict/words_filter.txt

for f in "$text_callhome" "$text_hkust" "$text_hub5" "$text_rt04f" "$text_uw" "$text_add" "$lexicon"; do
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
cleantrain_uw=$sdir/train_uw.no_oov
cleantrain_add=$sdir/train_add.no_oov
cleanheldout=$sdir/heldout.no_oov

rm $sdir/*.no_oov
echo "Preparing clean LM training data for each corpus as well as clean heldout set"
echo "Prepare clean LM data for callhome: $cleantrain_callhome ..."
cut -f 2- -d ' ' $sdir/train_callhome | awk -v lex=$lexicon 'BEGIN{while((getline<lex) >0){ seen[$1]=1; } } 
  {for(n=1; n<=NF;n++) {  if (seen[$n]) { printf("%s ", $n); } else {printf("<UNK> ");} } printf("\n");}' \
  > $cleantrain_callhome || exit 1;

echo "Prepare clean LM data for hkust: $cleantrain_hkust ..."
cut -f 2- -d ' ' $text_hkust | awk -v lex=$lexicon 'BEGIN{while((getline<lex) >0){ seen[$1]=1; } } 
  {for(n=1; n<=NF;n++) {  if (seen[$n]) { printf("%s ", $n); } else {printf("<UNK> ");} } printf("\n");}' \
  > $cleantrain_hkust || exit 1;

echo "Prepare clean LM data for hub5: $cleantrain_hub5 ..."
cut -f 2- -d ' ' $text_hub5 | awk -v lex=$lexicon 'BEGIN{while((getline<lex) >0){ seen[$1]=1; } } 
  {for(n=1; n<=NF;n++) {  if (seen[$n]) { printf("%s ", $n); } else {printf("<UNK> ");} } printf("\n");}' \
  > $cleantrain_hub5 || exit 1;

echo "Prepare clean LM data for rt04f: $cleantrain_rt04f ..."
cut -f 2- -d ' ' $text_rt04f | awk -v lex=$lexicon 'BEGIN{while((getline<lex) >0){ seen[$1]=1; } } 
  {for(n=1; n<=NF;n++) {  if (seen[$n]) { printf("%s ", $n); } else {printf("<UNK> ");} } printf("\n");}' \
  > $cleantrain_rt04f || exit 1;

echo "Prepare clean LM data for uw data: $cleantrain_uw ..."
cut -f 2- -d ' ' $text_uw | grep -v -F -f $words_filter | awk -v lex=$lexicon 'BEGIN{while((getline<lex) >0){ seen[$1]=1; } } 
  {for(n=1; n<=NF;n++) {  if (seen[$n]) { printf("%s ", $n); } else {printf("<UNK> ");} } printf("\n");}' \
  > $cleantrain_uw || exit 1;

echo "Prepare clean LM data for new training data: $cleantrain_add ..."
cut -f 2- -d ' ' $text_add | awk -v lex=$lexicon 'BEGIN{while((getline<lex) >0){ seen[$1]=1; } } 
  {for(n=1; n<=NF;n++) {  if (seen[$n]) { printf("%s ", $n); } else {printf("<UNK> ");} } printf("\n");}' \
      > $cleantrain_add || exit 1;

echo "Prepare clean LM data for heldout data... $cleanheldout ..."
cut -f 2- -d ' ' $sdir/heldout | awk -v lex=$lexicon 'BEGIN{while((getline<lex) >0){ seen[$1]=1; } } 
  {for(n=1; n<=NF;n++) {  if (seen[$n]) { printf("%s ", $n); } else {printf("<UNK> ");} } printf("\n");}' \
  > $cleanheldout || exit 1;
  
echo "Prepare wordlist for lm training: $sdir/wordlist ..."
cat $lexicon | grep -w -v '!SIL' $lexicon | awk '{print $1}' | \
  cat - <(echo "<s>"; echo "</s>"; echo "<UNK>") | sort -u > $sdir/wordlist || exit 1;

rm $sdir/srilm.*.gz
corpus=( callhome hkust hub5 rt04f add uw )
echo "Train individual LM for each corpus"
for c in ${corpus[@]} ; do
  cleantrain=cleantrain_$c
  echo "Train LM for $c ..."
  ngram-count -text ${!cleantrain} -order 3 -limit-vocab -vocab $sdir/wordlist -unk \
    -map-unk "<UNK>" -kndiscount -interpolate -lm $sdir/srilm.$c.o3g.kn.gz
done

echo "prune uw LM ..."
pr_thres=0.0000005
ngram -lm $sdir/srilm.uw.o3g.kn.gz -write-lm $sdir/srilm.pr_uw.o3g.kn.gz -prune $pr_thres 

rm $sdir/*.lm.ppl
corpus=( callhome hkust hub5 rt04f add uw pr_uw )
echo "Calculate perplexity of each LM on the heldout set"
for c in ${corpus[@]} ; do
  echo "Perplexity of LM srilm.$c.o3g.kn.gz on heldout: "
  ngram -order 3 -unk -lm $sdir/srilm.$c.o3g.kn.gz -ppl $cleanheldout ;
  ngram -debug 2 -order 3 -unk -lm $sdir/srilm.$c.o3g.kn.gz -ppl $cleanheldout > $sdir/$c.lm.ppl ;
done

echo "Determine optimun mixture weight among callhome hkust hub5 rt04f add uw"
compute-best-mix $sdir/callhome.lm.ppl $sdir/hkust.lm.ppl $sdir/hub5.lm.ppl \
  $sdir/rt04f.lm.ppl $sdir/add.lm.ppl $sdir/uw.lm.ppl> $sdir/best-mix-full.ppl

echo "Determine optimun mixture weight among callhome hkust hub5 rt04f add pr_uw"
compute-best-mix $sdir/callhome.lm.ppl $sdir/hkust.lm.ppl $sdir/hub5.lm.ppl \
  $sdir/rt04f.lm.ppl $sdir/add.lm.ppl $sdir/pr_uw.lm.ppl> $sdir/best-mix-pr.ppl

echo "create an interpolated LM from callhome hkust hub5 rt04f add uw"
# Extract lambda values from best-mix-full.ppl
lambdas=`cat $sdir/best-mix-full.ppl | sed 's=(==g' | sed 's=)==g' | \
  awk '{for(i=NF-5;i<=NF;i++)printf("%s ",$i);print""}' | head -1`
LAMBDAS=(0.2 0.16 0.16 0.16 0.16 0.16)
let i=0
for l in $lambdas ; do
  LAMBDAS[$i]=$l
  let i=i+1
done

echo "LM models interpolation using 'callhome' 'hkust' 'hub5' 'rt04f' 'add' 'uw'"
echo "stored as $sdir/mixed_lm_full.gz"
echo "Mixture weight: ${LAMBDAS[0]} ${LAMBDAS[1]} ${LAMBDAS[2]} ${LAMBDAS[3]} \
  ${LAMBDAS[4]} ${LAMBDAS[5]}"
ngram -order 3 -unk -map-unk "<UNK>" \
  -lm      $sdir/srilm.${corpus[0]}.o3g.kn.gz -lambda  ${LAMBDAS[0]} \
  -mix-lm  $sdir/srilm.${corpus[1]}.o3g.kn.gz \
  -mix-lm2 $sdir/srilm.${corpus[2]}.o3g.kn.gz -mix-lambda2 ${LAMBDAS[2]} \
  -mix-lm3 $sdir/srilm.${corpus[3]}.o3g.kn.gz -mix-lambda3 ${LAMBDAS[3]} \
  -mix-lm4 $sdir/srilm.${corpus[4]}.o3g.kn.gz -mix-lambda4 ${LAMBDAS[4]} \
  -mix-lm5 $sdir/srilm.${corpus[5]}.o3g.kn.gz -mix-lambda5 ${LAMBDAS[5]} \
  -write-lm $sdir/mixed_lm_full.gz
echo "Test perplexity of the mixed LM (all full) on heldout set"
# Test perplexity of the mixed LM (full) on heldout set
ngram -lm $sdir/mixed_lm_full.gz -ppl $cleanheldout

echo "create an interpolated LM from callhome hkust hub5 rt04f add pr_uw"
# Extract lambda values from best-mix-pr.ppl
lambdas=`cat $sdir/best-mix-pr.ppl | sed 's=(==g' | sed 's=)==g' | \
   awk '{for(i=NF-5;i<=NF;i++)printf("%s ",$i);print""}' | head -1`
LAMBDAS=(0.2 0.16 0.16 0.16 0.16 0.16)
let i=0
for l in $lambdas ; do
  LAMBDAS[$i]=$l
  let i=i+1
done

echo "LM models interpolation using 'callhome' 'hkust' 'hub5' 'rt04f' 'add' 'pr_uw'"
echo "stored as $sdir/mixed_lm_pr.gz"
echo "Mixture weight: ${LAMBDAS[0]} ${LAMBDAS[1]} ${LAMBDAS[2]} ${LAMBDAS[3]} \
  ${LAMBDAS[4]} ${LAMBDAS[5]}"
ngram -order 3 -unk -map-unk "<UNK>" \
  -lm      $sdir/srilm.${corpus[0]}.o3g.kn.gz -lambda  ${LAMBDAS[0]} \
  -mix-lm  $sdir/srilm.${corpus[1]}.o3g.kn.gz \
  -mix-lm2 $sdir/srilm.${corpus[2]}.o3g.kn.gz -mix-lambda2 ${LAMBDAS[2]} \
  -mix-lm3 $sdir/srilm.${corpus[3]}.o3g.kn.gz -mix-lambda3 ${LAMBDAS[3]} \
  -mix-lm4 $sdir/srilm.${corpus[4]}.o3g.kn.gz -mix-lambda4 ${LAMBDAS[4]} \
  -mix-lm5 $sdir/srilm.${corpus[6]}.o3g.kn.gz -mix-lambda5 ${LAMBDAS[5]} \
  -write-lm $sdir/mixed_lm_pr.gz
echo "Test perplexity of the mixed LM (with uw pruned) on heldout set"
# Test perplexity of the mixed LM (with uw pruned) on heldout set
ngram -lm $sdir/mixed_lm_pr.gz -ppl $cleanheldout

echo "Successfully generate mixed language models from different corpus"
