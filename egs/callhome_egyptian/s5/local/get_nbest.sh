#!/usr/bin/env bash
# Author : Gaurav Kumar, Johns Hopkins University 
# Creates n-best lists from Kaldi lattices
# This script needs to be run from one level above this directory

nbest=100
cmd=run.pl

min_lmwt=7
max_lmwt=17

stage=0
out=

. path.sh

. ./utils/parse_options.sh

if [ $# -lt 3 ]; then
  echo "Incorrect number of parameters!"
  echo "$0 <decodedir> <lang> <nbest-out>"
  exit 1
fi

data=$1
decode=$3
lang=$2

symbols=$lang/words.txt
if [ -z "$out" ] ; then
  out=$decode/nbest
fi

set -e
set -o pipefail

if [ ! -d $decode ] ; then
  echo "Decode directory $decode does not exist!"
fi

mkdir -p $out

if [ $stage -le 0 ] ; then
  echo "getting n-best lists"
  # Extract n-best from the lattices
  #Convert the n-best lattice to linear word based sentences
  $cmd JOB=$min_lmwt:$max_lmwt $out/log/nbest.JOB.log \
    acwt=\`perl -e \"print 1.0/JOB\"\`\; \
    lattice-to-nbest --acoustic-scale=\$acwt --n=$nbest \
    ark:"gunzip -c $decode/lat.*.gz |" ark:- \| \
    nbest-to-linear ark,t:- \
      ark,t:"|gzip - >$out/JOB.ali.gz" \
      ark,t:"|gzip - >$out/JOB.words.gz"\
      ark,t:"|gzip - >$out/JOB.lmscore.gz"\
      ark,t:"|gzip - >$out/JOB.acscore.gz" || exit 1

  echo "Done getting n-best"
fi

if [ $stage -le 1 ]; then
  echo "Converting the nbest to text"
  #Convert the int to word for each sentence
  $cmd JOB=$min_lmwt:$max_lmwt $out/log/int2sym.JOB.log \
    set -e \; set -o pipefail\;\
    gunzip -c $out/JOB.words.gz \| \
    utils/int2sym.pl -f 2- $symbols \| \
    gzip -c - \> $out/JOB.text.gz || exit 1
  echo "Done"
fi

if [ $stage -le 2 ]; then
  echo "Creating nbest-ref files"
  #Convert the int to word for each sentence
  $cmd JOB=$min_lmwt:$max_lmwt $out/log/makeref.JOB.log \
    gunzip -c $out/JOB.text.gz \| \
    perl local/ref2nbest.pl $data/text \| \
    gzip -c \> $out/JOB.ref.gz || exit 1
  echo "Done"
fi

if [ $stage -le 3 ]; then
  $cmd JOB=$min_lmwt:$max_lmwt $out/log/makeref.JOB.log \
    gunzip -c $out/JOB.text.gz  \|\
      IBM/training/mbw2utf.pl \|\
      sed 's/_en / /g' \|\
      perl -ane 'print "$F[0] 1 A  0.0 1.0 " . join(" ", @F[1...$#F]) . "\n";' | \
      hubscr.pl sortSTM \| \
      hamzaNorm.pl -i stm -- - - \| \
      tanweenFilt.pl -a -i stm -- - - \| \
      csrfilt.sh -s -i stm -t hyp  IBM/scoring/ar2009.glm \|\
      cut -f 1,6- -d ' ' \|\
      gzip -c - \> $out/JOB.text.filt.gz
fi

if [ $stage -le 4 ]; then
fi
exit 0;


