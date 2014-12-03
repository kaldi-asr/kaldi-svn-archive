#!/usr/bin/env bash
# Author : Gaurav Kumar, Jan Trmal (Johns Hopkins University)
# Converts the ctm file to the kaldi-format table
cmd=run.pl

min_lmwt=7
max_lmwt=17

stage=0

. path.sh

. ./utils/parse_options.sh

if [ $# -lt 3 ]; then
  echo "Incorrect number of parameters!"
  echo "$0 <decodedir> <lang> <nbest-out>"
  exit 1
fi

data=$1
sgml=$2
decode=$4
lang=$3

symbols=$lang/words.txt

set -e
set -o pipefail

if [ ! -d $decode ] ; then
  echo "Decode directory $decode does not exist!"
fi

$cmd JOB=$min_lmwt:$max_lmwt $decode/log/filter1best.JOB.log \
  cat $decode/score_JOB/`basename $data`.utt.ctm \| \
    local/ctm2text.pl $data/text \| \
    sed 's/[^ ]*-_bw//g' \| \
    sed "'s/(*%[a-zA-Z]*)*//g'" \| \
    IBM/training/mbw2utf.pl \|\
    sed 's/_en / /g' \| \
    perl -ane 'print "$F[0] 1 A  0.0 1.0 " . join(" ", @F[1...$#F]) . "\n";' \| \
    hamzaNorm.pl -i stm -- - - \| \
    tanweenFilt.pl -a -i stm -- - - \| \
    csrfilt.sh -s -i stm -t hyp IBM/scoring/ar2009.utf8.glm \|\
    cut -f 1,6- -d ' ' \|\
    local/text2sgml.pl $sgml $data \
    \> $decode/score_JOB/`basename $data`.utt.filt.sgml

$cmd JOB=$min_lmwt:$max_lmwt $decode/log/filterref.JOB.log \
  cat $data/text \| \
    IBM/training/mbw2utf.pl \|\
    sed 's/_en / /g' \| \
    perl -ane 'print "$F[0] 1 A  0.0 1.0 " . join(" ", @F[1...$#F]) . "\n";' \| \
    hamzaNorm.pl -i stm -- - - \| \
    tanweenFilt.pl -a -i stm -- - - \| \
    csrfilt.sh -s -i stm -t hyp IBM/scoring/ar2009.utf8.glm \|\
    cut -f 1,6- -d ' ' \|\
    local/text2sgml.pl $sgml $data \
    \> $decode/score_JOB/`basename $data`.ref.filt.sgml

$cmd JOB=$min_lmwt:$max_lmwt $decode/log/filter1best2.JOB.log \
  cat $decode/score_JOB/`basename $data`.utt.ctm \| \
    local/ctm2text.pl $data/text \| \
    sed 's/[^ ]*-_bw//g' \| \
    sed "'s/(*%[a-zA-Z]*)*//g'" \| \
    IBM/training/mbw2utf.pl \|\
    sed 's/_en / /g' \| \
    local/text2sgml.pl $sgml  $data \
    \> $decode/score_JOB/`basename $data`.utt.raw.sgml

$cmd JOB=$min_lmwt:$max_lmwt $decode/log/filterref2.JOB.log \
  cat $data/text \| \
    IBM/training/mbw2utf.pl \|\
    sed 's/_en / /g' \| \
    local/text2sgml.pl $sgml $data \
    \> $decode/score_JOB/`basename $data`.ref.raw.sgml
exit



