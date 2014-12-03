#!/bin/bash
# Copyright 2013  Johns Hopkins University (authors: Yenda Trmal)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

# This is a scoring script for the CTMS in <decode-dir>/score_<LMWT>/${name}.ctm
# it tries to mimic the NIST scoring setup as much as possible (and usually does a good job)

# begin configuration section.
cmd=run.pl
cer=0
min_lmwt=7
max_lmwt=17
model=
stage=0
ctm_name=
glm=
case_insensitive=true
use_icu=true
icu_transform=''
#end configuration section.

echo $0 $@

[ -f ./path.sh ] && . ./path.sh
[ -f ./cmd.sh ]  && . ./cmd.sh
. parse_options.sh || exit 1;

if [ $# -ne 3 ]; then
  echo "Usage: $0 [options] <dataDir> <langDir|graphDir> <decodeDir>" && exit;
  echo " Options:"
  echo "    --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes."
  echo "    --cer (0|1)                     # compute CER in addition to WER"
  exit 1;
fi

data=$1
lang=$2 # This parameter is not used -- kept only for backwards compatibility
dir=$3

set -e 
set -o pipefail
set -u

#ScoringProgram=`which sclite` || ScoringProgram=$KALDI_ROOT/tools/sctk/bin/sclite
#[ ! -x $ScoringProgram ] && echo "Cannot find scoring program at $ScoringProgram" && \
#                            echo "You might need to go to $KALDI_ROOT/tools and call 'make sclite' " && exit 1;
ScoringProgram=`which hubscr.pl` || ScoringProgram=$KALDI_ROOT/tools/sctk/bin/hubscr.pl
[ ! -x "$ScoringProgram" ] && echo "Cannot find scoring program at $ScoringProgram." && \
                            echo "You might need to go to $KALDI_ROOT/tools and call 'make sclite' " && exit 1;
SortingProgram=`which hubscr.pl` || SortingProgram=$KALDI_ROOT/tools/sctk/bin/hubscr.pl
[ ! -x "$ScoringProgram" ] && echo "Cannot find scoring program at $ScoringProgram." && \
                              echo "You might need to go to $KALDI_ROOT/tools and call 'make sclite' " && exit 1;


for f in $data/stm  ; do
  [ ! -f $f ] && echo "$0: expecting file $f to exist" && exit 1;
done


if [ -z $ctm_name ] ; then
  name=`basename $data`; # e.g. eval2000
else
  name=$ctm_name
fi

if [ ! -z $glm ] ; then
  glm=" -g $glm "
fi

if [ $stage -le 0 ]; then
  # Remove some stuff we don't want to score, from the ctm.
  for seqid in `seq $min_lmwt $max_lmwt` ;  do
    x=$dir/score_$seqid/$name.ctm; 
    echo $x
    cp $x $x.bkup1;
    cat $x.bkup1 |  \
      grep -v "(*%[a-zA-Z]*)*"  |\
      grep -v "[^ ]*-_bw" |\
      grep -v "[^ ]*-_en" \
   |sort +0 -1 +1 -2 +2nb -3 > $x;
  done
fi

mkdir -p $dir/scoring/log

scoring_dir=$(dirname $ScoringProgram)
# hubscr.pl seems to fail if the directory where it is located is not on the path.
[ "$scoring_dir" != "." ] && PATH=$PATH:$scoring_dir

if [ $stage -le 1 ] ; then
  $cmd LMWT=$min_lmwt:$max_lmwt $dir/scoring/log/score.LMWT.log \
    set -e';' set -o pipefail';' \
    utils/fix_ctm.sh $data/stm $dir/score_LMWT/${name}.ctm '&&' \
    cp $data/stm $dir/score_LMWT/stm '&&' \
    $ScoringProgram -V -v -d -H -T -l arabic -h rt-stt $glm -r $dir/score_LMWT/stm  $dir/score_LMWT/${name}.ctm
fi



echo "Finished scoring on" `date`
exit 0

