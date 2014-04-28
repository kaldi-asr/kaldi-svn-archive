#!/bin/bash

# Copyright 2014  Xiaohui Zhang
# Apache 2.0.


# This is a simple test script for calculating test error rate of nnet based image recognition.

if [ -f path.sh ]; then . ./path.sh; fi

if [ $# != 2 ]; then
  echo "Usage: $0 [opts] <test-data> <exp-dir>"
  echo " e.g.: $0 data/t10k exp/nnet1"
  echo ""
  exit 1;
fi


test_data=$1
dir=$2
if [ ! -f $dir/posteriors ]; then
  copy-feats scp:$test_data/feats.scp ark:- | nnet2-compute --raw=true --pad-input=false $dir/final.nnet ark:- ark,t:$dir/posteriors
  sed -i 'N;s/\[\n/\[/g' $dir/posteriors
fi
cat $dir/posteriors |   awk '{max=$3; argmax=3; for(f=3;f<NF;f++) { if ($f>max)
                        { max=$f; argmax=f; }} print $1, (argmax - 3); }' > $dir/output
compute-wer --mode=present --text ark:$test_data/labels ark:$dir/output
