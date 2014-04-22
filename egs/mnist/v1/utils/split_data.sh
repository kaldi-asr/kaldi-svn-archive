#!/bin/bash

# Copyright  2014     Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.


if [ $# != 2 ]; then
  echo "Usage: split_data.sh <data-dir> <num-to-split>"
  echo "This script will not split the data-dir if it detects that the output is newer than the input."
  exit 1
fi

data=$1
numsplit=$2

if ! [ $numsplit -gt 0 ]; then
  echo "Invalid number of splits $numsplit";
  exit 1;
fi

n=0;
feats=""
labels=""

nf=`cat $data/feats.scp | wc -l`
nl=`cat $data/labels | wc -l`

if [ $nf -ne $nl ]; then
  echo "split_data.sh: error, #lines in (feats.scp,labels) is ($nf,$nl); this script "
  echo " may produce incorrectly split data."
fi

s1=$data/split$numsplit/1
if [ ! -d $s1 ]; then 
  need_to_split=true
else 
  need_to_split=false
  for f in feats.scp labels; do
    if [[ -f $data/$f && ( ! -f $s1/$f || $s1/$f -ot $data/$f ) ]]; then
      need_to_split=true
    fi
  done
fi

if ! $need_to_split; then
  exit 0;
fi
  
for n in `seq $numsplit`; do
   mkdir -p $data/split$numsplit/$n
   feats="$feats $data/split$numsplit/$n/feats.scp"
   labels="$labels $data/split$numsplit/$n/labels"
done

utils/split_scp.pl $data/feats.scp $feats || exit 1

utils/split_scp.pl $data/labels $labels || exit 1


exit 0
