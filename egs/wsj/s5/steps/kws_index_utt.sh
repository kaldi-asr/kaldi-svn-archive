#!/bin/bash

# Copyright 2012  Lucas Ondel (Brno University of Technology)
# Apache 2.0


# Create a reverted index of all utterances of the lattices provided. The
# resulting index is a WFST itself.

# Configuration
acoustic_scale=0.83333
lm_scale=1.0

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh;
. parse_options.sh || exit 1;

if [ $# != 4 ]; then
   echo "Usage: steps/kws_index_utt.sh <lang-dir> <data-train-dir> <lattices-dir> <exp-dir>"
   echo " e.g.: steps/kws_index_utt.sh data/lang  data/train exp/mono0a/decode/lat.1 exp/kws/mono0a"
   echo "main options (for others, see top of script file)"
   echo "  --acoustic-scale                                 # acoustic scale use for lattice conversion (from Kaldi Lattice to OpenFst WFST)"
   echo "  --lm-scale                                       # language model scale use for lattice conversion (from Kaldi Lattice to OpenFst WFST)"
   exit 1;
fi

lang=$1
train=$2
latdir=$3
dir=$4

mkdir -p $dir

# Create the input/output symbol table
utils/kws_create_iosyms.awk $lang/words.txt $train/utt2spk > $dir/iosyms.txt

for lat in $latdir/lat.*.gz; do
gunzip -c $lat |\
lattice-to-fst --acoustic-scale=$acoustic_scale --lm-scale=$lm_scale ark:-  ark:- |\
  fstpush-kaldi --remove-total-weight=true --push-in-log=true ark:- ark:- |\
  fsttoindex $dir/iosyms.txt ark:- ark:- |\
  fstdeterminizestar --use-log=true ark:- ark:- |\
  fstpush-kaldi --remove-total-weight=false --push-in-log=true ark:- ark:-|\
  fstminimize-kaldi --encode ark:- ark:- |\
  fstunion-kaldi ark:- |\
  fstrmepsilon |\
  fstdeterminize-kws > $dir/$lat_index.fst
done

