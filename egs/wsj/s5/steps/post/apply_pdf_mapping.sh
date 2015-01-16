#!/bin/bash

# Copyright 2014 Xiaohui Zhang  Apache 2.0.
# This srcipt appliess a soft pdf mapping to a the posteriors/alignments of one system,
# in order to generate posteriors derived from the destination system, for use in neural-net 
# training on soft labels derived from the destination systems.

# Begin configuration section.
cmd=run.pl

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 3 ]; then
  echo "Usage: $0 <map-dir> [<input-pdf-post-dir>|<input-ali-dir>] <output-post-dir>"
  echo "e.g.: steps/post/apply_pdf_mapping.sh exp/nnet4a_4b_map exp/nnet4b_train_post exp/nnet4b_train_post_4a"
  echo "or: steps/post/apply_pdf_mapping.sh exp/nnet4a_4b_map exp/nnet4b_train_ali exp/nnet4b_train_post_4a"
  echo "Input is <input-pdf-post-dir>/pdf_post.*.{scp,ark} or <input-ali-dir>/ali.*.gz, output is"
  echo "<dest-post-dir>/pdf_post.*.{scp,ark}"
  exit 1;
fi

mapdir=$1
input_postdir=$2
output_postdir=$3

# Check some files.
for f in  $mapdir/pdf.map $mapdir/final.mdl $mapdir/tree $input_postdir/final.mdl; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

[ ! -f $input_postdir/post.1.scp ] && [ ! -f $input_postdir/ali.1.gz ] && echo "$0: no (soft) alignments provided" && exit 1;

mkdir -p $output_postdir/log
nj=`cat $input_postdir/num_jobs` || exit 1;  # number of jobs in alignment dir...
cp $mapdir/{final.mdl,tree} $output_postdir

echo "$0: Applying a stochastic pdf mapping to posteriors."
if [ -f $input_postdir/post.1.scp ]; then
  $cmd JOB=1:$nj $output_postdir/log/apply_pdf_map.JOB.log \
    apply-pdf-map $mapdir/pdf.map scp:$input_postdir/post.JOB.scp \
    ark,scp:$output_postdir/post.JOB.ark,$output_postdir/post.JOB.scp || exit 1;
else
  $cmd JOB=1:$nj $output_postdir/log/apply_pdf_map.JOB.log \
    apply-pdf-map $mapdir/pdf.map \
    "ark,s,cs:gunzip -c $input_postdir/ali.JOB.gz | ali-to-pdf $input_postdir/final.mdl ark:- ark:- | ali-to-post ark:- ark:- |" ark:- \| \
    rand-prune-post 0.01 ark:- ark,scp:$output_postdir/post.JOB.ark,$output_postdir/post.JOB.scp || exit 1;
fi

echo "$0: Finished applying a stochastic pdf mapping."
