#!/bin/bash

# Copyright 2014 Xiaohui Zhang  Apache 2.0.
# This srcipt computes a soft pdf mapping from one system to another, for use in neural-net 
# training on soft labels derived from other neural-net systems. Output is in <dir>/pdf.map. 

# Begin configuration section.
cmd=run.pl

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 3 ]; then
  echo "Usage: steps/nnet2/post/get_pdf_mapping.sh <source-alignment-dir> <dest-alignment-dir> <map-dir>"
  echo "Computes a soft pdf mapping from one system to another, for use in neural-net"
  echo "training on soft labels derived from other neural-net systems."
  echo "Output is in <map-dir>/pdf.map. "
  exit 1;
fi

src_alidir=$1
dest_alidir=$2
mapdir=$3

# Check some files.
for f in  $src_alidir/ali.1.gz $src_alidir/final.mdl $dest_alidir/ali.1.gz $dest_alidir/final.mdl $dest_alidir/tree; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

src_num_leaves=`tree-info $src_alidir/tree 2>/dev/null | awk '/num-pdfs/{print $NF}'` || exit 1;
dest_num_leaves=`tree-info $dest_alidir/tree 2>/dev/null | awk '/num-pdfs/{print $NF}'` || exit 1;
mkdir -p $mapdir
cp $dest_alidir/{final.mdl,tree} $mapdir

echo "$0: Computing a stochastic pdf mapping."
$cmd $mapdir/log/get_pdf_map.log \
  get-pdf-map --src_num_pdfs=$src_num_leaves --dest_num_pdfs=$dest_num_leaves \
    "ark:gunzip -c $src_alidir/ali.*.gz | ali-to-pdf $src_alidir/final.mdl ark:- ark:-|" \
    "ark:gunzip -c $dest_alidir/ali.*.gz | ali-to-pdf $dest_alidir/final.mdl ark:- ark:-|" \
    $mapdir/pdf.map || exit 1;

echo "$0: Finished computing the stochastic pdf mapping."
