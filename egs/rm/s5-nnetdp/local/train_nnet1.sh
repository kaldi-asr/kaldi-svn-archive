#!/bin/bash
# Copyright 2012  Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.

# This is a basic training recipe for Dan's "nnet1".

# Begin configuration section.
stage=-5
cmd=run.pl
#may need these options later.
#scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
#beam=10
#retry_beam=40
#realign_iters="10 20 30";
num_iters=35   # Number of iterations of training
max_iter_inc=25 # Last iter to increase #Gauss on.
power=0.2 # Exponent for number of gaussians according to occurrence counts
hidden_layer_size=500
initial_layer_context=2,2
hidden_layer_context=1,1
num_valid_utts=200
# End configuration section.

[ -f path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# != 6 ]; then
  echo "Usage: local/train_nnet1.sh <#pseudo-gauss> <data> <lang> <transform-dir> <tree-and-ali-dir> <exp-dir>"
  echo " e.g.: local/train_nnet1.sh 10000 data/train data/lang exp/tri2b_ali_si284 exp/tri3b"
  echo "Main options (for others, see top of script file)"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --config <config-file>                           # config containing options"
  echo "  --stage <stage>                                  # stage to do partial re-run from."
  exit 1;
fi

totgauss=$1
data=$2
lang=$3
transformdir=$4
tree_and_ali_dir=$5
dir=$6

for f in $data/feats.scp $lang/phones.txt $tree_and_ali_dir/ali.1.gz \
      $tree_and_ali_dir/tree $tree_and_ali_dir/tree.map \
      $transformdir/final.mat $transformdir/trans.1; do
  [ ! -f $f ] && echo "train_nnet1.sh: no such file $f" && exit 1;
done

numgauss=`copy-int-vector --binary=false $tree_and_ali_dir/tree.map - | wc -w` # might be
# off by two or so.
incgauss=$[($totgauss-$numgauss)/$max_iter_inc]  # per-iter #gauss increment

oov=`cat $lang/oov.int`
nj=`cat $tree_and_ali_dir/num_jobs` || exit 1;
silphonelist=`cat $lang/phones/silence.csl`
ciphonelist=`cat $lang/phones/context_indep.csl` || exit 1;
featdim=`feat-to-dim scp:$data/feats.scp -`

mkdir -p $dir/log


## Set up speaker-independent features.  Expect
## filterbank features plus LDA+MLLT plus probably fMLLR.
## Note: this is a bit different from normal because it's all in one process.
sifeats="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$data/utt2spk scp:$data/JOB/cmvn.scp scp:$data/feats.scp ark:- | transform-feats $transformdir/final.mat ark:- ark:- |"

## Get initial fMLLR transforms (possibly from alignment dir)
if [ -f $transformdir/trans.1 ]; then
  echo "$0: Using transforms from $transformdir"
  # o,cs below means we'll use each alignment only once (o), and the features
  # are sorted (cs).  It's not sorted (s) because of the way bash expansion of "*" works.
  feats="$sifeats transform-feats --utt2spk=ark:$data/utt2spk 'ark,o,cs:cat $transformdir/trans.*|' ark:- ark:- |"
else 
  echo "No transforms present."
  feats="$sifeats"
fi

cp $tree_and_ali_dir/tree $tree_and_ali_dir/tree.map $dir # Copy the tree and the map from 
  # leaves to "coarse" leaves.

if [ $stage -le -3 ]; then
  echo "Initializing the model"
  $cmd JOB=1 $dir/log/init_model.log \
    nnet1-init --layer-sizes=$featdim:$hidden_layer_size \
          --context-frames=$initial_layer_context \
          --learning-rates=0.0001 \
            $dir/tree $dir/tree.map $lang/topo $dir/1.nnet1 || exit 1;
fi

if [ $stage -le -1 ]; then
  # Links the alignments.
  utils/ln.pl $tree_and_ali_dir/ali.*.gz $dir
fi

# We're not going to align right now, so for now forget about the graphs.

#if [ $stage -le 0 ]; then
#
#  echo "$0: Compiling graphs of transcripts"
#  $cmd JOB=1:$nj $dir/log/compile_graphs.JOB.log \
#    compile-train-graphs $dir/tree $dir/1.nnet1  $lang/L.fst  \
#     "ark:utils/sym2int.pl --map-oov $oov -f 2- $lang/words.txt < $sdata/JOB/text |" \
#      "ark:|gzip -c >$dir/fsts.JOB.gz" || exit 1;
#fi

awk '{print $1}' $data/utt2spk | utils/shuffle_list.pl | head -$num_valid_utts > $dir/valid_uttlist

x=1
while [ $x -lt $num_iters ]; do
  echo Pass $x

  if [ $stage -le $x ]; then
    # o,cs below means we'll use each alignment only once (o), and the features
    # are sorted (cs).  It's not sorted (s) because of the way bash expansion of "*" works.
    $cmd $dir/train.$x.log \
      nnet1-train $dir/$x.nnet1 "$feats" "ark,o,cs:gunzip -c $dir/ali.*.gz|" $dir/valid_uttlist \
       $dir/$[$x+1].nnet1 || exit 1;
    # mix up.
    $cmd $dir/mixup.$x.log \
      nnet1-mix-up --target-neurons=$numgauss $dir/$[$x+1].nnet1 $dir/$[$x+1].nnet1 || exit 1;
  fi
  [ $x -le $max_iter_inc ] && numgauss=$[$numgauss+$incgauss];
  x=$[$x+1];
done

utils/summarize_warnings.pl $dir/log

echo Done
