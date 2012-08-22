#!/bin/bash
# Copyright 2012  Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.

# This script works on top of any features (including SAT features

# Train a model on top of existing features (no feature-space learning of any
# kind is done).  This script initializes the model from each stage of the
# previous system's model, judging the similarities based on overlap of counts
# in the tree stats.

# Begin configuration..
cmd=run.pl
stage=-5
# End configuration section.

[ -f path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# != 6 ]; then
  echo "Usage: local/train_two_level_tree.sh <num-leaves1> <num-leaves2> <data> <lang> <ali-dir> <exp-dir>"
  echo " e.g.: local/train_two_level_tree.sh 100 2500 data/train_si284 data/lang exp/tri3c_ali_si284 exp/tri4b"
  echo "Main options (for others, see top of script file)"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --config <config-file>                           # config containing options"
  echo "  --stage <stage>                                  # stage to do partial re-run from."
  exit 1;
fi

leaves1=$1
leaves2=$2
data=$3
lang=$4
alidir=$5
dir=$6

for f in $data/feats.scp $lang/L.fst $alidir/ali.1.gz $alidir/final.mdl; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

# Set various variables.
ciphonelist=`cat $lang/phones/context_indep.csl`
nj=`cat $alidir/num_jobs` || exit 1;
sdata=$data/split$nj
splice_opts=`cat $alidir/splice_opts 2>/dev/null` # frame-splicing options.

mkdir -p $dir/log
echo $nj >$dir/num_jobs
cp $alidir/splice_opts $dir 2>/dev/null
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;

## Set up features.
if [ -f $alidir/final.mat ]; then feat_type=lda; else feat_type=delta; fi
echo "$0: feature type is $feat_type"

case $feat_type in
  delta) sifeats="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas ark:- ark:- |";;
  lda) sifeats="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $alidir/final.mat ark:- ark:- |"
    cp $alidir/final.mat $dir    
    ;;
  *) echo "Invalid feature type $feat_type" && exit 1;
esac
if [ -f $alidir/trans.1 ]; then
  echo "$0: using transforms from $alidir"
  ln.pl $alidir/trans.* $dir # Link them to dest dir.
  feats="$sifeats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark,s,cs:$dir/trans.JOB ark:- ark:- |"
fi
##


if [ $stage -le -5 ]; then
  echo "$0: accumulating tree stats"
  $cmd JOB=1:$nj $dir/log/acc_tree.JOB.log \
    acc-tree-stats  --ci-phones=$ciphonelist $alidir/final.mdl "$feats" \
    "ark:gunzip -c $alidir/ali.JOB.gz|" $dir/JOB.treeacc || exit 1;
  [ "`ls $dir/*.treeacc | wc -w`" -ne "$nj" ] && echo "$0: Wrong #tree-stats" && exit 1;
  sum-tree-stats $dir/treeacc $dir/*.treeacc 2>$dir/log/sum_tree_acc.log || exit 1;
  rm $dir/*.treeacc
fi

if [ $stage -le -4 ]; then
  echo "$0: Getting questions for tree clustering."
  # preparing questions, roots file...
  cluster-phones $dir/treeacc $lang/phones/sets.int $dir/questions.int 2> $dir/log/questions.log || exit 1;
  cat $lang/phones/extra_questions.int >> $dir/questions.int
  compile-questions $lang/topo $dir/questions.int $dir/questions.qst 2>$dir/log/compile_questions.log || exit 1;

  echo "$0: Building the two-level tree"
  $cmd $dir/log/build_tree.log \
    build-tree-two-level --binary=false --verbose=1 --max-leaves-first=$leaves1 \
    --max-leaves-second=$leaves2 $dir/treeacc $lang/phones/roots.int \
    $dir/questions.qst $lang/topo $dir/tree $dir/tree.map || exit 1;
fi


echo Done building the tree

