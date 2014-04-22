#!/bin/bash

# Copyright 2012-2014 Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.



# Begin configuration section.
cmd=run.pl
num_valid_egs_combine=0    # #valid egs for combination weights at the very end.
num_train_egs_combine=5000 # # train egs for combination weights at the very end.
num_egs_diagnostic=4000    # number of egs for "compute_prob" jobs
samples_per_iter=15000      # each iteration of training, see this many samples
                            # per job.  This is just a guideline; it will pick a number
                            # that divides the number of samples in the entire data.
num_jobs_nnet=16   # Number of neural net jobs to run in parallel
io_opts="-tc 8"    # prevents too many jobs from running at once.
nj=4
stage=0

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 3 ]; then
  echo "Usage: steps/nnet2/get_egs.sh [opts] <train-data> <valid-data> <exp-dir>"
  echo " e.g.: steps/nnet2/get_egs.sh data/train_train data/train_heldout exp/nnet1"
  echo ""
  echo "Main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config file containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --num-jobs-nnet <num-jobs|16>                    # Number of parallel jobs to use for main neural net"
  echo "                                                   # training (will affect results as well as speed; try 8, 16)"
  echo "                                                   # Note: if you increase this, you may want to also increase"
  echo "                                                   # the learning rate."
  echo "  --samples-per-iter <#samples|400000>             # Number of samples of data to process per iteration, per"
  echo "                                                   # process."
  echo "  --num-egs-diagnostic <#egs|4000>                 # Number of egs used in computing (train,valid) diagnostics"
  echo "  --num-valid-egs-combine <#egs|0>                 # Number of heldout egs used in getting combination"
  echo "                                                   # weights at the very end of training"
  echo "  --num-train-egs-combine <#egs|5000>              # Number of heldout egs used in getting combination weights "
  echo "                                                   # at the very end of training"
  echo "  --stage <stage|0>                                # Used to run a partially-completed training process from"
  echo "                                                   # somewhere in the middle."
  exit 1;
fi

train_data=$1
valid_data=$2
dir=$3

# Check some files.
for f in {$train_data,$valid_data}/{feats.scp,labels}; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

mkdir -p $dir/egs || exit 1;

echo $num_jobs_nnet > $dir/num_jobs


mkdir -p $dir/log


if [ $stage -le 0 ]; then
  echo "$0: working out number of egs of training data"
  num_egs=$(cat $train_data/feats.scp | wc -l) || exit 1;
  echo $num_egs > $dir/num_egs
else
  num_egs=`cat $dir/num_egs` || exit 1;
fi

# Working out number of iterations per epoch.
iters_per_epoch=`perl -e "print int($num_egs/($samples_per_iter * $num_jobs_nnet) + 0.5);"` || exit 1;
[ $iters_per_epoch -eq 0 ] && iters_per_epoch=1
samples_per_iter_real=$[$num_egs/($num_jobs_nnet*$iters_per_epoch)]
echo "$0: Every epoch, splitting the data up into $iters_per_epoch iterations,"
echo "$0: giving samples-per-iteration of $samples_per_iter_real (you requested $samples_per_iter)."

num_pieces=$[$iters_per_epoch * $num_jobs_nnet];

output_archives=""
for j in $(seq $num_jobs_nnet); do
  for n in $(seq 0 $[$iters_per_epoch-1]); do
    output_archives="$output_archives ark:$dir/egs/egs.$j.$n.ark"
  done
done


if [ $stage -le 1 ]; then
  echo "$0: getting egs for diagnostics (train-set)"
  $cmd $dir/log/get_egs_train_diagnostic.log \
    utils/shuffle_list.pl $train_data/feats.scp \| head -n $num_egs_diagnostic \| sort \| \
    nnet2v-get-egs scp:- ark:$train_data/labels ark:$dir/egs/train_diagnostic.egs || exit 1;

  echo "$0: getting egs for diagnostics (validation-set)"
  $cmd $dir/log/get_egs_valid_diagnostic.log \
    utils/shuffle_list.pl $valid_data/feats.scp \| head -n $num_egs_diagnostic \| sort \| \
    nnet2v-get-egs scp:- ark:$valid_data/labels ark:$dir/egs/valid_diagnostic.egs || exit 1;

  echo "$0: getting egs for combination."

  $cmd $dir/log/get_egs_combine.log \
    nnet2v-get-egs "scp:utils/shuffle_list.pl $train_data/feats.scp | head -n $num_train_egs_combine; utils/shuffle_list.pl $valid_data/feats.scp | head -n $num_valid_egs_combine |" \
    "ark:cat $train_data/labels $valid_data/labels |" ark:$dir/egs/combine.egs || exit 1;
fi

if [ $stage -le 2 ]; then

# This is all done with one job.  In the future if we need to process more data,
# we can modify it to use more jobs.
  echo "$0: getting egs for training"

  $cmd $dir/log/get_egs.log \
    nnet2v-get-egs scp:$train_data/feats.scp ark:$train_data/labels ark:- \| \
    nnet-copy-egs ark:- $output_archives || exit 1;
fi

# Other scripts might need to know the following info:
echo $num_jobs_nnet >$dir/egs/num_jobs_nnet
echo $iters_per_epoch >$dir/egs/iters_per_epoch
echo $samples_per_iter_real >$dir/egs/samples_per_iter

echo "$0: Finished preparing training examples"
