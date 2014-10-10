#!/bin/bash
      
# Copyright 2012-2014  Johns Hopkins University (Author: Daniel Povey).
#           2013  Xiaohui Zhang
#           2013  Guoguo Chen
#           2014  Tom Ko
# Apache 2.0.


# train_pnorm_multi.sh is modified from train_pnorm_fast.sh, which
# supports multi-language or multi-corpus training.

# Begin configuration section.
cmd=run.pl
num_epochs=5      # Number of epochs during which we reduce
                   # the learning rate; number of iterations is worked out from this.
num_epochs_extra=5 # Number of epochs after we stop reducing
                   # the learning rate.
num_iters_final=20 # Maximum number of final iterations to give to the
                   # optimization over the validation set (maximum)
initial_learning_rate=0.04
final_learning_rate=0.004
bias_stddev=0.5
pnorm_input_dim=3000 
pnorm_output_dim=300
first_component_power=1.0  # could set this to 0.5, sometimes seems to improve results.
p=2
minibatch_size=128 # by default use a smallish minibatch size for neural net
                   # training; this controls instability which would otherwise
                   # be a problem with multi-threaded update. 

samples_per_iter=200000 # each iteration of training, see this many samples
                        # per job.  This option is passed to get_egs.sh
num_jobs_nnet1=16   # Number of neural net jobs to run in parallel.  This option
                   # is passed to get_egs.sh.
num_jobs_nnet2=16
get_egs_stage=0
online_ivector_dir1=
online_ivector_dir2=
weight1=1
weight2=6
num_lang=2

shuffle_buffer_size=5000 # This "buffer_size" variable controls randomization of the samples
                # on each iter.  You could set it to 0 or to a large value for complete
                # randomization, but this would both consume memory and cause spikes in
                # disk I/O.  Smaller is easier on disk and memory but less random.  It's
                # not a huge deal though, as samples are anyway randomized right at the start.
                # (the point of this is to get data in different minibatches on different iterations,
                # since in the preconditioning method, 2 samples in the same minibatch can
                # affect each others' gradients.

add_layers_period=2 # by default, add new layers every 2 iterations.
num_hidden_layers=3
stage=-5

io_opts="-tc 5" # for jobs with a lot of I/O, limits the number running at one time.   These don't
splice_width=4 # meaning +- 4 frames on each side for second LDA
randprune=4.0 # speeds up LDA.
alpha=4.0 # relates to preconditioning.
update_period=4 # relates to online preconditioning: says how often we update the subspace.
num_samples_history=2000 # relates to online preconditioning
max_change_per_sample=0.075
precondition_rank_in=20  # relates to online preconditioning
precondition_rank_out=80 # relates to online preconditioning

mix_up=0 # Number of components to mix up to (should be > #tree leaves, if
        # specified.)
num_threads=16
parallel_opts="-pe smp 16 -l ram_free=1G,mem_free=1G" # by default we use 16 threads; this lets the queue know.
  # note: parallel_opts doesn't automatically get adjusted if you adjust num-threads.
combine_num_threads=8
combine_parallel_opts="-pe smp 8"  # queue options for the "combine" stage.
cleanup=true
egs_dir1=
egs_dir2=
lda_opts=
lda_dim=
egs_opts=
transform_dir=     # If supplied, overrides alidir
cmvn_opts=  # will be passed to get_lda.sh and get_egs.sh, if supplied.  
            # only relevant for "raw" features, not lda.
feat_type=  # Can be used to force "raw" features.
prior_subset_size=10000 # 10k samples per job, for computing priors.  Should be
                        # more than enough.
# End configuration section.


echo "$0 $@"  # Print the command line for logging
which parse_options.sh
if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 7 ]; then
  echo "Usage: $0 [opts] <data1> <lang1> <ali-dir1> <data2> <lang2> <ali-dir2> <exp-dir>"
  echo "<data1> <lang1> <ali-dir1> are from the in-domain language and <data2> <lang2> <ali-dir2> are from the out-domain language"
  echo " e.g.: $0 wsj_dir/data/train_si284 wsj_dir/data/lang wsj_dir/exp/tri4b_ali_si284 data/train data/lang exp/tri3_ali exp/tri4_nnet"
  echo ""
  echo "Main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config file containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --num-epochs <#epochs|15>                        # Number of epochs of main training"
  echo "                                                   # while reducing learning rate (determines #iterations, together"
  echo "                                                   # with --samples-per-iter and --num-jobs-nnet)"
  echo "  --num-epochs-extra <#epochs-extra|5>             # Number of extra epochs of training"
  echo "                                                   # after learning rate fully reduced"
  echo "  --initial-learning-rate <initial-learning-rate|0.02> # Learning rate at start of training, e.g. 0.02 for small"
  echo "                                                       # data, 0.01 for large data"
  echo "  --final-learning-rate  <final-learning-rate|0.004>   # Learning rate at end of training, e.g. 0.004 for small"
  echo "                                                   # data, 0.001 for large data"
  echo "  --num-hidden-layers <#hidden-layers|2>           # Number of hidden layers, e.g. 2 for 3 hours of data, 4 for 100hrs"
  echo "  --add-layers-period <#iters|2>                   # Number of iterations between adding hidden layers"
  echo "  --mix-up <#pseudo-gaussians|0>                   # Can be used to have multiple targets in final output layer,"
  echo "                                                   # per context-dependent state.  Try a number several times #states."
  echo "  --num-jobs-nnet1 <num-jobs|8>               # Number of parallel jobs to use for main neural net"
  echo "                                                   # training (will affect results as well as speed; try 8, 16)"
  echo "                                                   # Note: if you increase this, you may want to also increase"
  echo "                                                   # the learning rate."
  echo "  --num-jobs-nnet2 <num-jobs|8>              # Number of parallel jobs to use for out-language neural net"
  echo "  --weight1 <weight|1>                        # The weight for the in-language when the neural nets are averaged"
  echo "  --weight2 <weight|1>                       # The weight for the out-language when the neural nets are averaged"
  echo "  --num-threads <num-threads|16>                   # Number of parallel threads per job (will affect results"
  echo "                                                   # as well as speed; may interact with batch size; if you increase"
  echo "                                                   # this, you may want to decrease the batch size."
  echo "  --parallel-opts <opts|\"-pe smp 16 -l ram_free=1G,mem_free=1G\">      # extra options to pass to e.g. queue.pl for processes that"
  echo "                                                   # use multiple threads... note, you might have to reduce mem_free,ram_free"
  echo "                                                   # versus your defaults, because it gets multiplied by the -pe smp argument."
  echo "  --io-opts <opts|\"-tc 10\">                      # Options given to e.g. queue.pl for jobs that do a lot of I/O."
  echo "  --minibatch-size <minibatch-size|128>            # Size of minibatch to process (note: product with --num-threads"
  echo "                                                   # should not get too large, e.g. >2k)."
  echo "  --samples-per-iter <#samples|400000>             # Number of samples of data to process per iteration, per"
  echo "                                                   # process."
  echo "  --splice-width <width|4>                         # Number of frames on each side to append for feature input"
  echo "                                                   # (note: we splice processed, typically 40-dimensional frames"
  echo "  --lda-dim <dim|250>                              # Dimension to reduce spliced features to with LDA"
  echo "  --num-iters-final <#iters|20>                    # Number of final iterations to give to nnet-combine-fast to "
  echo "                                                   # interpolate parameters (the weights are learned with a validation set)"
  echo "  --first-component-power <power|1.0>              # Power applied to output of first p-norm layer... setting this to"
  echo "                                                   # 0.5 seems to help under some circumstances."
  echo "  --stage <stage|-9>                               # Used to run a partially-completed training process from somewhere in"
  echo "                                                   # the middle."


  exit 1;
fi

data1=$1
lang1=$2
alidir1=$3

data2=$4
lang2=$5
alidir2=$6

dir=$7

# Check some files.
for f in $data1/feats.scp $lang1/L.fst $alidir1/ali.1.gz $alidir1/final.mdl $alidir1/tree; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done
for f in $data2/feats.scp $lang2/L.fst $alidir2/ali.1.gz $alidir2/final.mdl $alidir2/tree; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done


# Set some variables.
num_leaves1=`tree-info $alidir1/tree 2>/dev/null | awk '{print $2}'` || exit 1
[ -z $num_leaves1 ] && echo "\$num_leaves1 is unset" && exit 1
[ "$num_leaves1" -eq "0" ] && echo "\$num_leaves1 is 0" && exit 1

num_leaves2=`tree-info $alidir2/tree 2>/dev/null | awk '{print $2}'` || exit 1
[ -z $num_leaves2 ] && echo "\$num_leaves2 is unset" && exit 1
[ "$num_leaves2" -eq "0" ] && echo "\$num_leaves2 is 0" && exit 1

nj=`cat $alidir1/num_jobs` || exit 1;  # number of jobs in alignment dir...
# in this dir we'll have just one job.
sdata=$data1/split$nj
utils/split_data.sh $data1 $nj

mkdir -p $dir/log
echo $nj > $dir/num_jobs
cp $alidir1/tree $dir

extra_opts=()
[ ! -z "$cmvn_opts" ] && extra_opts+=(--cmvn-opts "$cmvn_opts")
[ ! -z "$feat_type" ] && extra_opts+=(--feat-type $feat_type)
[ ! -z "$online_ivector_dir1" ] && extra_opts+=(--online-ivector-dir $online_ivector_dir1)
[ -z "$transform_dir" ] && transform_dir=$alidir1
extra_opts+=(--transform-dir $transform_dir)
extra_opts+=(--splice-width $splice_width)

extra_opts2=()
[ ! -z "$cmvn_opts" ] && extra_opts2+=(--cmvn-opts "$cmvn_opts")
[ ! -z "$feat_type" ] && extra_opts2+=(--feat-type $feat_type)
[ ! -z "$online_ivector_dir2" ] && extra_opts2+=(--online-ivector-dir $online_ivector_dir2)
[ -z "$transform_dir" ] && transform_dir=$alidir2
extra_opts2+=(--transform-dir $transform_dir)
extra_opts2+=(--splice-width $splice_width)

echo ${extra_opts2[@]}

if [ $stage -le -5 ]; then
  echo "Estimate the LDA by the second language"
  echo "$0: calling get_lda.sh"
  steps/nnet2/get_lda.sh $lda_opts "${extra_opts2[@]}" --cmd "$cmd" $data2 $lang2 $alidir2 $dir || exit 1;
fi


# these files will have been written by get_lda.sh
feat_dim=$(cat $dir/feat_dim) || exit 1;
ivector_dim=$(cat $dir/ivector_dim) || exit 1;
lda_dim=$(cat $dir/lda_dim) || exit 1;


if [ $stage -le -4 ] && [ -z "$egs_dir1" ]; then
  echo "$0: calling get_egs.sh"
  [ ! -z $spk_vecs_dir ] && egs_opts="$egs_opts --spk-vecs-dir $spk_vecs_dir";

  echo Prepare the egs

  steps/nnet2/get_egs.sh $egs_opts "${extra_opts[@]}" \
      --samples-per-iter $samples_per_iter \
      --num-jobs-nnet $num_jobs_nnet1 --stage $get_egs_stage \
      --cmd "$cmd" $egs_opts --io-opts "$io_opts" \
      $data1 $lang1 $alidir1 $dir/egs1 || exit 1;
    
  steps/nnet2/get_egs.sh $egs_opts "${extra_opts2[@]}" \
      --samples-per-iter $samples_per_iter \
      --num-jobs-nnet $num_jobs_nnet2 --stage $get_egs_stage \
      --cmd "$cmd" $egs_opts --io-opts "$io_opts" \
      $data2 $lang2 $alidir2 $dir/egs2 || exit 1;

fi


if [ -z $egs_dir1 ]; then
  egs_dir1=$dir/egs1/egs
fi
if [ -z $egs_dir2 ]; then
  egs_dir2=$dir/egs2/egs
fi

echo $cmvn_opts >$dir/cmvn_opts 2>/dev/null

iters_per_epoch1=`cat $egs_dir1/iters_per_epoch`  || exit 1;
! [ $num_jobs_nnet1 -eq `cat $egs_dir1/num_jobs_nnet` ] && \
  echo "$0: Warning: using --num-jobs-nnet=`cat $egs_dir1/num_jobs_nnet` from $egs_dir1"
num_jobs_nnet1=`cat $egs_dir1/num_jobs_nnet` || exit 1;

iters_per_epoch2=`cat $egs_dir2/iters_per_epoch`  || exit 1;
! [ $num_jobs_nnet2 -eq `cat $egs_dir2/num_jobs_nnet` ] && \
  echo "$0: Warning: using --num-jobs-nnet=`cat $egs_dir2/num_jobs_nnet` from $egs_dir2"
num_jobs_nnet2=`cat $egs_dir2/num_jobs_nnet` || exit 1;

if ! [ $num_hidden_layers -ge 1 ]; then
  echo "Invalid num-hidden-layers $num_hidden_layers"
  exit 1
fi

if [ $stage -le -3 ]; then
  lda_mat=$dir/lda.mat
  tot_input_dim=$[$feat_dim+$ivector_dim]
  online_preconditioning_opts="alpha=$alpha num-samples-history=$num_samples_history update-period=$update_period rank-in=$precondition_rank_in rank-out=$precondition_rank_out max-change-per-sample=$max_change_per_sample"
  stddev=`perl -e "print 1.0/sqrt($pnorm_input_dim);"`

  # to hidden.config it will write the part of the config corresponding to a
  # single hidden layer; we need this to add new layers. 
  cat >$dir/hidden.config <<EOF
AffineComponentPreconditionedOnline input-dim=$pnorm_output_dim output-dim=$pnorm_input_dim $online_preconditioning_opts learning-rate=$initial_learning_rate param-stddev=$stddev bias-stddev=$bias_stddev
PnormComponent input-dim=$pnorm_input_dim output-dim=$pnorm_output_dim p=$p
NormalizeComponent dim=$pnorm_output_dim
EOF

  this_pid=()
  for langid in `seq 1 $num_lang`; do
    echo "$0: initializing neural net for language $langid"
    cat >$dir/nnet.config$langid <<EOF
SpliceComponent input-dim=$tot_input_dim left-context=$splice_width right-context=$splice_width const-component-dim=$ivector_dim
FixedAffineComponent matrix=$lda_mat
AffineComponentPreconditionedOnline input-dim=$lda_dim output-dim=$pnorm_input_dim $online_preconditioning_opts learning-rate=$initial_learning_rate param-stddev=$stddev bias-stddev=$bias_stddev
PnormComponent input-dim=$pnorm_input_dim output-dim=$pnorm_output_dim p=$p
EOF
    if [ $first_component_power != 1.0 ]; then
      echo "PowerComponent dim=$pnorm_output_dim power=$first_component_power" >> $dir/nnet.config$langid
    fi
    cat >>$dir/nnet.config$langid <<EOF
NormalizeComponent dim=$pnorm_output_dim
AffineComponentPreconditionedOnline input-dim=$pnorm_output_dim output-dim=$[num_leaves$langid] $online_preconditioning_opts learning-rate=$initial_learning_rate param-stddev=0 bias-stddev=0
SoftmaxComponent dim=$[num_leaves$langid]
EOF

    adir=alidir$langid
    bdir=lang$langid
    $cmd $dir/log/nnet_init.$langid.log \
      nnet-am-init ${!adir}/tree ${!bdir}/topo "nnet-init $dir/nnet.config$langid -|" \
      $dir/0.$langid.mdl || touch $dir/.error &
    this_pid[$langid]=$!
  done
  for langid in `seq 1 $num_lang`; do
    wait ${this_pid[$langid]}
  done
fi


if [ $stage -le -2 ]; then
  this_pid=()
  for langid in `seq 1 $num_lang`; do
    echo "Training transition probabilities and setting priors for language $langid"
    adir=alidir$langid
    $cmd $dir/log/train_trans.$langid.log \
      nnet-train-transitions $dir/0.$langid.mdl "ark:gunzip -c ${!adir}/ali.*.gz|" $dir/0.$langid.mdl \
      || touch $dir/.error &
    this_pid[$langid]=$!
  done
  for langid in `seq 1 $num_lang`; do
    wait ${this_pid[$langid]}
  done
fi

num_iters_reduce=$[$num_epochs * $iters_per_epoch1];
num_iters_extra=$[$num_epochs_extra * $iters_per_epoch1];
num_iters=$[$num_iters_reduce+$num_iters_extra]

echo "$0: Will train for $num_epochs + $num_epochs_extra epochs, equalling "
echo "$0: $num_iters_reduce + $num_iters_extra = $num_iters iterations(referring to data of the first language), "
echo "$0: (while reducing learning rate) + (with constant learning rate)."

finish_add_layers_iter=$[$num_hidden_layers * $add_layers_period]
# This is when we decide to mix up from: halfway between when we've finished
# adding the hidden layers and the end of training.
mix_up_iter=$[($num_iters + $finish_add_layers_iter)/2]

if [ $num_threads -eq 1 ]; then
  parallel_suffix="-simple" # this enables us to use GPU code if
                         # we have just one thread.
  parallel_train_opts=
  if ! cuda-compiled; then
    echo "$0: WARNING: you are running with one thread but you have not compiled"
    echo "   for CUDA.  You may be running a setup optimized for GPUs.  If you have"
    echo "   GPUs and have nvcc installed, go to src/ and do ./configure; make"
  fi
else
  parallel_suffix="-parallel"
  parallel_train_opts="--num-threads=$num_threads"
fi

x=0

if [ $stage -le -1 ]; then
  echo "Averaging the initial models"
  #Make sure they have the same starting point

  this_pid=()
  $cmd $dir/log/initaverage.1.log \
    nnet-am-average --skip-last-layer=true $dir/0.1.mdl $dir/0.2.mdl $dir/start.1.mdl || touch $dir/.error &
  this_pid[1]=$!  
  $cmd $dir/log/initaverage.2.log \
    nnet-am-average --skip-last-layer=true $dir/0.2.mdl $dir/0.1.mdl $dir/start.2.mdl || touch $dir/.error &
  this_pid[2]=$!

  for langid in `seq 1 $num_lang`; do
    wait ${this_pid[$langid]}
  done
  mv $dir/start.1.mdl $dir/0.1.mdl
  mv $dir/start.2.mdl $dir/0.2.mdl
fi


while [ $x -lt $num_iters ]; do
  if [ $x -ge 0 ] && [ $stage -le $x ]; then
    echo Set off jobs doing some diagnostics, in the background.
    for langid in `seq 1 $num_lang`; do
      edir=egs_dir$langid
      $cmd $dir/log/compute_prob_valid.$x.$langid.log \
        nnet-compute-prob $dir/$x.$langid.mdl ark:${!edir}/valid_diagnostic.egs &
      $cmd $dir/log/compute_prob_train.$x.$langid.log \
        nnet-compute-prob $dir/$x.$langid.mdl ark:${!edir}/train_diagnostic.egs &
    done

    if [ $x -gt 0 ] && [ ! -f $dir/log/mix_up.$[$x-1].1.log ]; then
      $cmd $dir/log/progress.$x.log \
        nnet-show-progress --use-gpu=no $dir/$[$x-1].1.mdl $dir/$x.1.mdl \
          ark:$egs_dir1/train_diagnostic.egs '&&' \
        nnet-am-info $dir/$x.1.mdl &
      this_pid=$!
    fi
    
    echo "Training neural net (pass $x)"

    if [ $x -gt 0 ] && \
      [ $x -le $[($num_hidden_layers-1)*$add_layers_period] ] && \
      [ $[($x-1) % $add_layers_period] -eq 0 ]; then
      echo ADD layer
      #Here we make sure the models are adding the same hidden layer
      nnet-init --srand=$x $dir/hidden.config $dir/$x.0.mdl
      wait $this_pid
      $cmd LANG=1:$num_lang $dir/log/insert.LANG.log \
        nnet-insert $dir/$x.LANG.mdl $dir/$x.0.mdl $dir/$x.LANG.mdl || exit 1;
      rm $dir/$x.0.mdl
    fi

    this_pid=()
    for langid in `seq 1 $num_lang`; do
      echo Train nnet for language $langid
      edir=egs_dir$langid
      $cmd $parallel_opts JOB=1:$[num_jobs_nnet$langid] $dir/log/train.$x.$langid.JOB.log \
        nnet-shuffle-egs --buffer-size=$shuffle_buffer_size --srand=$x \
        ark:${!edir}/egs.JOB.$[$x%$[iters_per_epoch$langid]].ark ark:- \| \
         nnet-train$parallel_suffix $parallel_train_opts \
          --minibatch-size=$minibatch_size --srand=$x $dir/$x.$langid.mdl \
          ark:- $dir/$[$x+1].$langid.JOB.mdl \
        || touch $dir/.error &
      this_pid[$langid]=$!
    done
    for langid in `seq 1 $num_lang`; do
      wait ${this_pid[$langid]}
    done
  
    learning_rate=`perl -e '($x,$n,$i,$f)=@ARGV; print ($x >= $n ? $f : $i*exp($x*log($f/$i)/$n));' $[$x+1] $num_iters_reduce $initial_learning_rate $final_learning_rate`;

    nnets_list=()
    for langid in `seq 1 $num_lang`; do
      nnets_list[$langid]=""
      for n in `seq 1 $[num_jobs_nnet$langid]`; do
        nnets_list[$langid]="${nnets_list[$langid]} $dir/$[$x+1].$langid.$n.mdl"
      done
    done

    echo Do the averaging    

    this_pid=()
    for langid in `seq 1 $num_lang`; do
      $cmd $dir/log/average.$x.$langid.log \
        nnet-am-average ${nnets_list[$langid]} $dir/$[$x+1].$langid.tempmdl || touch $dir/.error &
      this_pid[$langid]=$!
    done
    for langid in `seq 1 $num_lang`; do
      wait ${this_pid[$langid]}
    done

    $cmd $dir/log/averagecross.$x.1.log \
      nnet-am-average --skip-last-layer=true \
      --weights=$weight1:$weight2 $dir/$[$x+1].1.tempmdl $dir/$[$x+1].2.tempmdl - \| \
      nnet-am-copy --learning-rate=$learning_rate - $dir/$[$x+1].1.mdl || touch $dir/.error &
    this_pid[1]=$!
      
    $cmd $dir/log/averagecross.$x.2.log \
      nnet-am-average --skip-last-layer=true \
      --weights=$weight2:$weight1 $dir/$[$x+1].2.tempmdl $dir/$[$x+1].1.tempmdl - \| \
      nnet-am-copy --learning-rate=$learning_rate - $dir/$[$x+1].2.mdl || touch $dir/.error &
    this_pid[2]=$!
   
    for langid in `seq 1 $num_lang`; do
      wait ${this_pid[$langid]}
    done
    for langid in `seq 1 $num_lang`; do
      rm  $dir/$[$x+1].$langid.tempmdl
    done

    if [ "$mix_up" -gt 0 ] && [ $x -eq $mix_up_iter ]; then
      echo Mixing up from $num_leaves to $mix_up components for language $langid
      $cmd LANG=1:$num_lang $dir/log/mix_up.$x.LANG.log \
        nnet-am-mixup --min-count=10 --num-mixtures=$mix_up \
        $dir/$[$x+1].LANG.mdl $dir/$[$x+1].LANG.mdl || exit 1;
    fi
    for langid in `seq 1 $num_lang`; do
      rm ${nnets_list[$langid]}
      [ ! -f $dir/$[$x+1].$langid.mdl ] && exit 1;
      if [ -f $dir/$[$x-1].$langid.mdl ] && $cleanup && \
         [ $[($x-1)%100] -ne 0  ] && [ $[$x-1] -le $[$num_iters-$num_iters_final] ]; then
        rm  $dir/$[$x-1].$langid.mdl
      fi
    done
  fi
  x=$[$x+1]
done

# Now do combination.
# At the end, final.mdl will be a combination of the last e.g. 10 models.
if [ $num_iters_final -gt $num_iters_extra ]; then
  echo "Setting num_iters_final=$num_iters_extra"
fi
start=$[$num_iters-$num_iters_final+1]
nnets_list=()
for langid in `seq 1 $num_lang`; do
  nnets_list[$langid]=""
  for x in `seq $start $num_iters`; do
    if [ $x -gt $mix_up_iter ]; then
      nnets_list[$langid]="${nnets_list[$langid]} $dir/$x.$langid.mdl"
    fi
  done
done


if [ $stage -le $num_iters ]; then
  for langid in `seq 1 $num_lang`; do
    echo "Doing final combination to produce final.$langid.mdl"
    edir=egs_dir$langid
    num_egs=`nnet-copy-egs ark:${!edir}/combine.egs ark:/dev/null 2>&1 | tail -n 1 | awk '{print $NF}'`
    mb=$[($num_egs+$combine_num_threads-1)/$combine_num_threads]
    [ $mb -gt 512 ] && mb=512
    $cmd $combine_parallel_opts $dir/log/combine.$langid.log \
      nnet-combine-fast --initial-model=100000 --num-lbfgs-iters=40 --use-gpu=no \
        --num-threads=$combine_num_threads \
        --verbose=3 --minibatch-size=$mb ${nnets_list[$langid]} ark:${!edir}/combine.egs \
        $dir/final.$langid.mdl || exit 1;

    $cmd $dir/log/normalize.$langid.log \
      nnet-normalize-stddev $dir/final.$langid.mdl $dir/final.$langid.mdl || exit 1;
      
    $cmd $dir/log/compute_prob_valid.final.$langid.log \
      nnet-compute-prob $dir/final.$langid.mdl ark:${!edir}/valid_diagnostic.egs &
    $cmd $dir/log/compute_prob_train.final.$langid.log \
      nnet-compute-prob $dir/final.$langid.mdl ark:${!edir}/train_diagnostic.egs &
  done
fi


if [ $stage -le $[$num_iters+1] ]; then
  for langid in `seq 1 $num_lang`; do
    echo "Getting average posterior for purposes of adjusting the priors for language $langid"
    # Note: this just uses CPUs, using a smallish subset of data.
    rm $dir/post.$langid.*.vec 2>/dev/null
    edir=egs_dir$langid
    $cmd JOB=1:$[num_jobs_nnet$langid] $dir/log/get_post.$langid.JOB.log \
      nnet-subset-egs --n=$prior_subset_size ark:${!edir}/egs.JOB.0.ark ark:- \| \
      nnet-compute-from-egs "nnet-to-raw-nnet $dir/final.$langid.mdl -|" ark:- ark:- \| \
      matrix-sum-rows ark:- ark:- \| vector-sum ark:- $dir/post.$langid.JOB.vec || exit 1;

    sleep 3;  # make sure there is time for $dir/post.$langid.*.vec to appear.

    $cmd $dir/log/vector_sum.$langid.log \
      vector-sum $dir/post.$langid.*.vec $dir/post.$langid.vec || exit 1;
    rm $dir/post.$langid.*.vec;

    echo "Re-adjusting priors based on computed posteriors"
    $cmd $dir/log/adjust_priors.$langid.log \
      nnet-adjust-priors $dir/final.$langid.mdl $dir/post.$langid.vec $dir/final.$langid.mdl || exit 1;
  done
fi

echo "We are interested in the first language, copy final.1.mdl to final.mdl"
cp $dir/final.1.mdl $dir/final.mdl

sleep 2

echo Done

if $cleanup; then
  echo Cleaning up data
  for langid in `seq 1 $num_lang`; do
    edir=egs_dir$langid
    steps/nnet2/remove_egs.sh ${!edir}
  done
fi
