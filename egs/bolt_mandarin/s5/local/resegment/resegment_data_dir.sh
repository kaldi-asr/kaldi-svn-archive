segmentation_opts="--isolated-resegmentation \
  --min-inter-utt-silence-length 0.5 \
  --silence-proportion 0.2 "
nj=4
cmd="run.pl"

. utils/parse_options.sh

if [ $# -ne 4 ]; then
  echo "Usage: local/resegment/resegment_data_dir.sh <src-dir> <model-dir> <temp-dir> <dest-dir>"
  echo " e.g.: local/resegment/resegment_data_dir.sh data/bolt_dev exp/tri4b_whole exp/make_seg data/bolt_dev.seg"
  echo 
  echo "Options:"
  echo "    --nj <njobs>              # Number of parallel jobs"
  echo "    --cmd <run.pl|queue.pl>"
  exit 1
fi

src_dir=$1
model_dir=$2
temp_dir=$3
dest_dir=$4

dataset_id=`basename $src_dir`
x=${dataset_id}_whole
workdir=$temp_dir/data/$x

unseg_dir=$workdir
mkdir -p $unseg_dir

echo "Creating the $unseg_dir/wav.scp file"
cp $src_dir/wav.scp $unseg_dir

echo "Creating the $unseg_dir/reco2file_and_channel file"
cat $unseg_dir/wav.scp | awk '{print $1, $1, "A";}' > $unseg_dir/reco2file_and_channel
cat $unseg_dir/wav.scp | awk '{print $1, $1;}' > $unseg_dir/utt2spk
utils/utt2spk_to_spk2utt.pl $unseg_dir/utt2spk > $unseg_dir/spk2utt
  
mfccdir=param
steps/make_mfcc_pitch.sh --nj $nj --cmd "$cmd" $unseg_dir exp/make_mfcc/$x $mfccdir || exit 1;
steps/compute_cmvn_stats.sh $unseg_dir exp/make_mfcc/$x $mfccdir || exit 1;

local/resegment/generate_segments.sh --nj $nj --cmd "$cmd" \
  --noise_oov false --oov-word "<unk>" --segmentation_opts "$segmentation_opts" \
  $unseg_dir $model_dir \
  $workdir $dest_dir || exit 1

num_hours=`cat ${dest_dir}/segments | \
  awk '{secs+= $4-$3;} END{print(secs/3600);}'`

echo "Number of hours of the newly segmented data: $num_hours"

for x in stm glm reco2file_and_channel; do
  if [ -f $src_dir/$x ]; then
    cp $src_dir/$x $dest_dir
  fi
done

if [ -f $src_dir/segments ]; then
  local/resegment/evaluate_segmentation.pl $src_dir/segments $dest_dir/segments > $workdir/log/evaluate_segments.log
fi

mfccdir=param
steps/make_mfcc_pitch.sh --nj $nj --cmd "$cmd" $dest_dir exp/make_mfcc/`basename $dest_dir` $mfccdir || exit 1;
steps/compute_cmvn_stats.sh $dest_dir exp/make_mfcc/`basename $dest_dir` $mfccdir || exit 1;

