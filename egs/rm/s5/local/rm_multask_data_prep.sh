#!/bin/bash
#
# Copyright 2014 Pegah Ghahremani 
# Apache 2.0.

# To be run from one directory above this script.

# This script uses data prepared by loca/rm_data_prep.sh script.
# You need to run loca/rm_data_prep.sh before running this script 

# This script is parametrized by number of copies N of each utterance(default, N=2), and the SNR s(default s=2)
# For each male utterance, N distinct female utterances are picked randomely and combined with 
# the male utterance with the male being greater by "energy_factor". 
# If one is longer, the other is extended with zeros at the end.The same thing is done for female utterances.
# The utterance id for the combined utterance is (1st speaker)_(1st text)_(2nd text), 
# and scp file is sorted on utterance id.
# The transcription file for combined utterances only contain 1st-spoken transcription and is also
# sorted on utterance id.


# Begin configuration section.
num_utt_copy=2 # The number of copies of each utterance
energy_factor=100 # The energy of 1st utterance'wave is greater than 
                  # the 2nd one by energy-factor in combined wave.
# End configuration section.

echo "$0 $@"  # Print the command line for logging 

if [ $# != 2 ]; then
  echo "Usage:$0 [options] <data-dir> <new-data-dir>";
  echo "e.g.: $0 data data_multitask"
  echo "options: "
  echo "--num-utt-copy     #The number of copies of each utterance "
  echo "--energy-factor   #The energy of 1st utterance'wave is greater than the 2nd one by energy-factor in combined wave."      
  exit 1;
fi
data=$1
new_data=$2

if [ ! -d $data ]; then
  echo "you need to run ../../local/rm_data_prep.sh /path/to/RM"
  exit 1;
fi

mkdir -p $new_data
mkdir -p tmp

. ./path.sh || exit 1; # for KALDI_ROOT
. parse_options.sh || exit 1;

for t in train test_mar87 test_oct87 test_feb89 test_oct89 test_feb91 test_sep92; do 
  newdir=$new_data/$t
  dir=$data/$t
  tmpdir=tmp/$t
  wave_utt1=$newdir/wave1.scp
  wave_utt2=$newdir/wave2.scp
  mkdir -p $newdir
  mkdir -p $tmpdir

  grep -w m $dir/spk2gender | awk '{print $1}' > $tmpdir/m_spk_list
  grep -w f $dir/spk2gender | awk '{print $1}' > $tmpdir/f_spk_list

  utils/filter_scp.pl $tmpdir/m_spk_list $dir/spk2utt | awk '{for(n=2;n<=NF;n++) print $n;}' > $tmpdir/male_utt_list
  utils/filter_scp.pl $tmpdir/f_spk_list $dir/spk2utt | awk '{for(n=2;n<=NF;n++) print $n;}' > $tmpdir/female_utt_list

  utils/filter_scp.pl $tmpdir/male_utt_list $dir/wav.scp > $tmpdir/male_utt_list.scp
  utils/filter_scp.pl $tmpdir/female_utt_list $dir/wav.scp > $tmpdir/female_utt_list.scp 
  
  utils/filter_scp.pl $tmpdir/male_utt_list $dir/text > $tmpdir/male_utt_text
  utils/filter_scp.pl $tmpdir/female_utt_list $dir/text > $tmpdir/female_utt_text

# Create $num_copy copies of each utterance in male_utterance list 
  awk -v num_copy="${num_utt_copy}" '{for(i=0;i<num_copy;i++)print $0}' <$tmpdir/male_utt_list.scp > $tmpdir/copied_male_utt.scp
# Create $num_copy copies of each utt's text in male_utt_text list
  awk -v num_copy="${num_utt_copy}" '{for(i=0;i<num_copy;i++)print $0}' <$tmpdir/male_utt_text > $tmpdir/copied_male_utt_text

# Create $num_copy copies of each utterance in female_utterance list 
  awk -v num_copy="${num_utt_copy}" '{for(i=0;i<num_copy;i++)print $0}' <$tmpdir/female_utt_list.scp > $tmpdir/copied_female_utt.scp
# Create $num_copy copies of each utt's text in male_utt_text list to combine it with utterance id later.
  awk -v num_copy="${num_utt_copy}" '{for(i=0;i<num_copy;i++)print $0}' <$tmpdir/female_utt_text > $tmpdir/copied_female_utt_text

  num_female_utt=`cat $tmpdir/female_utt_list.scp | wc -l`
  num_male_utt=`cat $tmpdir/male_utt_list.scp | wc -l`

  pick_f_utt_num=`perl -e "print int($num_female_utt/$num_utt_copy)*$num_utt_copy"` # largest multiple of num_utt_copy less than num_female_utt
                                                                                    # each time, we pick first $pick_f_utt_num female utterances from 
                                                                                    # shuffled female utterance list.
                                                                                    # So all "num_utt_copy" copies of each male utterance has distinct 
                                                                                    # corresponding female utterances.
  pick_m_utt_num=`perl -e "print int($num_male_utt / $num_utt_copy) * $num_utt_copy"` # largest multiple of num_utt_copy less than num_male_utt 

  f_shuff_num=`perl -e "print int($num_utt_copy * $num_male_utt / $pick_f_utt_num)+1"`   # number of times, we need to shuffle female utterance list 
                                                                                         # to generate N distinct random female speaker for all male speaker
  m_shuff_num=`perl -e "print int($num_utt_copy * $num_female_utt / $pick_m_utt_num)+1"` # number of times, we need to shuffle male utterance list
                                                                                         # to generate N distinct random male speaker for each female speaker 

  for i in $(seq $f_shuff_num); do
    utils/shuffle_list.pl --srand $((100 *i)) $tmpdir/female_utt_list.scp | head -n $pick_f_utt_num >> $tmpdir/male_utt2_list.scp;
  done
  head -n $((num_utt_copy * num_male_utt)) $tmpdir/male_utt2_list.scp > $newdir/male_utt2_list.scp

  for i in $(seq $m_shuff_num); do
    utils/shuffle_list.pl --srand $((100 * i)) $tmpdir/male_utt_list.scp | head -n $pick_m_utt_num >> $tmpdir/female_utt2_list.scp;
  done
  head -n $((num_utt_copy * num_female_utt)) $tmpdir/female_utt2_list.scp > $newdir/female_utt2_list.scp

# combine male and female wave and text file
  cat $tmpdir/copied_male_utt.scp $tmpdir/copied_female_utt.scp > $tmpdir/wave1.scp
  cat $newdir/male_utt2_list.scp $newdir/female_utt2_list.scp > $tmpdir/wave2.scp
  rm $newdir/male_utt2_list.scp $newdir/female_utt2_list.scp

# sort 1st utterance wavefile and change order of 2nd utterance list based on 1st wavefile.
  paste $tmpdir/wave1.scp $tmpdir/wave2.scp | sort -k1 > $tmpdir/sorted_wave.scp
  cat $tmpdir/sorted_wave.scp | cut -f1 > $wave_utt1
  cat $tmpdir/sorted_wave.scp | cut -f2 > $wave_utt2
  cat $tmpdir/copied_male_utt_text $tmpdir/copied_female_utt_text | sort -k1 > $tmpdir/text 

# combine wavefiles from wave1.scp with wave2.scp wavefiles
  combine-waves --energy-factor=$energy_factor scp:$wave_utt1 scp:$wave_utt2 ark,scp:$newdir/wav.ark,$newdir/wav.scp
  rm $wave_utt1 $wave_utt2
# Add utterance id for different copies of each utterances to text file 
  awk 'NR==FNR{a[FNR]=$1;next} NR>FNR{$1=a[FNR];print}'  $newdir/wav.scp $tmpdir/text > $newdir/text

# Create utt2spk
  cat $newdir/wav.scp | perl -ane 'm/^((\w+)\w_\w+_\w+) / || die; print "$1 $2\n"' > $newdir/utt2spk
  utils/utt2spk_to_spk2utt.pl $newdir/utt2spk > $newdir/spk2utt
rm -r $tmpdir
done
rm -r tmp
cp -rT $data/local $new_data/local
cp -rT $data/lang  $new_data/lang
cp -rT $data/lang_ug $new_data/lang_ug
