#!/bin/bash
#
# Johns Hopkins University : (Gaurav Kumar)
# The input is the Callhome Egyptian Arabic Dataset which contains *.sph files 
# In addition the transcripts are needed as well. 

#TODO: Rewrite intro, copyright stuff and dir information
# To be run from one directory above this script.

stage=0

echo $0 $@
. ./utils/parse_options.sh

set -e
set -o pipefail

if [ $# -lt 3 ]; then
   echo "Arguments should be the location of the Callhome Egyptian Arabic Speech and Transcript Directories, se
e ../run.sh for example."
   exit 1;
fi


TEXT_DATA=$2
AUDIO_DATA=$1
OUTPUT=$3

mkdir -p $OUTPUT

. ./path.sh || exit 1; # Needed for KALDI_ROOT

sph2pipe=`which sph2pipe` || sph2pipe=$KALDI_ROOT/tools/sph2pipe_v2.5/sph2pipe
if [ ! -x $sph2pipe ]; then
   echo "Could not find (or execute) the sph2pipe program at $sph2pipe";
   exit 1;
fi
sph2pipe=`readlink -f $sph2pipe`


# Basic spot checks to see if we got the data that we needed
if [ ! -d $TEXT_DATA ] || [ ! -d $AUDIO_DATA ]; then
        echo "Either $TEXT_DATA or $AUDIO_DATA data directories do not exist (or are not directories)"
        exit 1;
fi

train_dir=$OUTPUT/train.callhome
evltest_dir=$OUTPUT/evltest.callhome
devtest_dir=$OUTPUT/devtest.callhome

if [ "$stage" -le 0 ] ; then
  for dataset in train devtest ; do
    eval datadir=\$${dataset}_dir
    
    echo "datadir=$datadir"
    mkdir -p $datadir
    find -L $AUDIO_DATA -ipath "*/$dataset/*" -iname '*.SPH' > $datadir/sph.flist
    find -L $TEXT_DATA -ipath "*/transcrp/$dataset/script/*" -iname '*.scr' > $datadir/txt.flist
    
    eval fcount_${dataset}=`cat $datadir/sph.flist| wc -l`
    eval fcount_t_${dataset}=`cat $datadir/txt.flist| wc -l`

    eval echo "\\\$fcount_${dataset}=\$fcount_${dataset}"
    eval echo "\\\$fcount_t_${dataset}=\$fcount_t_${dataset}"
  done
  for datadir in $evltest_dir ; do
    #This is because LDC has a poor QA and they called the same directory 
    #evaltest (for text data) and evltest (for audio data)
    echo "datadir=$datadir"
    mkdir -p $datadir
    find -L $AUDIO_DATA -ipath "*/evltest/*" -iname '*.SPH' > $datadir/sph.flist
    find -L $TEXT_DATA -ipath "*/transcrp/evaltest/script/*" -iname '*.scr' > $datadir/txt.flist
    
    fcount_evltest=`cat $datadir/sph.flist| wc -l`
    fcount_t_evltest=`cat $datadir/txt.flist| wc -l`

    echo "\$fcount_evltest=$fcount_evltest"
    echo "\$fcount_t_evltest=$fcount_t_evltest"
  done

  #Now check if we got all the files that we needed
  if [ "$fcount_train" -ne 80   ] || [ "$fcount_t_train" -ne 80   ] || \
     [ "$fcount_devtest" -ne 20 ] || [ "$fcount_t_devtest" -ne 20 ] || \
     [ "$fcount_evltest" -ne 20 ] || [ "$fcount_t_evltest" -ne 20 ]; then                                                                               
          echo "Incorrect number of files in the data directories"
          echo "The paritions should contain 80/20/20 files"
          echo "Current counts: "
          echo "  TRAIN  : audio=$fcount_train text=$fcount_t_train"
          echo "  DEVTEST: audio=$fcount_devtest text=$fcount_t_devtest"
          echo "  EVLTEST: audio=$fcount_evltest text=$fcount_t_evltest"
          exit 1;                                                                    
  fi   
fi 
if [ $stage -le 1 ]; then
  for dataset in train devtest evltest ; do
    eval datadir=\$${dataset}_dir
    
    cat $datadir/txt.flist |\
  	perl local/callhome_data_convert.pl $datadir/sph.flist $datadir
    uconv -f utf-8 -t utf-8 -x "Arabic-Latin" $datadir/transcripts.txt > $datadir/transcripts.transliterated.txt
  done
fi

#icu_transform=( -x "Arabic-Latin;Any-Upper")
#icu_transform=( -x "Arabic-Latin")
icu_transform=( )
if [ $stage -le 2 ]; then
  for dataset in train devtest evltest ; do
    eval datadir=\$${dataset}_dir
    
    cat $datadir/transcripts.txt |\
      #First, we remap the arabic punctuation marks to normal latin ones
      #I'm removing the sentence //she just...// because I dont really know if thats comment or what is it
      #Lastly, map (( )) to (()) to simplify parsing and remove [[.*]] comments
      perl -e 'use utf8; 
               use charnames ":full";
               binmode STDIN, "utf8";
               binmode STDOUT, "utf8";
               while (<>) {
                s/\N{ARABIC QUESTION MARK}/ ? /g;
                s/\N{ARABIC SEMICOLON}/;/g;
                s/\N{ARABIC COMMA}/ , /g;
                print;
               }
              ' | \
      grep -v -T "//she just hanged ((the phone))//" |\
      #grep "AR_5627-B-078221-078404" |\
      #grep "AR_4849-B-052026-052517" |\
      sed  's/((\s\s*))/(())/g' |\
      perl -pe 's/\[\[.*?\]\]//g;' |\
      uconv -f utf-8 -t utf-8 "${icu_transform[@]}" |\
      local/callhome_normalize.pl | \
      perl -ane 'print if @F > 1; ' |\
      sed 's/\s*$//g' > $datadir/text
  done
fi

echo "CALLHOME ECA Data preparation succeeded."
