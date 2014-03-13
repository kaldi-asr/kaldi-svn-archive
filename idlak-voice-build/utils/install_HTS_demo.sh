#!/bin/bash
# bash script to install HTS demos
# Matthew Aylett 27/2/14



if [ 4 -ne $# ]; then
   echo "install_HTS_demo.sh <HTKID> <HTK PASSWORD> <INSTALL LOCATION> <STRAIGHT MATLAB CODE|\"NOSTRAIGHT\">"
   echo "    4 arguments expected"
   exit 1;
fi

if [ "NOSTRAIGHT" != $4 ] && [ ! -d $4 ]; then
   echo "install_HTS_demo.sh <HTKID> <HTK PASSWORD> <INSTALL LOCATION> <STRAIGHT MATLAB CODE|\"NOSTRAIGHT\">"
   echo "    specify STRAIGHT directory for MATLAB code i.e. /usr/local/STRAIGHTV40_007d or NOSTRAIGHT"
   exit 1;
fi

if [ -d $3 ]; then
    echo "$4 Demo Installing in $3"
else
    mkdir $3
    echo "$4 Demo Creating and installing in $3"
fi

HTKID=$1
HTKPSWD=$2
INSTALL_LOCATION=`readlink -f $3`
STRAIGHT=$4
PWD=`pwd`

cd $INSTALL_LOCATION
echo cd $INSTALL_LOCATION
mkdir installdir

wget http://htk.eng.cam.ac.uk/ftp/software/HTK-3.4.1.tar.gz --http-user=$HTKID --http-password=$HTKPSWD

wget http://htk.eng.cam.ac.uk/ftp/software/hdecode/HDecode-3.4.1.tar.gz --http-user=$HTKID --http-password=$HTKPSWD

wget http://hts.sp.nitech.ac.jp/archives/2.3alpha/HTS-2.3alpha_for_HTK-3.4.1.tar.bz2

## Unpack everything:
tar -zxvf HTK-3.4.1.tar.gz
tar -zxvf HDecode-3.4.1.tar.gz
tar -xvf HTS-2.3alpha_for_HTK-3.4.1.tar.bz2

## Apply HTS patch:
cd htk
patch -p1 -d . < ../HTS-2.3alpha_for_HTK-3.4.1.patch



## Finally, configure and compile:
./configure --prefix=$INSTALL_LOCATION/installdir --without-x --disable-hslab
make
make install
cd ..

# sptk

wget http://downloads.sourceforge.net/sp-tk/SPTK-3.6.tar.gz
tar xvfz  SPTK-3.6.tar.gz
cd SPTK-3.6
./configure --prefix=$INSTALL_LOCATION/installdir
make -j 4
make install
cd ..

# hts engine

wget http://downloads.sourceforge.net/hts-engine/hts_engine_API-1.07.tar.gz
tar xvfz hts_engine_API-1.07.tar.gz
cd hts_engine_API-1.07
./configure --prefix=$INSTALL_LOCATION/installdir CFLAGS="-m32 -g -O2 -Wall"
make -j 4
make install
cd ..

# speech tools
wget http://www.cstr.ed.ac.uk/downloads/festival/2.1/speech_tools-2.1-release.tar.gz
tar xfvz speech_tools-2.1-release.tar.gz
cd speech_tools
./configure
make
cd ..
# festival
wget http://www.cstr.ed.ac.uk/downloads/festival/2.1/festival-2.1-release.tar.gz
tar xvfz festival-2.1-release.tar.gz
cd festival
./configure
make
cd ..

wget http://www.cstr.ed.ac.uk/downloads/festival/2.1/festlex_CMU.tar.gz
wget http://www.cstr.ed.ac.uk/downloads/festival/2.1/festvox_cmu_us_slt_arctic_hts.tar.gz

tar xvfz festlex_CMU.tar.gz
tar xvfz festvox_cmu_us_slt_arctic_hts.tar.gz

if [ "STRAIGHT" == "$STRAIGHT" ]; then
    # get STRAIGHT
    # wget http://www.wakayama-u.ac.jp/~kawahara/puzzlet/STRAIGHTtipse/Resources/STRAIGHTV40_007d.zip
    # unzip STRAIGHTV40_007d.zip
    wget http://hts.sp.nitech.ac.jp/archives/2.3alpha/HTS-demo_CMU-ARCTIC-SLT_STRAIGHT.tar.bz2
    tar xvfj HTS-demo_CMU-ARCTIC-SLT_STRAIGHT.tar.bz2
    unzip STRAIGHTtrial.zip
    cd HTS-demo_CMU-ARCTIC-SLT_STRAIGHT 
    ./configure --with-matlab-search-path=/usr/bin \
                     --with-straight-path=$INSTALL_LOCATION/$STRAIGHT \
                     --with-fest-search-path=$INSTALL_LOCATION/festival/examples \
                     --with-sptk-search-path=$INSTALL_LOCATION/installdir/bin \
                     --with-hts-search-path=$INSTALL_LOCATION/installdir/bin \
                     --with-hts-engine-search-path=$INSTALL_LOCATION/installdir/bin
cd ..
else
    wget http://hts.sp.nitech.ac.jp/archives/2.3alpha/HTS-demo_CMU-ARCTIC-SLT.tar.bz2
    tar xvfj HTS-demo_CMU-ARCTIC-SLT.tar.bz2
    cd HTS-demo_CMU-ARCTIC-SLT
    ./configure      --with-fest-search-path=$INSTALL_LOCATION/festival/examples \
                     --with-sptk-search-path=$INSTALL_LOCATION/installdir/bin \
                     --with-hts-search-path=$INSTALL_LOCATION/installdir/bin \
                     --with-hts-engine-search-path=$INSTALL_LOCATION/installdir/bin
cd ..
fi

cd $PWD

#export LD_LBRARY_PATH=$INSTALL_LOCATION/installdir/lib:$LD_LBRARY_PATH

# Note make will first do data extraction (which takes an hour or so)
# in an active terminal window and then do training in the background
# This takes longer then a standard AFS token so will fail without 
# using longer length tokens. This will probably work if you run make 
# using a long life token.

# alternatively you could copy to scratch but you will need to
#  recompile festival which has hard coded dir paths in its scripts
# and reconfigure the demo to point to new bib directories.

# Note if you want to use the kaldi front end look at the Idlak 
# documentation before building;.


