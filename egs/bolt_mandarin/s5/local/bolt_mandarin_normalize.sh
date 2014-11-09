#!/bin/bash



f=$1   # list of files from the "input" directory
out=$2
basedir=`dirname $out`

# TODO: This should be moved to kaldi-tools directory
# Transcripts normalization and segmentation (this needs external tools).
# Download and configure word segmentation tool.
pyver=`python --version 2>&1 | sed -e 's:.*\([2-3]\.[0-9]\+\).*:\1:g'`
export PYTHONPATH=$PYTHONPATH:`pwd`/tools/mmseg-1.3.0/lib/python${pyver}/site-packages
if [ ! -d tools/mmseg-1.3.0/lib/python${pyver}/site-packages ]; then
  echo "--- Downloading mmseg-1.3.0 ..."
  echo "NOTE: it assumes that you have Python, Setuptools installed on your system!"
  wget -P tools http://pypi.python.org/packages/source/m/mmseg/mmseg-1.3.0.tar.gz
  tar xf tools/mmseg-1.3.0.tar.gz -C tools
  cd tools/mmseg-1.3.0
  mkdir -p lib/python${pyver}/site-packages
    python setup.py build
  python setup.py install --prefix=.
  cd ../..
  if [ ! -d tools/mmseg-1.3.0/lib/python${pyver}/site-packages ]; then
    echo "mmseg is not found - installation failed?"
    exit 1
  fi
fi

# TODO: The text filtering should be improved
echo -e "\n----Preparing audio training transcripts $f"
cat $f |\
  sed -e 's/Englishl/English/gi' |\
  perl -pe 's/\[distortion.\](.*?)\[.distortion\]/\1/gi' |\
  perl -pe 's/\[background.\](.*?)\[.background\]/\1/gi' |\
  perl -pe 's/\[noise.\](.*?)\[.noise\]/\1/gi' |\
  perl -pe 's/\[static.\](.*?)\[.static\]/\1/gi' |\
  sed -e 's/\(。\)/./g' |\
  sed -e 's/\([.,!?]\)/ \1 /g' |\
  sed -e 's/\[[a-zA-Z]*_noise/[noise/g' |\
  sed -e 's/{[a-zA-Z]*_noise/{noise/g' |\
  sed -e 's/(( *))/(())/g' |\
  sed -e 's/((\([^)]\{1,\}\)))/\1/g' |\
  sed '/^\s*$/d' |\
  uconv -f utf-8 -t utf-8 -x "Any-Upper" |\
  local/callhome_normalize.pl |\
  python local/callhome_mmseg_segment.py |\
  awk '{if (NF > 1) print $0;}' | sort -u > $out


  


