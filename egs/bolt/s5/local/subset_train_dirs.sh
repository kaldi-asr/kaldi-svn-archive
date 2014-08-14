 #!/bin/bash

 numutt=`cat data/train/feats.scp | wc -l`;
utils/subset_data_dir.sh data/train  5000 data/train_sub1
if [ $numutt -gt 10000 ] ; then
  utils/subset_data_dir.sh data/train 10000 data/train_sub2
else
  (cd data; ln -s train train_sub2 )
fi
if [ $numutt -gt 20000 ] ; then
  utils/subset_data_dir.sh data/train 20000 data/train_sub3
else
  (cd data; ln -s train train_sub3 )
fi

