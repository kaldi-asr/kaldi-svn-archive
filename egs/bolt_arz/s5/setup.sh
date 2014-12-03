corpora=(/export/a04/gkumar/corpora/GALE_select_ARZ
          /export/corpora/LDC/LDC2002S22/
          /export/corpora/LDC/LDC2002S37/
          /export/corpora/LDC/LDC2002T38
          /export/corpora/LDC/LDC2002T39/
          /export/corpora/LDC/LDC2014E103/
          /export/corpora/LDC/LDC2014E39
          /export/corpora/LDC/LDC2014E67
          /export/corpora/LDC/LDC2014E70
          /export/corpora/LDC/LDC2014E71
          /export/corpora/LDC/LDC2014E79
          /export/corpora/LDC/LDC2014E80
          /export/corpora/LDC/LDC2014E86
          /export/corpora/LDC/LDC96S49
          /export/corpora/LDC/LDC97S45/
          /export/corpora/LDC/LDC97T19
        )

mkdir corpora; cd corpora
for corpus in "${corpora[@]}"; do
  ln -s $corpus .
done
cd ..

git clone /export/a09/jtrmal/bolt/callhome_azr16/IBM IBM



