#!/bin/bash 
#
set -e
set -o pipefail

if [ -f path.sh ]; then . path.sh; fi

silprob=0.5
mkdir -p data/lang_test data/train


arpa_lm=data/local/lm/srilm/mixed_lm.gz 
#arpa_lm=data/local/lm/srilm/srilm.o3g.kn.gz
#arpa_lm=data/local/lm/3gram-mincount/lm_unpruned.gz

[ ! -f $arpa_lm ] && echo No such file $arpa_lm && exit 1;

# Copy stuff into its final locations...

for f in spk2utt utt2spk wav.scp text segments reco2file_and_channel; do
  cp data/local/train/$f data/train/$f;
done

# for f in spk2utt utt2spk wav.scp text segments reco2file_and_channel stm; do
cp /export/a12/xzhang/kaldi-bolt/egs/bolt/20141103/CU-BOLT-Mandarin-dev14-scoring-20141029/glms/ma970904.glm conf/glm
cp conf/glm data/train
cp conf/glm data/bolt_dev 
cp conf/glm data/bolt_tune
cp conf/glm data/bolt_test
cp /export/a12/xzhang/kaldi-bolt/egs/bolt/20141103/CU-BOLT-Mandarin-dev14-scoring-20141029/stms/dev-test.stm data/bolt_test/stm 
cp /export/a12/xzhang/kaldi-bolt/egs/bolt/20141103/CU-BOLT-Mandarin-dev14-scoring-20141029/stms/dev-tune.stm data/bolt_tune/stm
cp /export/a12/xzhang/kaldi-bolt/egs/bolt/20141103/CU-BOLT-Mandarin-dev14-scoring-20141029/stms/dev-dev.stm data/bolt_dev/stm 
rm -r data/lang_test
cp -r data/lang data/lang_test

# grep -v '<s> <s>' etc. is only for future-proofing this script.  Our
# LM doesn't have these "invalid combinations".  These can cause 
# determinization failures of CLG [ends up being epsilon cycles].
# Note: remove_oovs.pl takes a list of words in the LM that aren't in
# our word list.  Since our LM doesn't have any, we just give it
# /dev/null [we leave it in the script to show how you'd do it].
gunzip -c "$arpa_lm" | \
   grep -v '<s> <s>' | \
   grep -v '</s> <s>' | \
   grep -v '</s> </s>' | \
   arpa2fst - | fstprint | \
   utils/remove_oovs.pl /dev/null | \
   utils/eps2disambig.pl | utils/s2eps.pl | fstcompile --isymbols=data/lang_test/words.txt \
     --osymbols=data/lang_test/words.txt  --keep_isymbols=false --keep_osymbols=false | \
    fstrmepsilon > data/lang_test/G.fst


echo  "Checking how stochastic G is (the first of these numbers should be small):"
fstisstochastic data/lang_test/G.fst || true

## Check lexicon.
## just have a look and make sure it seems sane.
echo "First few lines of lexicon FST:"
(fstprint   --isymbols=data/lang/phones.txt --osymbols=data/lang/words.txt data/lang/L.fst  | head ) || true

echo Performing further checks

# Checking that G.fst is determinizable.
fstdeterminize data/lang_test/G.fst /dev/null || echo Error determinizing G.

# Checking that L_disambig.fst is determinizable.
fstdeterminize data/lang_test/L_disambig.fst /dev/null || echo Error determinizing L.

# Checking that disambiguated lexicon times G is determinizable
# Note: we do this with fstdeterminizestar not fstdeterminize, as
# fstdeterminize was taking forever (presumbaly relates to a bug
# in this version of OpenFst that makes determinization slow for
# some case).
fsttablecompose data/lang_test/L_disambig.fst data/lang_test/G.fst | \
   fstdeterminizestar >/dev/null || echo Error

# Checking that LG is stochastic:
fsttablecompose data/lang/L_disambig.fst data/lang_test/G.fst | \
   fstisstochastic || echo LG is not stochastic


echo callhome_format_data succeeded.

