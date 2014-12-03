#!/bin/bash 
#
[ -f path.sh ] &&  . ./path.sh;

arpa_lm=data/lm/3gram-mincount/lm_unpruned.gz
input_dir=data/lang
output_dir=data/lang_test
. ./utils/parse_options.sh


mkdir -p $output_dir

[ ! -f $arpa_lm ] && echo No such file $arpa_lm && exit 1;

cp -rT ${input_dir} $output_dir

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
   utils/eps2disambig.pl | utils/s2eps.pl | fstcompile --isymbols=${output_dir}/words.txt \
     --osymbols=${output_dir}/words.txt  --keep_isymbols=false --keep_osymbols=false | \
    fstrmepsilon > ${output_dir}/G.fst
  fstisstochastic ${output_dir}/G.fst


echo  "Checking how stochastic G is (the first of these numbers should be small):"
fstisstochastic ${output_dir}/G.fst 

## Check lexicon.
## just have a look and make sure it seems sane.
echo "First few lines of lexicon FST:"
fstprint   --isymbols=${output_dir}/phones.txt --osymbols=${output_dir}/words.txt ${output_dir}/L.fst  | head

echo Performing further checks

# Checking that G.fst is determinizable.
fstdeterminize ${output_dir}/G.fst /dev/null || echo Error determinizing G.

# Checking that L_disambig.fst is determinizable.
fstdeterminize ${output_dir}/L_disambig.fst /dev/null || echo Error determinizing L.

# Checking that disambiguated lexicon times G is determinizable
# Note: we do this with fstdeterminizestar not fstdeterminize, as
# fstdeterminize was taking forever (presumbaly relates to a bug
# in this version of OpenFst that makes determinization slow for
# some case).
fsttablecompose ${output_dir}/L_disambig.fst ${output_dir}/G.fst | \
   fstdeterminizestar >/dev/null || echo Error

# Checking that LG is stochastic:
fsttablecompose ${output_dir}/L_disambig.fst ${output_dir}/G.fst | \
   fstisstochastic || echo "[log:] LG is not stochastic"


echo "$0 succeeded"

