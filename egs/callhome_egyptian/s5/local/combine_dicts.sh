#!/usr/bin/env bash

stage=0

##END OF OPTIONS

. utils/parse_options.sh

[ -f path.sh ]  && . ./path.sh
[ -f cmd.sh ] && . ./cmd.sh

set -e 
set -o pipefail

#First get the list of unique words from our text file
if [ $# -lt 3 ]; then
  echo 'Usage callhome_prepare_dict.sh lexicon'
  exit 1;
fi

args=( "$@" )
input=${args[0]};
output=${args[${#args[@]}-1]}
unset args[0];
unset args[${#args[@]}]

echo "Args: ${args[@]}"
echo "Input: $input"
echo "Output: $output"

mkdir -p $output
cp $input/optional_silence.txt $output
cp $input/silence_phones.txt $output

echo "Merging file extra_questions.txt"
for dir in "$input" "${args[@]}"; do
  echo $dir >&2 
  [ -f $dir/extra_questions.txt ] && cat $dir/extra_questions.txt 
done > $output/extra_questions.txt

for file in nonsilence_phones.txt lexicon.txt ; do
  echo "Merging the the file $file "
  for dir in "$input" "${args[@]}"; do
    echo $dir >&2 
    [ ! -f $dir/$file ] && echo "File $file must be present in $dir!" >&2 && exit 1
    cat $dir/$file
  done | sort -u > $output/$file
done


#Generate questions and merge with already existing questions. We do it on-the-fly
cat $output/nonsilence_phones.txt |\
  perl -e '
    while ($line = <>) {
      chomp $line;
      $tag = $line;
      $tag =~ s/.*?_//;
      $tag = "" if $tag eq $line;
      push @{$tags{$tag}}, $line;
    }
    foreach $tag(keys %tags) {
      $question=join(" ", @{$tags{$tag}} );
      print $question . "\n";
    }
  ' | cat - $output/extra_questions.txt | sort -u > $output/extra_questions.txt.s
mv $output/extra_questions.txt.s $output/extra_questions.txt
    
utils/validate_dict_dir.pl $output

