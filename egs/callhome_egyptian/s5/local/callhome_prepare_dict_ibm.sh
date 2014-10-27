#!/usr/bin/env bash

stage=0
dictdir=data/local/dict
sourcedir=data/local/train
. utils/parse_options.sh

[ -f path.sh ]  && . ./path.sh
[ -f cmd.sh ]  && . ./cmd.sh

set -e 
set -o pipefail

mkdir -p $dictdir
#Lets remove the diphtongs... Seems they are actually harming the performance
cat conf/phones.txt | awk '{print $1}' | grep -e "^$" -v | \
  grep -v "^\s*aw" | \
  grep -v "^\s*ay"  > $dictdir/phones

if false; then
grep -v "NEEE" | perl -ane ' 
    next if @F != 2;

    @prons= split "//", $F[1]; 
    @words= split / /, $F[0];
  
    foreach $pron(@prons) {
      
      print "$F[0]\t$pron\n" if @words eq 1;
  }' > $dictdir/lexicon.full

(
  cut -f 2- $sourcedir/text | sed 's/ /\n/g' 
  #cut -f 1  $dictdir/lexicon.full 
)  | sort -u > $dictdir/words.txt

#Convert the pronunciation field into a sequence of phones separated with spaces
#Skip all words that are not to be found in the "training" text
cat $dictdir/lexicon.full | \
  perl -e '
    use Data::Dumper;
    binmode STDIN, "utf-8";
    binmode STDOUT, "utf-8";

    open(PHONES, ${ARGV[0]}) or die "Could not open file ${ARGV[0]}: $!\n";
    while ($line=<PHONES>) {
      chomp $line;
      $line =~ s/^\s+|\s+$//g;
      next if not $line;

      @F = split /\s+/, $line;
      $phonemap{$F[0]} = 1;
      $l=length($F[0]);
      push @{$lengths{$l}}, $F[0];
    }
    @list = sort { $b <=> $a } keys(%lengths);
    $max = $list[0];
    print STDERR "Performing phoneme decomposition, max phoneme length: $max\n";

    open(WORDS, ${ARGV[1]}) or die "Could not open file ${ARGV[1]}: $!\n";
    while ($line=<WORDS>) {
      chomp $line;
      next if not $line;
      $words{$line}=0;
    }

    while($line = <STDIN>) {
      chomp $line;
      @F = split /\s+/, $line;
      die "Unknown format for line $line" if @F != 2;
      $word = @F[0];
      die "Cannot process non-blackwalter transcript: $line!" unless $word =~ m/.*_bw/; 
      $pron = @F[1];
      next unless exists $words{$word};
      $orig_pron=$pron;

      @out_pron=qw();
      while ( $pron ) {
        $found = 0;
        foreach $len(@list) {
          #print STDERR "Current length: $len\n";
          #print STDERR "Current candidates: " .  Dumper($lengths{$len});
          $subseq=substr $pron, 0, $len;
          if ( /\Q$subseq/ ~~ @{$lengths{$len}} ) {
            $pron = substr $pron, $len;
            $found = 1;
            push @out_pron, $subseq;
            last;
          }
        }
        if (not $found) {
          die "Unknown phoneme decomposition (unknown phoneme): $pron in $word ->  $orig_pron\n";
        }
      }
      $pron=join(" ", @out_pron);
      print "$word\t$pron\n";

    }
  ' $dictdir/phones  $dictdir/words.txt | sort -u > $dictdir/lexicon-arz.txt
#  ' $dictdir/phones <(cat $dictdir/lexicon.full | awk '{print $1}') | sort -u > $dictdir/lexicon
#  ' $dictdir/phones  $dictdir/words.txt | sort -u > $dictdir/lexicon

(set +e; set +o pipefail;  diff $dictdir/words.txt <(cat $dictdir/lexicon-arz.txt | awk '{print $1}') | \
  grep "^<" | awk '{print $2}' | grep -v '<.*>' | sed '/^\s*$/d' > $dictdir/oov.txt; true) || true


grep "^.*_en$" $dictdir/oov.txt > $dictdir/words-en-oov.txt
grep "^.*_bw$" $dictdir/oov.txt | grep -v "_en_bw$" > $dictdir/words-arz-oov.txt

if true; then
##
##This is for the english OOV processing
##
if [ ! -f exp/g2p/en/model.english ]; then
  mkdir -p exp/g2p/en;
  echo "--- Downloading a pre-trained Sequitur G2P model ..."
  wget http://sourceforge.net/projects/kaldi/files/sequitur-model4 -O exp/g2p/en/model.english
  if [ ! -f exp/g2p/en/model.english ]; then
    echo "Failed to download the g2p model!"
    exit 1
  fi
fi

cat $dictdir/words-en-oov.txt | sed 's/_en$//g' > $dictdir/words-en-oov.txt.notags
local/apply_g2p.sh --icu-transform ""  --model exp/g2p/en/model.english \
  --with-probs false --output-lex $dictdir/vocab-en-oov.lex.notags \
  $dictdir/words-en-oov.txt.notags exp/g2p/en exp/g2p/en
cat $dictdir/vocab-en-oov.lex.notags | \
  perl -ane 'print "$F[0]_en\t"; print join(" ", @F[1..$#F]) . "\n"; ' > $dictdir/vocab-en-oov.lex
cat $dictdir/vocab-en-oov.lex | \
  perl -e '
    use strict;
    use warnings;

    use Data::Dumper;
    open(MAP, $ARGV[0]) or die "Cannot open the char map: $!";
    my %MAPPING;
    while (my $line=<MAP>) {
      chomp $line;
      (my $phone, my $mappings) = split " ", $line, 2;
      #print STDERR $phone . "\n";
      #print STDERR $mappings . "\n";
      my @mapseq = split /,/, $mappings;
      #print STDERR Dumper(\@mapseq);
      push @{$MAPPING{$phone}}, @mapseq;
      #print STDERR Dumper(\%MAPPING);
    }
    close(MAP);
    #print STDERR Dumper(\%MAPPING);
    while ( my $line=<STDIN> ) {
      #print STDERR $line;
      chomp $line;
      (my $phone, my $pron_str) = split " ", $line, 2;
      my @pron = split " ", $pron_str;
      my @out_prons = ("") ;
      foreach $phone (@pron) {
        #print STDERR Dumper(\@out_prons, \@pron, $pron_str);
        my @replacements = @{$MAPPING{$phone}};
        my @tmp;
        foreach my $repl (@replacements) {
          foreach my $pos (@out_prons) {
            push @tmp, "$pos $repl";
          }
        }
        @out_prons= @tmp;
        @tmp = ();
      }
      #print STDERR Dumper(\@out_prons);

      foreach  my $pron_var (@out_prons) {
        my $s = "$phone\t$pron_var";
        $s =~ s/^\s+|\s+$//g;
        print $s . "\n";
      }
    }

  ' conf/cmu2arz.map | sort -u > $dictdir/lexicon-en-oov.txt
##END of ENGLISH OOV processing
fi

##START of ARABIC OOV processing
cat $dictdir/lexicon-arz.txt | sed 's/_bw//g' > $dictdir/lexicon-arz.txt.notags
cat $dictdir/words-arz-oov.txt | sed 's/_bw$//g' > $dictdir/words-arz-oov.txt.notags
local/train_g2p.sh --icu-transform "" --cmd "$decode_cmd" --lexicon $dictdir/lexicon-arz.txt.notags\
  data/local/dict/ exp/g2p/arz
local/apply_g2p.sh --cmd "$decode_cmd"  --icu-transform "" --with-probs false \
  --output-lex $dictdir/lexicon-arz-oov.txt.notags \
  $dictdir/words-arz-oov.txt.notags exp/g2p/arz exp/g2p/arz
fi
cat $dictdir/lexicon-arz-oov.txt.notags | \
  perl -ane 'print "$F[0]_bw\t"; print join(" ", @F[1..$#F]) . "\n"; ' > $dictdir/lexicon-arz-oov.txt
##END of ARABIC OOV processing
cat $dictdir/lexicon-arz.txt $dictdir/lexicon-arz-oov.txt $dictdir/lexicon-en-oov.txt | \
  sort -u | sed '/^\s*$/d' >  $dictdir/lexicon

#Regenerate the phones file once more:
mv $dictdir/phones $dictdir/phones.full
cat $dictdir/lexicon | grep -v "<.*>" | cut -f 2- | sed 's/ /\n/g' | sort -u > $dictdir/phones.used

awk '{print $1}' $dictdir/phones.used | sort -u | grep -v "^\s*$" > $dictdir/nonsilence_phones.txt


# silence phones, one per line. 
for w in sil laughter "v-noise" noise unk hes; do 
  echo "<$w>"
done > $dictdir/silence_phones.txt
echo "<sil>" > $dictdir/optional_silence.txt

for w in `cat $dictdir/silence_phones.txt | grep -v "<sil>"`; do
  perl -e 'print uc($ARGV[0]) . "\t" . $ARGV[0] . "\n";' $w
done | cat - $dictdir/lexicon   > $dictdir/lexicon.txt

# An extra question will be added by including the silence phones in one class.
cat $dictdir/silence_phones.txt | paste -s  > $dictdir/extra_questions.txt || exit 1;
cat $dictdir/nonsilence_phones.txt |\
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
    ' >> $dictdir/extra_questions.txt

utils/validate_dict_dir.pl $dictdir
exit 0;

cat $datadir/text  | \
awk '{for (n=2;n<=NF;n++){ count[$n]++; } } END { for(n in count) { print count[n], n; }}' | \
sort -nr > $tmpdir/word_counts

awk '{print $1}' $dir/lexicon.txt | \
perl -e '($word_counts)=@ARGV;
open(W, "<$word_counts")||die "opening word-counts $word_counts";             
while(<STDIN>) { chop; $seen{$_}=1; }
while(<W>) {
 ($c,$w) = split;
 if (!defined $seen{$w}) { print; }                                          
} ' $tmpdir/word_counts > $tmpdir/oov_counts.txt                                    
                                                                             
echo "*Highest-count OOVs are:"                                                  
head -n 20 $tmpdir/oov_counts.txt 


