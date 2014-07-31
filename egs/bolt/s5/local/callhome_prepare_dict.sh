#!/bin/bash
# prepare dictionary for CALLHOME
# it is done for English and Chinese separately, 
# For English, we use CMU dictionary, and Sequitur G2P
# for OOVs, while all englist phone set will concert to Chinese
# phone set at the end. For Chinese, we use an online dictionary,
# for OOV, we just produce pronunciation using Charactrt Mapping.
  
. path.sh

set -e
set -o pipefail

[ $# != 0 ] && echo "Usage: local/callhome_prepare_dict.sh" && exit 1;

train_dir=data/local/train.callhome
dev_dir=data/local/devtest.callhome
dict_dir=data/local/dict
mkdir -p $dict_dir


# extract full vocabulary
cat $train_dir/text $dev_dir/text | cut -f 2- -d ' '|\
  sed -e 's/ /\n/g' | sort -u | grep -v "<.*>"  > $dict_dir/vocab-full.txt  

# prepare the non-speech phones"
(
echo "<SIL> <sil>";
echo "<UNK> <unk>";
cat $train_dir/text $dev_dir/text | cut -f 2- -d ' '|\
  sed -e 's/ /\n/g' | sort -u | grep  "<.*>" | grep -v "UNK|SIL" |\
  perl -ne '
    chomp;
    $word=$_;
    $line=$word;
    #$line=~ s/^<(.*)>$/$1/;
    $line=~ s/-//g;
    $line=lc($line);
    print "$word $line\n"
  ') > $dict_dir/lexicon_extra.txt


# split into English and Chinese
cat $dict_dir/vocab-full.txt | grep '[a-zA-Z]' > $dict_dir/vocab-en.txt
cat $dict_dir/vocab-full.txt | grep -v '[a-zA-Z]' > $dict_dir/vocab-ch.txt


# produce pronunciations for english 
if [ ! -f $dict_dir/cmudict/cmudict.0.7a ]; then
  echo "--- Downloading CMU dictionary ..."
  svn co https://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict \
    $dict_dir/cmudict || exit 1;
fi

echo "--- Striping stress and pronunciation variant markers from cmudict ..."
perl $dict_dir/cmudict/scripts/make_baseform.pl \
  $dict_dir/cmudict/cmudict.0.7a /dev/stdout |\
  sed -e 's:^\([^\s(]\+\)([0-9]\+)\(\s\+\)\(.*\):\1\2\3:' > $dict_dir/cmudict-plain.txt

echo "--- Searching for English OOV words ..."
gawk 'NR==FNR{words[$1]; next;} !($1 in words)' \
  $dict_dir/cmudict-plain.txt $dict_dir/vocab-en.txt |\
  egrep -v '<.?s>' > $dict_dir/vocab-en-oov.txt

gawk 'NR==FNR{words[$1]; next;} ($1 in words)' \
  $dict_dir/vocab-en.txt $dict_dir/cmudict-plain.txt |\
  egrep -v '<.?s>' > $dict_dir/lexicon-en-iv.txt

wc -l $dict_dir/vocab-en-oov.txt
wc -l $dict_dir/lexicon-en-iv.txt


if [ ! -f exp/g2p/model.english ]; then
  mkdir -p exp/g2p;
  echo "--- Downloading a pre-trained Sequitur G2P model ..."
  wget http://sourceforge.net/projects/kaldi/files/sequitur-model4 -O exp/g2p/model.english
  if [ ! -f exp/g2p/model.english ]; then
    echo "Failed to download the g2p model!"
    exit 1
  fi
fi

local/apply_g2p.sh --icu-transform ""  --model exp/g2p/model.english \
  --with-probs false --output-lex $dict_dir/vocab-en-oov.lex \
  $dict_dir/vocab-en-oov.txt exp/g2p/ exp/g2p

cat  $dict_dir/vocab-en-oov.lex  $dict_dir/lexicon-en-iv.txt |\
  sort > $dict_dir/lexicon-en-phn.txt




# produce pronunciations for chinese 
if [ ! -f $dict_dir/cedict_1_0_ts_utf-8_mdbg.txt ]; then
  wget -P $dict_dir http://www.mdbg.net/chindict/export/cedict/cedict_1_0_ts_utf-8_mdbg.txt.gz 
  gunzip $dict_dir/cedict_1_0_ts_utf-8_mdbg.txt.gz
fi

cat $dict_dir/cedict_1_0_ts_utf-8_mdbg.txt | grep -v '#' | awk -F '/' '{print $1}' |\
 perl -e '   
  while (<STDIN>) {
    @A = split(" ", $_);
    print $A[1];
    for($n = 2; $n < @A; $n++) {
      $A[$n] =~ s:\[?([a-zA-Z0-9\:]+)\]?:$1:; 
      $tmp = uc($A[$n]); 
      print " $tmp";
    }
    print "\n";
  }
 ' | sort -k1 > $dict_dir/ch-dict.txt 


echo "--- Searching for Chinese OOV words ..."
gawk 'NR==FNR{words[$1]; next;} !($1 in words)' \
  $dict_dir/ch-dict.txt $dict_dir/vocab-ch.txt |\
  egrep -v '<.?s>' > $dict_dir/vocab-ch-oov.txt

gawk 'NR==FNR{words[$1]; next;} ($1 in words)' \
  $dict_dir/vocab-ch.txt $dict_dir/ch-dict.txt |\
  egrep -v '<.?s>' > $dict_dir/lexicon-ch-iv.txt

echo "---Word counts (Mandarin) ---"
wc -l $dict_dir/vocab-ch-*.txt


# first make sure number of characters and pinyins 
# are equal  
cat $dict_dir/vocab-ch-oov.txt |\
  perl -e '
  use utf8;
  binmode(STDOUT, ":utf8");
  binmode(STDERR, ":utf8");
  binmode(STDIN, ":utf8");
  use Data::Dumper;

  $dictname = $ARGV[0];
  open($dict, "<:utf8", "$dictname");
  while ($line=<$dict>) {
    @A = split(" ", $line);
    $word_len = length($A[0]);
    $pron_len = @A - 1 ; 

    next if ($word_len != $pron_len);

    for ($q = 0; $q < $word_len; $q+=1) {
      $ch=substr( $A[0], $q , 1 );
      $p=$A[$q+1];
      
      $vocab{$ch}{$p}+=1;
    }
  }
  close($dict);

  LINE: while ($line=<STDIN>) {
    chomp $line;
    #print STDERR $line . "\n";
    $word_len= length($line);
    
    @variants = ();
    push @variants, $line;
    @tmp = ();
    for ($i = 0; $i < $word_len; $i+= 1) {
      $ch=substr($line, $i, 1);
      if ( not exists $vocab{$ch} ) {
        print STDERR "Could not translate OOV $line: $ch has unknow pinyin.\n";
        next LINE;
      }
      @pinyins= keys $vocab{$ch};
      foreach $fragment(@variants) {
        foreach $p(@pinyins) {
          push @tmp, "$fragment $p"
        }
      }
      @variants = @tmp;
      @tmp = ();
      #print STDERR Dumper(\@variants);
    }

    foreach $pronunciation(@variants) {
      print $pronunciation . "\n";
    }

  }
  ' $dict_dir/ch-dict.txt | sort -u > $dict_dir/lexicon-ch-oov.txt


cat $dict_dir/lexicon-ch-oov.txt $dict_dir/lexicon-ch-iv.txt |\
  awk '{if (NF > 1) print $0;}' > $dict_dir/lexicon-ch.txt 


cat $dict_dir/lexicon-ch.txt | \
  sed -e 's/U:/V/g' | \
  sed -e 's/ R\([0-9]\)/ ER\1/g'|\
  sed -e 's/.* M2$//g' |\
  sed '/^\s*$/d' |\
  utils/pinyin_map.pl conf/pinyin2cmu > $dict_dir/lexicon-ch-cmu.txt


cat conf/cmu2pinyin | awk '{print $1;}' | sort -u > $dict_dir/cmu 
cat conf/pinyin2cmu | awk -v cmu=$dict_dir/cmu \
  'BEGIN{while((getline<cmu)) dict[$1] = 1;}
   {for (i = 2; i <=NF; i++) if (dict[$i]) print $i;}' | sort -u > $dict_dir/cmu-used
cat $dict_dir/cmu | awk -v cmu=$dict_dir/cmu-used \
  'BEGIN{while((getline<cmu)) dict[$1] = 1;}
   {if (!dict[$1]) print $1;}' > $dict_dir/cmu-not-used 

gawk 'NR==FNR{words[$1]; next;} ($1 in words)' \
  $dict_dir/cmu-not-used conf/cmu2pinyin |\
  egrep -v '<.?s>' > $dict_dir/cmu-py

cat $dict_dir/cmu-py | \
  perl -e '
  open(MAPS, $ARGV[0]) or die("could not open map file");
  my %py2ph;
  foreach $line (<MAPS>) {
    @A = split(" ", $line);
    $py = shift(@A);
    $py2ph{$py} = [@A];
  }
  my @entry;
  while (<STDIN>) {
    @A = split(" ", $_);
    @entry = ();
    $W = shift(@A);
    push(@entry, $W);
    for($i = 0; $i < @A; $i++) { push(@entry, @{$py2ph{$A[$i]}}); }
    print "@entry";
    print "\n";  
  }  
' conf/pinyin2cmu > $dict_dir/cmu-cmu 

cat $dict_dir/lexicon-en-phn.txt | \
  perl -e '
  open(MAPS, $ARGV[0]) or die("could not open map file");
  my %py2ph;
  foreach $line (<MAPS>) {
    @A = split(" ", $line);
    $py = shift(@A);
    $py2ph{$py} = [@A];
  }
  my @entry;
  while (<STDIN>) {
    @A = split(" ", $_);
    @entry = ();
    $W = shift(@A);
    push(@entry, $W);
    for($i = 0; $i < @A; $i++) { 
      if (exists $py2ph{$A[$i]}) { push(@entry, @{$py2ph{$A[$i]}}); }
      else {push(@entry, $A[$i])};
    }
    print "@entry";
    print "\n";  
  }
' $dict_dir/cmu-cmu > $dict_dir/lexicon-en.txt 

cat $dict_dir/lexicon-en.txt $dict_dir/lexicon-ch-cmu.txt |\
  sort -u > $dict_dir/lexicon1.txt

cat $dict_dir/lexicon1.txt | awk '{ for(n=2;n<=NF;n++){ phones[$n] = 1; }} END{for (p in phones) print p;}'| \
  sort -u |\
  perl -e '
  my %ph_cl;
  while (<STDIN>) {
    $phone = $_;
    chomp($phone);
    chomp($_);      
    $phone =~ s:([A-Z]+)[0-9]:$1:; 
    if (exists $ph_cl{$phone}) { push(@{$ph_cl{$phone}}, $_)  }
    else { $ph_cl{$phone} = [$_]; }
  }
  foreach $key ( keys %ph_cl ) {
     print "@{ $ph_cl{$key} }\n"
  }
  ' | sort -k1 > $dict_dir/nonsilence_phones.txt  || exit 1;

# extract silence-phones

cut -f 2 -d ' '  $dict_dir/lexicon_extra.txt |sed 's/ /\n/g' | sort -u > $dict_dir/silence_phones.txt

echo "<sil>" > $dict_dir/optional_silence.txt

# No "extra questions" in the input to this setup, as we don't
# have stress or tone

#echo -n > $dict_dir/extra_questions.txt
cat $dict_dir/silence_phones.txt| awk '{printf("%s ", $1);} END{printf "\n";}' > $dict_dir/extra_questions.txt || exit 1;
cat $dict_dir/nonsilence_phones.txt | perl -e 'while(<>){$line=$_; foreach $p (split(" ", $_)) {
  $p =~ m:^([^\d]+)(\d*)$: || die "Bad phone $_ in \"$line\""; $q{$2} .= "$p "; } } foreach $l (values %q) {print "$l\n";}' \
 >> $dict_dir/extra_questions.txt || exit 1;


cat $dict_dir/lexicon_extra.txt  $dict_dir/lexicon1.txt | sort -u > $dict_dir/lexicon.txt || exit 1;


echo "Done preparing the CALLHOME Mandarin dictionary..."
