#!/usr/bin/perl
# Copyright 2014  Johns Hopkins University (author: Jan Trmal)\
#                 Hainan Xu, Xiaohui Zhang

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.


# Converts the CALLHOME corpus into kaldi data directory.
# Tries to be smart about preventing upper-case/lower-case issues
# (which might not be really smart)


use utf8;
use warnings;
use strict;

use PerlIO::encoding;
use Encode qw(:fallbacks decode encode );

binmode STDIN, ":utf8";
binmode STDOUT, ":utf8";
binmode STDERR, ":utf8";

#$PerlIO::encoding::fallback = sub{ sprintf "<U+%04X>", shift };

if ( @ARGV != 2 ) {
  print STDERR "The script reads the paths of the txt files (transcriptions) \n";
  print STDERR "from the STDIN and converts these files into kaldi-compatible directory \n\n\n";
  print STDERR "Called with unsupported numbers of arguments\n\n";
  print STDERR "callhome_data_convert.pl <SPH_FILES_LIST> <TARGET_KALDI_DIR>\n";
  print STDERR "\nwhere \n";
  print STDERR "SPH_FILES_LIST is a list of the audio files including the correct(abs) path\n";
  print STDERR "               The list is needed to create proper reference to the audio\n";
  print STDERR "TARGET_KALDI_DIR is the directory where all kaldi data files will be created \n";
  print STDERR "                 directory must exist already!\n";
}

my $sph_list=$ARGV[0];
my $dest_dir=$ARGV[1];
my %UTT2SPH;


open(my $sph, "<$sph_list")
  or die "Could not open the file $sph_list: $_\n";

while(my $line=<$sph>) {
  chomp $line;
  my $rec_id=`basename $line`;
  chomp $rec_id;
  $rec_id=~ s/\.sph//i;
  $rec_id=uc($rec_id);

  if (exists $UTT2SPH{$rec_id} ) {
    print STDERR "The utterance id $rec_id maps to multiple audio files:\n";
    print STDERR ">>file A: $line\n";
    print STDERR ">>file B: $UTT2SPH{$rec_id}\n";
    die "Will not continue...";
  }
  $UTT2SPH{$rec_id}=$line;
}
close($sph);

open(my $wav, ">$dest_dir/wav.scp")
  or die "Could not open the file $dest_dir/wav.scp: $_\n";
open(my $segments, ">$dest_dir/segments")
  or die "Could not open the file $dest_dir/segments: $_\n";
open(my $text, ">:encoding(utf8)", "$dest_dir/transcripts.txt")
  or die "Could not open the file $dest_dir/transcripts.txt: $_\n";
open(my $utt2spk, ">$dest_dir/utt2spk")
  or die "Could not open the file $dest_dir/utt2spk: $_\n";
open(my $reco, ">$dest_dir/reco2file_and_channel")
  or die "Could not open the file $dest_dir/reco2file_and_channel: $_\n";

my $sph2pipe=`which sph2pipe`
  or die "Could not find the sph2pipe binary on PATH: $_";
chomp $sph2pipe;

while (my $filename=<STDIN>) {
  print $filename;
  chomp $filename;
  my $rec_id=`basename $filename`;
  chomp $rec_id;
  $rec_id=~ s/\.txt//i;
  $rec_id=uc($rec_id);

  die "Could not remap the utterance id $rec_id into audio recording filename\n"
    unless exists $UTT2SPH{$rec_id};
  print $wav "$rec_id-A $sph2pipe -f wav -p -c 1 " . $UTT2SPH{$rec_id} . "|\n";
  print $wav "$rec_id-B $sph2pipe -f wav -p -c 2 " . $UTT2SPH{$rec_id} . "|\n";

  print $reco "$rec_id-A ", lc($rec_id), " 1\n";
  print $reco "$rec_id-B ", lc($rec_id), " 2\n";

  #open(my $fh, "cat $filename | uconv -f GB18030 -t utf-8 --callback substitute|")
  open(my $fh, $filename)
    or die "Could not open the file $filename: $_\n";

  while (my $line=<$fh>) {
    $line= decode("GBK", $line,  sub{ sprintf "<U+%04X>", shift });
    #$line= decode("GBK", $line,  sub{ sprintf "<U+FFFD>", shift });
    if ($line =~ /\<U\+[a-fA-f0-9]+\>/i ) {
      print "Skipping line $line";
      next;
    }
    chomp $line;
    next if $line =~ /^#.*/;
    $line =~ s/^\s+|\s+$//g;
    my @A=split(" ",$line);
    if (@A <= 3) {
      print STDERR "Cannot parse line \"$line\" in $filename\n" if $line;
      next;
    } else {
      my $spk_id=$A[2];
      $spk_id=~ s/^([AB]\d*).*/$1/;

      my $chan_id=$spk_id;
      $chan_id=~s/^([AB]).*/$1/;

      my $utt_start=$A[0];
      my $utt_end=$A[1];

      next if (($utt_end - $utt_start) * 100 le 1);

      my $utt_id=sprintf "%s-%s-%06.0f-%06.0f", $rec_id, $spk_id, 100*$utt_start, 100*$utt_end;
      print $text $utt_id;
      for(my $n = 3; $n < @A; $n++) { print $text " $A[$n]" };
      print $text "\n";

      print $utt2spk "$utt_id $rec_id-$spk_id\n";
      print $segments "$utt_id $rec_id-$chan_id $utt_start $utt_end\n";


    }
  }
  close($fh);
}

close($wav);
close($segments);
close($text);
close($utt2spk);
close($reco);

