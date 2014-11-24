#!/usr/bin/perl
# Copyright 2014  Johns Hopkins University (author: Jan Trmal<jtrmal@gmail.com>)


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


# Converts the BOLT XML annotations into kaldi data directory.
use utf8;
use warnings;
use strict;
use XML::Parser;

use PerlIO::encoding;
use Encode qw(:fallbacks decode encode );
use Data::Dumper;

binmode STDIN, ":utf8";
binmode STDOUT, ":utf8";
binmode STDERR, ":utf8";

my %PARAMS;
my $CONVERSTATIONID;
my $SEQ=0;
my $ERR=0;
my $REC_ID;
my $UTT_ID;

my $TEXT;
my $UTT2SPK;
my $SEGMENTS;
my $total_seconds=0.0;

sub conversation {
  $CONVERSTATIONID=$_[3];
  #print Dumper(\@_);
}

sub su {
  my $state=$_[0];
  my @args=@_[1 .. $#_];

  my $i = 1;
  while ($i < $#args) {
    $PARAMS{ $args[$i] } = $args[$i+1];
    $i += 2;
  }
  $PARAMS{"text"} = "";

  #print Dumper(\%PARAMS);
}

sub body {
  my @args=@_[1 .. $#_];
  $PARAMS{"body_context"} = 1;
}
sub body_ {
  my $state=$_[0];
  my @args=@_[1 .. $#_];
  delete $PARAMS{"body_context"};
  printout();
  #print Dumper(\%PARAMS);
  %PARAMS=();
}

sub foreign() {
  my $state=$_[0];
  my @args=@_[1 .. $#_];
  $args[2]=~ s/^\s+|\s+$//g;
  $PARAMS{"foreign_context"} = $args[2];
  #print Dumper(\@args);
}

sub foreign_() {
  my $state=$_[0];
  my @args=@_[1 .. $#_];
  undef $PARAMS{"foreign_context"};
}

sub add_text {
  my $chars = shift;
  if ( $PARAMS{"foreign_context"} ) {
    $PARAMS{"text"} .= " <$PARAMS{'foreign_context'} $chars>";
  } elsif ( $PARAMS{"body_context"} ) {
    $PARAMS{"text"} .= " $chars";
  }
  #print Dumper(\%PARAMS);
  #print Dumper($chars);
}

sub char_handler
{
  my @args=@_[1 .. $#_];
  my $chars=join(' ', @args);
  $chars =~ s/^\s+|\s+$//g;
  add_text ($chars) if $chars;
  #print Dumper($chars) if $chars;
}


sub printout {
  my $begin=sprintf("%.2f",$PARAMS{'begin'}); 
  my $end=sprintf("%.2f",$PARAMS{'end'}); 
  my $id=$PARAMS{'id'};
  my $utterance=$PARAMS{'text'};

  return if (($end - $begin) * 100 le 1);

  $utterance=~ s/  */ /g;
  $utterance=~ s/^\s+|\s+$//g;

  my $spk_id=$PARAMS{'speaker'};
  $spk_id=~ s/^([AB]\d*).*/$1/;

  my $chan_id=$spk_id;
  $chan_id=~s/^([AB]).*/$1/;

  my $utt_id=sprintf "%s-%s-%06.0f-%06.0f", $REC_ID, $spk_id, 100*$begin, 100*$end;
  print $TEXT "$utt_id ";
  print $TEXT "$utterance\n";

  if ( ! $PARAMS{'text'} ) {
    print STDERR "Empty text for conversation $CONVERSTATIONID-$id\n";
    $ERR+=1;
  }
  if (($id gt 1) && ($id ne  ($SEQ + 1 ))) {
    print STDERR "Non-sequentials IDs: current $id vs previous $SEQ!\n";
    $ERR+=1;
  }
  $SEQ=$id;
  
  $total_seconds+=$end - $begin;
  print $UTT2SPK "$utt_id $REC_ID-$spk_id\n";
  print $SEGMENTS "$utt_id $REC_ID-$chan_id $begin $end\n";
}

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
  die;
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
  $rec_id=~ s/\.flac//i;
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
#print Dumper(\%UTT2SPH);

open(my $wav, ">$dest_dir/wav.scp")
  or die "Could not open the file $dest_dir/wav.scp: $_\n";
open($SEGMENTS, ">$dest_dir/segments")
  or die "Could not open the file $dest_dir/segments: $_\n";
open($TEXT, ">:encoding(utf8)", "$dest_dir/transcripts.txt")
  or die "Could not open the file $dest_dir/transcripts.txt: $_\n";
open($UTT2SPK, ">$dest_dir/utt2spk")
  or die "Could not open the file $dest_dir/utt2spk: $_\n";
open(my $reco, ">$dest_dir/reco2file_and_channel")
  or die "Could not open the file $dest_dir/reco2file_and_channel: $_\n";

my $sph2pipe=`which sph2pipe`
  or die "Could not find the sph2pipe binary on PATH: $_";
chomp $sph2pipe;

my $sox=`which sox`
  or die "Could not find the sox binary on PATH: $!";
chomp $sox;

while (my $filename=<STDIN>) {
  print $filename;
  chomp $filename;

  my $rec_id=`basename $filename`;
  chomp $rec_id;
  $rec_id=~ s/\..*$//i;
  $rec_id=uc($rec_id);
  $REC_ID=$rec_id;
  
  die "Could not remap the utterance id x${rec_id}x into audio recording filename\n"
    unless exists $UTT2SPH{$rec_id};
  my $audio=$UTT2SPH{$rec_id};

  if ( $audio =~ /.*\.flac$/i ) {
    print $wav "$rec_id-A $sox " . $UTT2SPH{$rec_id} . " -r 8000 -c 1 -t wav - remix 1|\n";
    print $wav "$rec_id-B $sox " . $UTT2SPH{$rec_id} . " -r 8000 -c 1 -t wav - remix 2|\n";
  } elsif ( $audio =~ /.*\.sph$/i ) { 
    print $wav "$rec_id-A $sph2pipe -f wav -p -c 1 " . $UTT2SPH{$rec_id} . "|\n";
    print $wav "$rec_id-B $sph2pipe -f wav -p -c 2 " . $UTT2SPH{$rec_id} . "|\n";
  } else {
    die "Unknow format (extension) of $audio -- flac or sph expected!"
  }

  print $reco "$rec_id-A ", lc($rec_id), " A\n";
  print $reco "$rec_id-B ", lc($rec_id), " B\n";


  my $p1 = new XML::Parser(Style => 'Subs');
  $p1->setHandlers(Char => \&char_handler);
  my $x=$p1->parsefile($filename);

  print STDERR "During parsing the file $filename caught $ERR errors\n" if $ERR;
  $ERR=0;
}

close($wav);
close($SEGMENTS);
close($TEXT);
close($UTT2SPK);
close($reco);

print "$0: Extracted $total_seconds seconds of audio (" . $total_seconds/3600.0 . " hours of audio)\n";
print "$0: Done\n";

