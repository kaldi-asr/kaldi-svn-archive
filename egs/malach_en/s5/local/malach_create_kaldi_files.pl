#!/usr/bin/env perl

# Copyright 2015  Johns Hopkins University (Author: Jan Trmal).  Apache 2.0.


#Parses the transcription files and generate the basic kaldi 
#files:
#  wav.scp
#  text
#  segments
#  utt2spk

#There is the question of which channel we should process (as the audio is stereo)
#For file 00002-002.mp2 we will generate two entries in the wav.scp:
#  00002-002-A
#  00002-002-B

#Next is the speaker identity -- in trs files, the speaker is usually 
#named as spk1, spk2... and so on..
#As generally the speaker will be per file, even though the interviewees could
#span multiple segments (such as 00002-001, 00002-002, and so on) and 
#the interviewer could even span several conversations (00002-002, 00002-003, 00003-001)
#we will ignore this for the time being, as we do not have any guarantee
#that the speaker IDs are synchronized (i.e. that spk1 in two files refer to 
#the same person. While it might be true for segments of a single conversation
#it is certainly not true for different conversations
#The segment-local speaker ID will have this form:
#  00002-002-A-SPK1

#Now, the utterance
#each utterance is defined by a <Sync ..> event in the trs. we just add the 
#time information  (seconds since beginning of the file multiplied by 100,
#which should be fine, as there the time stamp has mostly only 2 decimal digits
#in the transcription. The segments are (at most) 30 minutes (i.e. 18000) long,
#i.e to make the utterances IDs "aligned", we will need 6digits ID
#
#  00002-002-A-SPK1-000230


use strict;
use warnings;
use Data::Dumper;
use Getopt::Long;

binmode STDIN, ":utf8";
binmode STDOUT, ":utf8";
binmode STDERR, ":utf8";

my $rate=16000;

my $audio=$ARGV[0];
my $transcripts=$ARGV[1];
my $datadir=$ARGV[2];

open(my $A, $audio) or die "Could not open \"$audio\": $!\n";
open(my $T, "<:utf8",  $transcripts) or die "Could not open \"$transcripts\": $!\n";
open(my $TEXT, ">:utf8", "$datadir/text") or die "Could not open \"$datadir/text\": $!\n";
open(my $SEGMENTS, ">:utf8", "$datadir/segments") or die "Could not open \"$datadir/segments\": $!\n";
open(my $WAV, ">:utf8", "$datadir/wav.scp") or die "Could not open \"$datadir/wav.scp\": $!\n";
open(my $UTT2SPK, ">:utf8", "$datadir/utt2spk") or die "Could not open \"$datadir/utt2spk\": $!\n";
open(my $RECO, ">:utf8", "$datadir/reco2file_and_channel") or die "Could not open \"$datadir/reco2file_and_channel\": $!\n";


my $SOX=`which sox` or die "Could not find sox executable: $!\n";
my $MPG=`which mpg123` or die "Could not find mpg123 excutable: $!\n";
chomp $SOX;
chomp $MPG;

while (my $audio=<$A>) {
  chomp $audio;
  my $file=`basename $audio`;
  chomp $file;

  my $base=$file;
  if ($file =~ /\.mp2$/ ) {
    $base=~s/\.mp2$//g;
    print $WAV "$base-A $MPG -q -b 10240 -s $audio | $SOX -t raw -r 44100 -e signed -b 16 -c 2 - -r $rate -c 1 -t wav - remix 1|\n";
    print $WAV "$base-B $MPG -q -b 10240 -s $audio | $SOX -t raw -r 44100 -e signed -b 16 -c 2 - -r $rate -c 1 -t wav - remix 2|\n";
  } elsif ($file =~ /\.wav$/) { 
    $base=~s/\.wav$//g;
    print $WAV "$base-A $SOX $audio -r $rate -c 1 -t wav - remix 1|\n";
    print $WAV "$base-B $SOX $audio -r $rate -c 1 -t wav - remix 2|\n";
  } else {
    die "Unknown audio format for file $audio\n";
  }
}
close($WAV);

my @UNK_END_TIME;
my $empty_lines_warn=0;
my $empty_lines_warn_max=3;

while (my $line=<$T>) {
  chomp $line;
  (my $header, my $text) = split(" ", $line, 2);
  unless ($text) {
    print STDERR "Warning, empty line: $line\n" if $empty_lines_warn <=$empty_lines_warn_max;
    print STDERR "Maximum number of warning reached, not warning again.\n" if $empty_lines_warn == $empty_lines_warn_max;
    $empty_lines_warn += 1;
    next;
  }
  (my $file, my $time_start, my $time_end, my $spk)  = ($header=~ /\[(.*?)\]\[(.*?)\]\[(.*?)\]\[(.*?)\]/g);
  #my @matches = ($header=~ /\[(.*?)\]\[(.*?)\]\[(.*?)\]/g);
  #print Dumper(\@matches);
  
  if ( $time_end eq "???" ) {
    if ( @UNK_END_TIME) {

      if ( ($UNK_END_TIME[0]->[0] ne $file ) || 
           ($UNK_END_TIME[0]->[1] ne $time_start ) ) {

        undef @UNK_END_TIME;

      }
    }
    push @UNK_END_TIME, [$file, $time_start, $spk, $text];
    next;
  }
  
  if ( @UNK_END_TIME ) {
    if ( ($UNK_END_TIME[0]->[0] eq $file ) &&
         ($UNK_END_TIME[0]->[1] eq $time_start ) ) {
      
      foreach my $entry (@UNK_END_TIME) {
        my $spk = $entry->[2];
        my $text = $entry->[3];
        foreach my $chan ( ("A", "B") ) {
          my $time_id=sprintf("%08d", ($time_start) * 1000);
          my $segment_id="$file-$chan-". uc($spk) . "-$time_id";
          
          if ( ($time_start - $time_end) > -0.01 ) {
            print STDERR "Warning: Segment is too short or time_end > time_start: ";
            print STDERR "$segment_id $file-$chan $time_start $time_end $text\n";
            next;
          }
          
          print $SEGMENTS "$segment_id $file-$chan $time_start $time_end\n";
          print $TEXT "$segment_id $text\n";
          print $UTT2SPK "$segment_id $file-$chan-" . uc($spk) . "\n";
        }
      }
      undef @UNK_END_TIME;
    } elsif ($UNK_END_TIME[0]->[0] eq $file ) {
      foreach my $entry (@UNK_END_TIME) {
        my $spk = $entry->[2];
        my $text = $entry->[3];
        my $local_time_start=$entry->[1];
        my $local_time_end=$time_start;
        foreach my $chan ( ("A", "B") ) {
          my $time_id=sprintf("%08d", ($local_time_start) * 1000);
          my $segment_id="$file-$chan-". uc($spk) . "-$time_id";
          if ( ($local_time_start - $local_time_end) > -0.1 ) {
            print STDERR "Warning: Segment is too short or time_end > time_start: ";
            print STDERR "$segment_id $file-$chan $local_time_start $local_time_end $text\n";
            next;
          }
        
          print $SEGMENTS "$segment_id $file-$chan $local_time_start $local_time_end\n";
          print $TEXT "$segment_id $text\n";
          print $UTT2SPK "$segment_id $file-$chan-" . uc($spk) . "\n";
        }
      }
    } else {
      print STDERR "Warning: Could not deduce the ending time before this line: \n\t$line\n";
      print STDERR "\tprevious line: ";
      foreach my $entry (@UNK_END_TIME) {
        my $file = $entry->[0];
        my $time_start = $entry->[1];
        my $spk = $entry->[2];
        my $text = $entry->[3];
        print STDERR "[$file][$time_start][???][$spk] $text\"\n";
      }
    }
    undef @UNK_END_TIME;
  }

  foreach my $chan ( ("A", "B") ) {
    my $time_id=sprintf("%08d", ($time_start) * 1000);
    my $segment_id="$file-$chan-". uc($spk) . "-$time_id";
    
    if ( ($time_start - $time_end) > -0.01 ) {
      print STDERR "Warning: Segment is too short or time_end > time_start: ";
      print STDERR "$segment_id $file-$chan $time_start $time_end $text\n";
      next;
    }
    print $SEGMENTS "$segment_id $file-$chan $time_start $time_end\n";
    print $TEXT "$segment_id $text\n";
    print $UTT2SPK "$segment_id $file-$chan-" . uc($spk) . "\n";
  }
}

close($TEXT);
close($SEGMENTS);
close($UTT2SPK);
close($RECO);

