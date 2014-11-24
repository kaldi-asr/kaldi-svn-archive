#!/usr/bin/perl

use Getopt::Long;

$insert_silence = "true";
$silence_token = "<SIL>";

GetOptions("insert-silence=s" => \$insert_silence, 
           "silence-token=s" => \$silence_token
           );

if (@ARGV != 2) {
  print STDERR "This script combines read an SCP file and combines \n";
  print STDERR "all the segments corresponding to a particular recording \n";
  print STDERR "into one utterance.\n\n";
  print STDERR "Usage: local/resegment/combine_segments.pl <TYPE> <SEGMENTS_FILE>\n";
  print STDERR " e.g.: cat data/train/text | local/resegment/combine_segments.pl  text data/train/segments > exp/make_segments/data/train_whole/segments\n";
  print STDERR "<TYPE> must be one of \"text\" or \"utt2spk\"\n";
  print STDERR "Options: \n";
  print STDERR "\t--insert-silence <true|false>   # Insert silences while \n";
  print STDERR "\t                                # combining utterances\n";
  print STDERR "\t--silence-token  <token>        # Token to be inserted as \n";
  print STDERR "\t                                # silence (default: <SIL>)\n";
  exit(1);
}

print STDERR $0 . " " . join(" ", @ARGV) . "\n";

if ($insert_silence ne "true" && $insert_silenc ne "false") {
  print STDERR "insert-silence must be either true or false. But it is $insert_silence\n";
  exit(1);
}

$type = $ARGV[0];
$segmentsFile = $ARGV[1];

if ($type ne "utt2spk" && $type ne "text") {
  print STDERR "<TYPE> must be either utt2spk, or text. But it is $text\n";
  exit(1);
}

open(SEGMENTS, $segmentsFile)
    || die "Unable to read segments file $segmentsFile";
$numSegments = 0;

my $num_failed_parses=0;
my $num_failed_parses_max=10;

while ($line = <SEGMENTS>) {
  @tokens = split(/\s+/, $line);
  unless ($#tokens == 3) {
    $num_failed_parses+=1;
    print STDERR "$0: Couldn't parse line $. in $segmentsFile\n" 
    if ($num_failed_parses == 1);
    print STDERR ("\tLine: $line")
    if ($num_failed_parses le $num_failed_parses_max);
    print STDERR "$0: Maximal threshold for failed line parses reached. Not warning anymore\n"
    if ($num_failed_parses eq $num_failed_parses_max);
    next;
  }
  $segmentID = shift @tokens;
  if (exists $fileID{$segmentID}) {
    print STDERR ("$0: Skipping duplicate segment ID $segmentID in $segmentsFile\n");
    next;
  }
  $fileID{$segmentID}    = shift @tokens;
  
  ++$numSegments;
}
close(SEGMENTS);
print STDERR ("$0: Read info about $numSegments segment IDs from $segmentsFile\n");
print STDERR ("$0: $num_failed_parses lines failed to parse.\n\n");

# Read SCP file
$numLines = 0;
$num_missing = 0;
$num_missing_max = 10;

while ($line = <STDIN>) {
  if ($type eq "text") {
    @tokens = split(/\s+/, $line);
    unless (@tokens >= 2) {
      print STDERR "$0: Couldn't parse line $line in text file\n";
      exit(1);
    }
    $segmentID = shift @tokens;
    
    unless (exists $fileID{$segmentID}) {
      ++$num_missing;
      print STDERR ("$0: Recording ID not found for segment $segmentID in $segmentsFile\n")
      if ($num_missing == 1);

      print STDERR ("Segment $segmentID\n") 
      if ($num_missing <= $num_missing_max);

      print STDERR "$0: Maximal threshold for failed line parses reached. Not warning anymore\n"
      if ($num_missing == $num_missing_max);
      next;
    }

    $this_fileID = $fileID{$segmentID};
    
    if (! defined $prev_fileID) {
      print STDOUT $this_fileID . " ";
      if ($insert_silence eq "true") {
        print STDOUT $silence_token . " ";
      }
      print STDOUT join(" ", @tokens);
    } elsif ($this_fileID eq $prev_fileID) {
      if ($insert_silence eq "true") {
        print STDOUT " " . $silence_token;
      }
      print STDOUT " " . join(" ", @tokens);
    } else {
      if ($insert_silence eq "true") {
        print STDOUT " " . $silence_token;
      }
      print STDOUT "\n";
      print STDOUT $this_fileID . " ";
      if ($insert_silence eq "true") {
        print STDOUT $silence_token . " ";
      }
      print STDOUT join(" ", @tokens);
    }
    $prev_fileID = $this_fileID;
  } elsif($type eq "utt2spk") {
    @tokens = split(/\s+/, $line);
    unless (@tokens == 2) {
      print STDERR "$0: Couldn't parse line $line in utt2spk-like file\n";
      exit(1);
    }
    $segmentID = shift @tokens;
    $speakerID = shift @tokens;

    unless (exists $fileID{$segmentID}) {
      ++$num_missing;
      print STDERR ("$0: Recording ID not found for segment $segmentID in $segmentsFile\n")
      if ($num_missing == 1);

      print STDERR ("Segment $segmentID\n") 
      if ($num_missing <= $num_missing_max);

      print STDERR "$0: Maximal threshold for failed line parses reached. Not warning anymore\n"
      if ($num_missing == $num_missing_max);
      next;
    }

    if (! exists $seenFileID{$fileID{$segmentID}}) {
      print STDOUT "$fileID{$segmentID} $speakerID\n";
      $seenFileID{$fileID{$segmentID}} = 1;
    }
  }
  ++$numLines;
}
