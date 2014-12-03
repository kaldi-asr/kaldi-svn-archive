#! /usr/bin/env perl

$data=$ARGV[0];
open (SEGMENTS_IN, "$data/segments") or die "Cannot read $data/segments: $!";
open (RECO, ">$data/reco2file_and_channel.crosstalk");
open (SEGMENTS_OUT, "> $data/segments.crosstalk");

while ($line = <SEGMENTS_IN>) {
  ($utt, $file, $start, $end)= split " ", $line;
  ($file, $spk, $xstart, $xend)=split "-", $utt;
  $file_ctm=lc($file) . ".eng_$spk";
  $file="${file}.eng_$spk";
  print SEGMENTS_OUT "$utt $file $start $end\n";
  
  $chan=1;
  $chan=2 if ($spk =~ /^B.*/);

  print RECO "$file $file_ctm $chan\n";
}

close(SEGMENTS_IN);
close(SEGMENTS_OUT);
close(RECO)
