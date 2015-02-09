#!/usr/bin/env perl

use strict;
use warnings;
use XML::Parser;
use Data::Dumper;
use Getopt::Long;

binmode STDIN, ":utf8";
binmode STDOUT, ":utf8";
binmode STDERR, ":utf8";

my @TEXT;
my @SPK;
my $FILENAME;
my $SYNC;
my $WHO;

my $print_unks;
my $print_crosstalks;
my $text_max_len = 50;
my $warn_inline_speakers;

GetOptions ("unks!" => \$print_unks,
            "crosstalks!"   => \$print_crosstalks,
            "max-text-len=i" => \$text_max_len,
            "warn-inline-speakers!" => \$warn_inline_speakers,
            );

sub Warning {
  my $FILENAME=$_[0];
  my $SYNC=$_[1];
  my $TEXT=$_[2];

  print STDERR "Warning: $FILENAME:$SYNC $TEXT\n";
}

sub print_line {
  my $FILENAME=$_[0];
  my $SYNC=$_[1];
  my $SPK=$_[2];
  my $TEXT=$_[3];
  
  my @words = split(" ", $TEXT);
  if ((@words > $text_max_len ) && ($text_max_len > 0) ){
    Warning($FILENAME, $SYNC, "text too long to be usefull: \"$TEXT\"");
  }
  $TEXT =~ s/   */ /g;
  $TEXT =~ s/^\s+|\s+$//g ;
  print "[$FILENAME][$SYNC][$SPK] $TEXT\n" if $TEXT;
}

sub produce_text {
  my $text=join("", @TEXT);
  $text =~ s/\< */ </g;
  $text =~ s/ *\>/> /g;
  $text =~ s/   */ /g;
  $text =~ s/@ /@/g;
  $text =~ s/^\s+|\s+$//g ;
  return if not $text;
  
  if ( @SPK) {
    if ($text =~ m/\<spkr?[0-9]\>/) {
      my $spk=$&;
      my @spk_count = $text =~ /\<spkr?[0-9]\>/g;
      if (@spk_count > 1) {
        Warning($FILENAME, $SYNC, "multiple inline speaker-ids [" . join(" ", @spk_count) . "] in string \"$text\"");
        Warning($FILENAME, $SYNC, "reseting speaker info");
        undef @SPK;
        return
      }
      $spk=~s/\<|\>|r//g;
      @SPK=split(" ", $spk);
      if (@SPK > 1) {
        Warning($FILENAME, $SYNC, "multiple inline speaker-ids [" . join(" ", @SPK) . "] in string \"$text\"");
        Warning($FILENAME, $SYNC, "reseting speaker info");
        undef @SPK;
        return
      }
      if ($warn_inline_speakers) {
        Warning($FILENAME, $SYNC, "inline speaker info in \"$text\"");
      }
      $text =~s/ *\<spkr?[0-9]\> */ /; 
      print_line $FILENAME, $SYNC, $SPK[0], $text; 
    }else {
      if (defined $WHO) {
        if (@SPK eq 1) {
          Warning($FILENAME, $SYNC, "cross-talk info detected, only one speaker info present");
          return
        }
        if ( $print_crosstalks ) {
          #print STDERR Dumper( [$FILENAME,$SYNC,\@SPK, $WHO,] );
          print_line $FILENAME, $SYNC, $SPK[$WHO-1], $text; 
        } else {
          Warning($FILENAME, $SYNC, "cross-talk info detected, ignoring text \"$text\"");
        }
      } else {
        print_line $FILENAME, $SYNC, $SPK[0], $text; 
      }
    }
  } else {
    if ($text =~ m/\<spkr?[0-9]\>/) {
      my $spk=$&;
      $spk=~s/\<|\>|r//g;
      my @spk_count = $text =~ /\<spkr?[0-9]\>/g;
      if (@spk_count > 1) {
        Warning($FILENAME, $SYNC, "multiple inline speaker-ids [" . join(" ", @spk_count) . "] in string \"$text\"");
        return
      }
      @SPK=split(" ", $spk);
      if (@SPK > 1) {
        Warning($FILENAME, $SYNC, "multiple inline speaker-ids [" . join(" ", @SPK) . "] in string \"$text\"");
        return
      }
      if ($warn_inline_speakers) {
        Warning($FILENAME, $SYNC, "inline speaker info in \"$text\"");
      }
      $text =~s/ *\<spkr?[0-9]\> */ /; 
      print_line  $FILENAME, $SYNC, $SPK[0], $text; 
    }else {
      if ( $print_unks ) {
        print_line $FILENAME, $SYNC, "unk", $text;
      } else {
        Warning($FILENAME, $SYNC, "unknown speaker in string \"$text\"");
      }
    }

  }
}

sub Who {
  my @P=@_[1..$#_];
  #print Dumper(\@P);
  
  unless ( $WHO ) {
    my( $index )= grep { $P[$_] eq "nb" } 0..$#P;
    $WHO=$P[$index+1] if defined $index;
    undef @TEXT;
    return
  };

  produce_text;

  my( $index )= grep { $P[$_] eq "nb" } 0..$#P;
  $WHO=$P[$index+1] if defined $index;
  undef @TEXT;
}

sub Turn  {
  my @P=@_[1..$#_];
  #print Dumper(\@P);
  my( $index )= grep { $P[$_] eq "speaker" } 0..$#P;
  @SPK=split(" ",$P[$index+1]) if defined $index;
  #print "Speaker defined: " . join(" ", @SPK)\n" if defined @SPK;
}

sub Turn_  {
  my @P=@_[1..$#_];
  #print Dumper(\@P);
  #print "End-of-turn\n";
  my $text=join("", @TEXT);
  #print Dumper(\@TEXT);
  unless ( $text ) {
    undef @TEXT;
    undef @SPK;
    undef @TEXT;
    undef $SYNC;
    undef $WHO;
    return
  }
  unless ($SYNC) {
    undef @TEXT;
    undef @SPK;
    undef @TEXT;
    undef $SYNC;
    undef $WHO;
    return
  
  };
  produce_text;

  undef @TEXT;
  undef @SPK;
  undef @TEXT;
  undef $SYNC;
  undef $WHO;
}

sub Sync {
  my @P=@_[1..$#_];
  #print Dumper(\@P);
  my( $index )= grep { $P[$_] eq "time" } 0..$#P;
  my $time=$P[$index + 1];
  
  unless ( @TEXT ) {
    $SYNC=$time;
    return
  }
  
  unless (defined $SYNC) {
    $SYNC=$time;
    #print Dumper(\@TEXT);
    undef @TEXT;
    return
  
  };
  
  produce_text;

  $SYNC=$time;
  undef @TEXT;
}

sub Sync_ {
}

sub Char {
  my $H=$_[0];
  my @P=@_[1..$#_];
  #print Dumper($H);
  if (defined $H->{Context}) {
    my @C=@{$H->{Context}};
    #print Dumper("Context", $C[$#C]);
    chomp $P[0];
    return unless $P[0];
    push @TEXT, $P[0] if $C[$#C] eq "Turn";
    #print Dumper("TEXT", $C[$#C], \@TEXT, \@P);
    #print Dumper("TEXT", \@TEXT, \@P);
  }

}
sub Event  {
  my $H=$_[0];
  my @P=@_[1..$#_];
  
  my( $index_desc )= grep { $P[$_] eq "desc" } 0..$#P;
  my( $index_extent )= grep { $P[$_] eq "extent" } 0..$#P;

  my $desc = $P[$index_desc + 1];
  my $extent = $P[$index_extent + 1];

  if ($extent eq "instantaneous") {
    push @TEXT, "<$desc>";
  } elsif ($extent eq "begin") {
    push @TEXT, "<${desc}_BEGIN>";
  } elsif ($extent eq "end") {
    push @TEXT, "<${desc}_END>";
  } elsif ($extent eq "previous") {
    push @TEXT, "<${desc}_PREV>";
  } else {
    print STDERR "Unknown extent $extent for event $desc\n";
  }

}

while (my $line = <STDIN>) {
  print STDERR $line;
  chomp $line;
  $FILENAME=`basename $line`;
  chomp $FILENAME;
  $FILENAME=~s/\..*//;
  my $p1 = new XML::Parser(Style => 'Subs');
  $p1->setHandlers('Char'  => \&Char);
  $p1->parsefile($line);
}



