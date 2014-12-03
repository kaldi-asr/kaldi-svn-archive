#/usr/bin/env perl

$current_order=0;

@SYMBOLS;
open(WORDS, $ARGV[0]) or die "Cannot open $ARGV[0]: $!";
while (<WORDS>) {
  chomp;
  push @SYMBOLS, $_;
  #print STDERR "$_\n";
}
close(WORDS);

while ($line = <STDIN>) {
  if ( $line =~ /^.1-grams:/) {
    $current_order=1;
    print STDERR $line;
    print $line;
  } elsif ( $line =~ /^.[0-9]-grams:/) {
    $current_order=0;
    print $line;
  } else {
    if ($current_order == 1 ) {
      
      @F=split " ", $line;
      next if @F == 0;

      die "Cannot parse line $line"  if ((@F < 2 ) || ( @F > 3));
      if ($F[1] eq "<UNK>") {
        foreach $symbol (@SYMBOLS) {
          print "$F[0] $symbol $F[2]\n";
          print STDERR "Adding: $F[0] $symbol $F[2]\n";
        }
        $current_order=0;
      }     
    }
    print $line;
  }
}
