      open (TEXT, $ARGV[0]);
      my %REF;
      while( <TEXT> ) {
        @F = split " ";
        $REF{$F[0]} = join(" ", @F[1...$#F]);
      }
      close(TEXT);

      while (<STDIN> ){
        @F=split " ";
        my $utt = $F[0];
        $utt =~ s/-[0-9][0-9]*$//;
        print $F[0] . " " . $REF{$utt} . "\n";
      }
