#!/usr/bin/perl

# This script is used to convert the headers cblas.h and common_interface.h in
# this directory, by adding const.  This makes them easier to use in external code
# which uses "const".  We just add const to everything.  This is not really right.
# What needs to be done is to use "const" correctly throughout the project, but
# this would take a lot of time.

if (@ARGV != 2) {
  print "Usage: add_const_guided.pl const-source-file source-file >dest-file\n"
}

open(C, "<$ARGV[0]") || die "Opening $ARGV[0]";
open(S, "<$ARGV[1]") || die "Opening $ARGV[1]";

sub is_decl { 
  # Ireturn 1 if this line is the beginning of a function
  # declaration
  my $l = $_[0];
  if ($l  =~ m/^\s*\w+\s+/) {}
}

while (<C>) { # read the source of the "const" in declarations.
  $a = $_;
  $a =~ s:/\*[^*]+\*/::g; # Remove comments.
  while ($a =~ m/,\s*$/ ||           # line ends with a comma -> whatever-it-is spills over to next line.
         $a =~ m/catlas_.set\s*$/ || # this is is a special case in cblas.h
         $a =~ m/\([^)]+$/) {        # unmatched paren; happens in clapack.h
    $a .= <C>; # Append the next line.
  }
  if ($a =~ m/\);\s*$/) { # A function declaration
    if ($a =~ m/^\s*\w+\s+(\w+)\s*\((.*)\)\;\s*$/s) {
      $name = $1;
      $decl = $2;
      $name =~ s/cblas_//; # remove cblas_ prefix
      $name =~ s/_$//; # remove _ suffix for LAPACK functions
      if (defined $decl_of{$name}) { print STDERR "Warning: overwriting decl of $name\n"; }
      $decl_of{$name} = $decl;
    } else {
      print STDERR "Warning: decl $a doesn't look right\n";
    }
  }
}

while (<S>) { # read the source while to which we'll add const.
  $a = $_;
  while ($a =~ m/,\s*$/) {           # line ends with a comma -> whatever-it-is spills over to next line.
    $a .= <S>; # Append the next line.
  }
  if ($a =~ m/\);\s*$/) { # A function declaration
    if ($a =~ m/^(\s*\w+\s+)(\w+|BLASFUNC\(\w+\))(\s*)\(([^)(]*)\)\;\s*$/s) {
      $prefix = $1;
      $orig_name = $2; $name = $orig_name;
      $space = $3;
      $decl = $4;
      
      $name =~ s/cblas_//; # remove cblas_ prefix
      if ($name =~ m/^BLASFUNC\((\w+)\)$/) {
        $name = $1; # remove the BLASFUNC(...) macro around the name.
      }
      $netlib_decl = $decl_of{$name};
      if (defined $netlib_decl) {
        @Decl = split(",", $decl);
        @NetlibDecl = split(",", $netlib_decl);
        $n1 = @Decl; $n2 = @NetlibDecl; # lengths of arrays.
        if ($n1 == $n2) {
          for ($n = 0; $n < $n1; $n++) {
            if ($NetlibDecl[$n] =~ m/^\s*const/ && !($Decl =~ m/^\s*const/)) {
              $d = $Decl[$n];
              $Decl[$n] =~ s/^(\s*)(\S)/$1const $2/; # put "const " after any initial space.
            }
          }
          $decl = join(",", @Decl); # re-join after adding const.
          $a = "${prefix}${orig_name}${space}(${decl});\n";
        } else {
          print STDERR "Declarations have different sizes for $name: $n1 vs $n2\n";
        }
      } else {
        print STDERR "No declaration in netlib for $name\n";
      }
    } else {
      print STDERR "Warning: decl $a in $ARGV[1] doesn't look right\n";
    }
  }
  print $a;
}

