#!/usr/local/bin/perl

# File:  add_kana
# History:  
#    version 1.0  Released 9509XX

$Version=1.0;

$usage ="Usage: add_kana <OPTIONS> infile outfile|-\n".
"Version: $Version\n".
"\n".
"Desc:  add_kana adds alternate transcriptions to the CTM file\n".
"       'infile' if the kanji word can be represented in kana.  The program\n".
"       reads in the first two columns from the LDC Japanese dictionary \n".
"       '-d dictionary' then looks up all words in the input CTM file\n".
"       'infile' to see if there is an alternate transcription entry for\n".
"       that word.  If an alternate transcription exists, the filter \n".
"       adds an alternative transcription to the output file 'outfile'.'\n".
"       The script will write the output to stdout if '-' is used for\n".
"       'outfile'.\n".
"\n".
"   OPTIONS :==\n".
"       -d dictionary  The filename for the LDC Japanese dictionary\n\n";


# Get the command line arguments
require "getopts.pl";
&Getopts('d:');
if (defined($opt_d)) {
    $Dictionary = $opt_d;
} else {
    print "\n$usage\nError: LDC Dictionary not defined\n\n";
    exit();   
}


#### The main functions calls:
if ($#ARGV > 1) { print "\n$usage\nToo many arguements\n\n"; exit; } 
if ($#ARGV == 0) { print "\n$usage\nOutput Not Specified\n\n"; exit; } 
if ($#ARGV == -1) { print "\n$usage\nInput and Output Not Specified\n\n";
		    exit; } 
$InFile=$ARGV[0];
$OutFile=$ARGV[1];

if ($InFile !~ /^-$/ && ! -r $InFile){
    print "\n$usage\nInput file $InFile does not exist\n\n";
    exit;
}

###
### Read in the dictionary, saving the words in an associative array
### based on the first field
%Dict = ();
open(DICT,"< $Dictionary") || die("cannot open dictionary file $Dictionary");
while (<DICT>){
    s/^\s+//;
    @line = split;

    if ($line[0] ne $line[1]){
	$Dict{$line[0]} = $line[1];
    }
}
close(DICT);

open(FILE,"< $InFile") || die("cannot open input file $InFile"); 
open(OUTPUT,"> $OutFile") || die("cannot open output file $OutFile"); 

while (<FILE>){
    chop;

    s/^\s+//;			# delete initial spaces
    @line = split;
    if ($Dict{$line[4]} !~ /^$/){
	print OUTPUT "$line[0] $line[1] * * <ALT_BEGIN>\n";
	print OUTPUT "$line[0] $line[1] $line[2] $line[3] $line[4]\n";
	print OUTPUT "$line[0] $line[1] * * <ALT>\n";
	print OUTPUT "$line[0] $line[1] $line[2] $line[3] $Dict{$line[4]}\n";
	print OUTPUT "$line[0] $line[1] * * <ALT_END>\n";
    } else {
	print OUTPUT "$_\n";
    }
}
close(FILE);
close(OUTPUT);
