tranfilt/readme.txt:

 	           Transcription Prefilter Scripts
			  for ARPA CSR Tests
                             Version 1.14


I. Description

This directory contains a filter package used to "pre-filter" .lsn
hypothesis and reference transcriptions to apply a set of word-mapping
rules prior to scoring.  The word-mapping rules permit the elimination
of ambiguous lexical representations.  The filter applies the rules
located in two word map files to the transcriptions.  The first word
map file, "<TEST>_<VERSION>.glm", applies a set of rules globally to
all transcriptions.  The second, "<TEST>_<VERSION>.utm", applies
particular rules to particular utterances.  The two .map files are
named so as to indicated the test they pertain to.

To implement the filtering process, the shell script, 'csrfilt.sh', is
first called.  The heart of the filter is the C program, "rfilter1",
which applies the set of string-transformation rules in the two
word map files.

In order to allow several variants of a word or phrase to match the
reference transcription without error, the filter has been used to
apply transformation rules that map all lexical variants into just
one.  The filter is applied to both reference and hypothesized
transcriptions.


II. Installation and Usage

The filter, 'csrfilt.sh', requires the following software/packages:

	UNIX sh Shell
	Perl 
	ANSI C Compiler (gcc)
	Make

The first step is to compile the rfilter1 executable.  Do so
by typing the UNIX commands:

	% cp -r ./tranfilt <YOUR_DIR>
	% make

After the make is complete, you can use the filter as a stdin/stdout
filter by using the syntax:

	% csrfilt.sh global-map-file utterance-map-file < filein > fileout

The option '-dh' can be added to the command line to delete hyphens from
all hyphenated words.


Note:  You must have an installed copy of the PERL interpreter to use
       this utility.  PERL is a publically available software package
       which combines the functionality of several UNIX text processing
       utilities with a programming language.


III. Testing

	A test input and output file has been provided with this
distribution.  To run the test, type the command:
	
	% make test

If there are no error messages, the test completed successfully.


IV. Revision history.

A revision log is maintained at the top of each script or source code
file.  

