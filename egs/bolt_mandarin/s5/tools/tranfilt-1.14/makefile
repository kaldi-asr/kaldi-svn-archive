# **********************************************************
# Makefile for SRC Binary Directory
# **********************************************************
SHELL = /bin/sh

rfilter1: rfilter1.c scripts
	gcc -o rfilter1 rfilter1.c

scripts:
	sed -e 's:^EXE_DIR=.*:EXE_DIR='`pwd`':' < csrfilt. > csrfilt.sh ;
	sed -e 's:EXE_DIR=.*:EXE_DIR="'`pwd`'";:' < ut_rfilt. > ut_rfilt.pl
	chmod +x csrfilt.sh ut_rfilt.pl acomp.pl add_kana.pl chfilt.pl def_art.pl text2lex.pl;
	
test:
	echo "If no output is produced by diff, the test passes"
	sh csrfilt.sh -dh example.glm example.utm < test.in | diff - test.out
	sh csrfilt.sh -i ctm -dh example.glm example.utm < test_ctm.in | diff - test_ctm.out
	./chfilt.pl -m -l english testmss.in -- - | diff - testmss.out
	./chfilt.pl -l english -i -b 0 -e 999999 -k test.txt -- - | diff - test.txt.-optdel.out
	./chfilt.pl -d -l english -i -b 0 -e 999999 -k test.txt -- - | diff - test.txt.out
	./chfilt.pl -c -l english -i -b 0 -e 999999 -k test.txt -- - | diff - test.txt.-contexp.out
	./chfilt.pl -c -l arabic -i -b 0 -e 999999 -k test.arb.txt -- - | diff - test.arb.txt.out

clean:
	rm -f csrfilt.sh ut_rfilt.pl rfilter1

dist:
	echo "tranfilt-"`grep Version\  readme.txt | head -1 | perl -pe 's/.*Version\s+//'` > .fname	
	echo "Building a release version" `cat .fname`
	echo "Verifying and up-to-date CVS copy"
	cvs up 
	cvs co -d `cat .fname` tranfilt 
	echo "Building the TAR file"
	tar czf `cat .fname`"-"`date +%Y%m%d-%H%M`.tgz --exclude CVS `cat .fname` 
	rm -r `cat .fname`

