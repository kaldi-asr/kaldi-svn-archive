Installation TIPS for KALDI and installation INSTRUCTIONS for my additional repositories
=================================================================================
Intro
-----
Kaldi has very good instructions and tutorial
for building it from source. It is easy and straightforward.
However, I needed also to build shared libraries
and maybe you will face some of my problems too.
So this is the reasons for writing my building procedure down.

Please note than I committed changes which enables much easier installation to `sandbox/oplatek`.
The changes is not merged into trunk at the time of writing (Let me know if this is obsoleted!).
The mark 'deprecated' on the line bellow marks the changes which I did and that you need to do *ONLY FOR THE TRUNK VERSION OF KALDI*.
*[DEPRECATED BY `sandbox/oplatek`]*

Installing external dependencies
================================
See `kaldi-trunk/tools/INSTALL` for info.
Basically it telss you to use `kaldi-trunk/tools/Makefile`, which I used also.

How have I installed OpenBlas?
----------------------
```bash
make openblas
```

How have I installed Openfst?
----------------------
```bash
make openfst_tgt
```

*[DEPRECATED BY `sandbox/oplatek`]*
In order to install also shared libraries
I changed the line 37 in 
`kaldi-trunk/tools/Makefile`

```sh
*** Makefile 
************
*** 34,38 ****

openfst-1.3.2/Makefile: openfst-1.3.2/.patched
		cd openfst-1.3.2/; \
!		./configure --prefix=`pwd` --enable-static --disable-shared --enable-far --enable-ngram-fsts

--- 34,38 ----

openfst-1.3.2/Makefile: openfst-1.3.2/.patched
		cd openfst-1.3.2/; \
!		./configure --prefix=`pwd` --enable-static --enable-shared --enable-far --enable-ngram-fsts

```
Than I ran
```bash
make openfst_tgt
```

How have I installed PortAudio?
--------------------------
NOTE: Necessary only for Kaldi online decoder
```bash
extras/install_portaudio.sh
```

*[DEPRECATED BY `sandbox/oplatek`]* 
In kaldi-trunk/tools/extras/install_portaudio.sh
I changed line
```
./configure --prefix=`pwd`/install
```
to
```
./configure --prefix=`pwd`/install --with-pic
```

Then I ran
```bash
extras/install_portaudio.sh
```


How have I built Kaldi?
------------------
```bash
./configure --openblas-root=`pwd`/../tools/OpenBLAS/install --fst-root=`pwd`/../tools/openfst --static-math=no
```

*[DEPRECATED BY `sandbox/oplatek`]* 
Edit the `kaldi.mk` and add the `-fPIC` flag.
```bash
EXTRA_CXXFLAGS=-fPIC make
EXTRA_CXXFLAGS=-fPIC make ext
```
But the local makefiles overrides `EXTRA_CXXFLAGS`.

If you updated from the svn repository do not forget to run `make depend`
Since by *default it is turned of! I always forget about that!*
```
# DO NOT FORGET TO CHANGE kaldi.mk TODO SCRIPT IT!
# make depend and make ext_depend are necessary only if dependencies changed
make depend && make ext_depend && make && make ext
```

How have I updated Kaldi src code?
----------------------------
I checkout the kaldi-trunk version.

[Kaldi install instructions](http://kaldi.sourceforge.net/install.html)

Note: If you checkout Kaldi before March 2013 you need to relocate svn. See the instructions in the link above!


What setup did I use?
--------------------
In order to use Kaldi binaries everywhere I add them to `PATH`. 
In addition, I needed to add `openfst` directory to `LD_LIBRARY_PATH`, I compiled Kaldi dynamically linked against `openfst`. To conclude, I added following lines to my `.bashrc`.
```bash
############# Kaldi ###########
kaldisrc=/net/work/people/oplatek/kaldi/src
export PATH="$PATH":$kaldisrc/bin:$kaldisrc/fgmmbin:$kaldisrc/gmmbin:$kaldisrc/nnetbin:$kaldisrc/sgmm2bin:$kaldisrc/tiedbin:$kaldisrc/featbin:$kaldisrc/fstbin:$kaldisrc/latbin:$kaldisrc/onlinebin:$kaldisrc/sgmmbin

### Openfst ###
openfst=/ha/home/oplatek/50GBmax/kaldi/tools/openfst
export PATH="$PATH":$openfst/bin
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH":$openfst/lib 
```

Which tool for building a Language Model (LM) have I used?
---------------------------------------------------------
None. I received built LM in Arpa format.

NOTE: Probably, I should build my own LM. 


How have I installed Atlas the complicated way?
--------------------
 * The main problem with building ATLAS was for me disabling CPU throtling. I solved it by :
```bash
# running following command under root in my Ubuntu 12.10
# It does not turn off CPU throttling in fact, but I do not need the things optimaze on my local machine
# I ran it for all of my 4 cores
for n in 0 1 2 3 ; do echo 'performance' > /sys/devices/system/cpu/cpu${n}/cpufreq/scaling_governor ; done
```

 * Then I needed to install Fortran compiler (The error from configure was little bit covered by consequent errors) by 

```bash
sudo apt-get install gfortran
```

 * On Ubuntu 12.04 I had issue with 

```bash
/usr/include/features.h:323:26: fatal error: bits/predefs.h
```

   Which I solved by

```bash
sudo apt-get install --reinstall libc6-dev
```

 * Finally, in `kaldi-trunk/tools/ATLAS` I run:

```bash
mkdir build 
mkdir ../atlas_install
cd build
../configure --shared --incdir=`pwd`/../../atlas_install
make 
make install
 ```
