#!/usr/bin/python

# Copyright 2014 CereProc Ltd.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

# Takes input XML from idlakcex and generates lab files in an output directory

import sys, os

if len(sys.argv) < 3:
    print "cat alice_text.txt | ../../src/idlaktxpbin/idlaktxp --pretty --tpdb=../../idlak-data/en/ga/slt - - | ../../src/idlaktxpbin/idlakcex --pretty --cex-arch=hts --tpdb=../../idlak-data/en/ga/slt - - | python output_hts_test_labs.py test alice" 
    sys.exit()
    
outputdir = sys.argv[1]
id = sys.argv[2]

# create output dir if required
if not os.path.isdir(outputdir):
    os.mkdir(outputdir)

from xml.dom.minidom import parse, parseString

data = sys.stdin.read()

dom = parseString(data)

fno = 1
for idx, spt in enumerate(dom.getElementsByTagName('spt')):
    sptnostr = ('000' + str(fno))[-3:]
    phons = spt.getElementsByTagName('phon')
    tks = spt.getElementsByTagName('tk')
    fp = open(os.path.join(outputdir, id + sptnostr + '.lab'), 'w')
    fpt = open(os.path.join(outputdir, id + sptnostr + '.txt'), 'w')
    for idx2, p in enumerate(phons):
        cex_string = p.firstChild.nodeValue
        fp.write(cex_string + '\n')
    words = []
    for idx2, tk in enumerate(tks):
        w = tk.getAttribute('norm')
        if w:
            words.append(w)
    fpt.write(' '.join(words) + '\n')    
    fp.close()
    fpt.close()
    fno += 1
