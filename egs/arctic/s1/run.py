#!/usr/bin/python

# Copyright 2013 CereProc Ltd.

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

import sys
import os
import xml.sax

#URL of arctic transcriptions
TRANSURL = 'http://festvox.org/cmu_arctic/cmuarctic.data'
WAVSURL = 'http://www.speech.cs.cmu.edu/cmu_arctic/packed/cmu_us_bdl_arctic-0.95-release.zip'
TRANSNAME = "cmuarctic.data"
WAVDIR = "cmu_us_bdl_arctic/wav"


# sax handler 
class saxhandler(xml.sax.ContentHandler):
    def __init__(self):
        self.id = ''
        self.data = [[]]
        self.ids = []
        self.lex = {}
        self.oov = {}
        
    def startElement(self, name, attrs):
        if name == "fileid":
            print self.id, attrs['id']
            newid = attrs['id']
            if self.id and newid != self.id:
                self.data.append([])
                self.id = newid
                self.ids.append(self.id)
            if not self.id:
                self.id = newid
                self.ids.append(self.id)
        if name == "tk":
            word = attrs['norm'].upper()
            print word
            self.data[-1].append(word)
            if not self.lex.has_key(word):
                self.lex[word] = {}
            if attrs.has_key('lts') and attrs['lts'] == 'true':
                self.oov[word] = 1
            if attrs.has_key('altprons'):
                prons = attrs['altprons'].split(', ')
            else:
                prons = [attrs['pron']]
            for p in prons:
                self.lex[word][p] = 1
    
                
                            
            
            
# Download corpus
# force will download whether already present or not
def download(datadir, force = False):
    # check cmu arctic transciption file is present else download
    transfile = os.path.join(datadir, "cmuarctic.data")
    if force or not os.path.isfile(transfile):
        os.system("wget %s -P %s\n" % (TRANSURL, datadir))

    # check cmu arctic bdl data is present else download
    # we are only interested in the wav files in the donwload from bdl
    # other data is prepared by CMU and is useful as comparison data
    wavdir = os.path.join(datadir, "cmu_us_bdl_arctic")
    if force or not os.path.isdir(wavdir):
        os.system("wget %s -P %s\n" % (WAVSURL, datadir))
        os.system("unzip -d %s %s/cmu_us_bdl_arctic-0.95-release.zip" % (datadir, datadir))


def textnorm(datadir, force = False):
    # text norm
    transfile = os.path.join(datadir, TRANSNAME)
    if not os.path.isfile(os.path.join(datadir, "text_norm.xml")):
        fpx = open(os.path.join(datadir, "text.xml"), 'w')
        fpx.write("<document>\n")
        lines = open(transfile).readlines()
        for l in lines:
            print l,
            ll = l.split()
            # remove brackets
            ll = ll[1:-1]
            id = ll[0]
            # remove quotes
            ll[1] = ll[1][1:]
            ll[-1] = ll[-1][:-1]
            text = ' '.join(ll[1:])
            # we could use idlaktxp to normalise text at this point
            # but we will just do some specific global search and replace
            # for this data
            # some numbers
            text = text.replace(' 16,', ' sixteenth,')
            text = text.replace(' 17,', ' seventeenth,')
            text = text.replace(' 18,', ' eighteenth,')
            text = text.replace(' 1908.', ' nineteen oh eight.')
            text = text.replace(' 29th ', ' twenty ninth')
            # replace abbreviations
            text = text.replace('Mr ', ' Mister ')
            text = text.replace('Mrs ', ' Missus ')
            text = text.replace(' etc.', ' etcetera.')
            text = text.replace(' -- ', ', ')
            fpx.write("<fileid id='%s'>\n%s\n</fileid>\n" % (id, text))
            # remove punctuation
            text = text.replace(',', '')
            text = text.replace('.', '')
            text = text.replace('?', '')
            text = text.replace('!', '')
            text = text.replace('-', '')
            # upper case
            text = text.upper()
            # fp.write("%s %s\n" % (id, text))
            print "%s %s" % (id, text)
        fpx.write("</document>\n")
        fpx.close()

        #run text through the idlak text processing module
        com = '../../../src/idlaktxpbin/idlaktxp --pretty --tpdb=../../../idlak-data/en/ga %s %s\n' % (os.path.join(datadir, "text.xml"), os.path.join(datadir, "text_norm.xml"))
        print com
        os.system(com)

def kaldidata(datadir, force=False):
    if force or not os.path.isdir(os.path.join(datadir, "train")):
        os.mkdir(os.path.join(datadir, "train"))
        # load into XML
        p = xml.sax.make_parser()
        handler = saxhandler()
        p.setContentHandler(handler)
        p.parse(open( os.path.join(datadir, "text_norm.xml"),"r"))
        fp = open(os.path.join(datadir, "train", "text"), 'w')
        for i in range(len(handler.ids)):
            fp.write("%s %s\n" % (handler.ids[i], ' '.join(handler.data[i])))
        fp.close()
        #write wav list and utt 2 spk mapping (all same speaker)
        fp = open(os.path.join(datadir, "train", "wav.scp"), 'w')
        fputt2spk = open(os.path.join(datadir, "train", "utt2spk"), 'w')
        for uttid in handler.ids:
            fp.write("%s %s\n" % (uttid, os.path.join(datadir, WAVDIR, uttid + '.wav')))
            fputt2spk.write("%s bdl\n" % (uttid))
        fp.close()
        fputt2spk.close()
        fpoov = open(os.path.join(datadir, "oov.txt"), 'w')
        fplex = open(os.path.join(datadir, "lexicon.txt"), 'w')
        # add oov word and phone (should never be required!
        fplex.write("<OOV> oov\n")
        # write transcription lexicon and oov lexicon for info
        words = handler.lex.keys()
        words.sort()
        phones = {}
        chars = {}
        for w in words:
            prons = handler.lex[w].keys()
            prons.sort()
            utf8w = w.decode('utf8')
            # get all the characters as a check on normalisation
            for c in utf8w:
                chars[c] = 1
            # get phone set from transcription lexicon
            for p in prons:
                pp = p.split()
                for phone in pp:
                    phones[phone] = 1
                fplex.write("%s %s\n" % (w, p))
            if handler.oov.has_key(w):
                fpoov.write("%s %s\n" % (w, prons[0]))
        fplex.close()
        fpoov.close()
        # write phone set
        # Should throw if phone set is not conformant
        # ie. includes sp or ^a-z@
        fp = open(os.path.join(datadir, "nonsilence_phones.txt"), 'w')
        phones = phones.keys()
        phones.sort()
        fp.write('\n'.join(phones) + '\n')
        fp.close()
        # write character set
        fp = open(os.path.join(datadir, "characters.txt"), 'w')
        chars = chars.keys()
        chars.sort()
        fp.write((' '.join(chars)).encode('utf8') + '\n')
        fp.close()
        # silence models
        fp = open(os.path.join(datadir, "silence_phones.txt"), 'w')
        fp.write("sp\nsil\noov\n")
        fp.close()
        # optional silence models
        fp = open(os.path.join(datadir, "optional_silence.txt"), 'w')
        fp.write("sp\n")
        fp.close()
        # an empty file for the kaldi utils/prepare_lang.sh script
        fp = open(os.path.join(datadir, "extra_questions.txt"), 'w')
        fp.close()
        
        
#
# Command line running
#
def main():
    from optparse import OptionParser

    # Default input/output directory
    cwd = os.getcwd()
    usage="Usage: %prog [-h] tpdb_file\n\nDownload and segment arctic speaker bdl."
    parser = OptionParser(usage=usage)

    # set up directories
    datadir = os.path.join(cwd, "data")
    if not os.path.isdir(datadir):
        os.mkdir(datadir)

    # download corpus
    download(datadir)
    
    # build input files for kaldi
    # textnorm
    textnorm(datadir)

    # kaldi input data
    kaldidata(datadir)

    # build a mono model
    os.system('steps/train_mono.sh --nj 1 data/train data/lang exp')

    # extract the alignment
    os.system('../../../src/bin/show-alignments data/lang/phones.txt exp/40.mdl "ark:gunzip -c exp/ali.1.gz|" > align.dat')

    # convert alignment into individual lab files that can be checked by wavesurfer etc.
    os.system('cat align.dat | python align2lab.py labs')
    
if __name__ == '__main__':
    main()
