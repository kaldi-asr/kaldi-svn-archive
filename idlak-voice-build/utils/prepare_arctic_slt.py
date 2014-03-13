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

# This script downloads arctic slt data base and creates the input required for
# the voice building process:
#     wavs files
#     transcription
#     additional lexical items

# By default it will copy data into idlak-data/en/ga/slt

# Idlak filenames have the form <spk>_<g>nnnn_nnn[_nnn].ext
# where:
# spk - a three letter lower case speaker name
# g - a single lower case ascii character which indicates a genre of utterances
# nnnn - a four digit number which can be used for paragraph or utterance set etc.
#    (start at 001)
# nnn - a three digit utterance number
#    (start at 001)
# for utterenaces that have been further split (i.e by extra pauses) an optional
# three digit 'spurt' number (start at 001)

# arctic have the form arctic_a0nnn and arctic_b0nnn which will be remapped to
# slt_a0001_nnn and slt_b0001_nnn

# audio is 16khz and copied to wavdir/16000_orig In general all original corpus audio
# should be copied to such a directory name reflecting sample rate etc.
# a symbolic link is then made between this directory and wavdir/16000 which is always
# the true input to the kaldi voice build system. If audio preprocessing is carried out
# then remove this link and create copies as appropriate.

import os, glob

# get scripts full directory name
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__)) + os.sep
# get the output directory for the transcription and lexical data
DATA_DIR = os.path.realpath(os.path.join(SCRIPT_DIR, '../../idlak-data/en/ga/slt'))
# get the output directory for audio
WAV_DIR = os.path.realpath(os.path.join(SCRIPT_DIR, '../../idlak-data/en/ga/slt'))
# set directory for data generated by script (scratch data) build id set to '00000000'
SCRATCH_DATA_DIR = os.path.realpath(os.path.join(SCRIPT_DIR, '../idlak-scratch/en/ga/slt/prepare_arctic_slt/00000000'))

#URL of arctic transcriptions and sub directory names
TRANSURL = 'http://festvox.org/cmu_arctic/cmuarctic.data'
WAVSURL = 'http://www.speech.cs.cmu.edu/cmu_arctic/packed/cmu_us_slt_arctic-0.95-release.zip'
TRANSNAME = "cmuarctic.data"
SLTWAVDIR = "cmu_us_slt_arctic/wav"

# Create directories as required
def create_dirs():
    if not os.path.isdir(DATA_DIR):
        os.makedirs(DATA_DIR)
    if not os.path.isdir(os.path.join(WAV_DIR, '16000_orig')):
        os.makedirs(os.path.join(WAV_DIR, '16000_orig'))
    if not os.path.isdir(SCRATCH_DATA_DIR):
        os.makedirs(SCRATCH_DATA_DIR)
        
# Download corpus
# force will download whether already present or not
def download(datadir):
    # check cmu arctic transciption file is present else download
    transfile = os.path.join(datadir, "cmuarctic.data")
    os.system("wget %s -P %s\n" % (TRANSURL, datadir))

    # check cmu arctic slt data is present else download
    # we are only interested in the wav files in the donwload from slt
    # other data is prepared by CMU and is useful as comparison data
    wavdir = os.path.join(datadir, "cmu_us_slt_arctic")
    os.system("wget %s -P %s\n" % (WAVSURL, datadir))
    os.system("unzip -d %s %s/cmu_us_slt_arctic-0.95-release.zip" % (datadir, datadir))
    
def process_transcription(scratchdir, datadir):
    # text norm
    transfile = os.path.join(scratchdir, TRANSNAME)
    fpm = open(os.path.join(datadir, "corpusid2idlakid.txt"), 'w')
    fpx = open(os.path.join(datadir, "text.xml"), 'w')
    fpx.write("<document>\n")
    lines = open(transfile).readlines()
    for l in lines:
        print l,
        ll = l.split()
        # remove brackets
        ll = ll[1:-1]
        corpusid = ll[0]
        idlakid = corpusid[7] + '0001_' + corpusid[9:12]
        fpm.write("%s %s\n" % (corpusid, idlakid))
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
        fpx.write("<fileid id='%s'>\n%s\n</fileid>\n" % (idlakid, text))
        print "%s %s" % (idlakid, text)
    fpx.write("</document>\n")
    fpx.close()
    fpm.close()

def copy_wavs(scratchdir, wavdir):
    files = glob.glob(os.path.join(scratchdir, SLTWAVDIR, '*.wav'))
    for f in files:
        stem = os.path.splitext(os.path.split(f)[1])[0]
        newstem =  'slt_' + stem[7] + '0001_' + stem[9:12]
        com = 'cp %s %s/16000_orig/%s.wav' % (f, wavdir, newstem)
        print com
        os.system(com)
    # create a link between original and idlak 16000 directory
    com = 'ln -s %s/16000_orig %s/16000' % (wavdir, wavdir)
    print com
    os.system(com)
            
def main():
    from optparse import OptionParser

    usage="Usage: %prog [-h]\n\nDownload audio and process transcription for arctic speaker slt."
    parser = OptionParser(usage=usage)
    # Options
    parser.add_option("-k", "--keep", dest="keep", action="store_true",
                      help="DO NOT delete scratch data after completing")
    opts, args = parser.parse_args()

    #create directories as required
    create_dirs()

    # Download corpus
    download(SCRATCH_DATA_DIR)

    # Process transcription
    process_transcription(SCRATCH_DATA_DIR, DATA_DIR)

    # Copy wav files
    copy_wavs(SCRATCH_DATA_DIR, WAV_DIR)

    if not opts.keep:
        # Clean up: No data is required so remove all scratch
        com = 'rm -rf %s' % SCRATCH_DATA_DIR
        print 'Clean-up:', com
        os.system(com)
    
if __name__ == '__main__':
    main()
