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

# prepares data created using cex_def voice build to build an HTS voice
# using HTSDEMO installed with  install_HTS_demo.sh

# WARNING: If HTS demo changes this will not work and will need to be adapted

import sys, os.path, time, glob, re

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
SCRIPT_NAME = os.path.splitext(os.path.split(__file__)[1])[0]
DESCRIPTION = 'Takes output from cex_def and alters HTSDEMO to build voices from it'

# If this question isn't in the main qset the gv clustering will fail
# BAD to have this hard coded here MA070314
UTTQSET = ['C-Phrase_Num-Words']

# Add to path
sys.path = sys.path + [SCRIPT_DIR + '/../utils']
sys.path = sys.path + [SCRIPT_DIR]

# import voice build utilities
import build_configuration

def main():
    # process the options based on the default build configuration
    build_conf, parser = build_configuration.get_config(SCRIPT_NAME, DESCRIPTION, SCRIPT_NAME)
    # parse commamd line
    if __name__ == '__main__':
        opts, args = parser.parse_args()
        # and load custom configurations
        if opts.bldconf:
            build_conf.parse(opts.bldconf)
        if opts.spkconf:
            build_conf.parse(opts.spkconf)
        else:
            parser.error("Speaker configuration is required e.g. speaker_conf/bdl.xml")
            
        build_conf.updatefromopts(opts)
    # set up logging, check idlak-scratch, check dependencies and build as required
    logger = build_conf.set_build_environment(SCRIPT_NAME)

    # MODULE SPECIFIC CODE
    # get required input files from idlak-data
    # text for testing voice after build (same as original HTSDEMO test text
    alicetxtfile = os.path.join(build_conf.idlakdata, build_conf.lang, 'testdata', 'alice.xml')
    # Maping from original arctic corpus id to idlak corpus id
    corpusid2idlakidfile = os.path.join(build_conf.idlakdata, build_conf.lang,
                                        build_conf.acc, build_conf.spk, 'corpusid2idlakid.txt')
    # get required directories from dependent modules
    cexdir = build_conf.get_input_dir('cex_def')
    # examine general settings and set as appropriate
    htsdemodir = build_conf.getval('hts_test', 'htsdemodir')
    if not os.path.isdir(htsdemodir):
        logger.log('critical', '[%s] does not exist' % (htsdemodir))
    if not os.path.isdir(os.path.join(htsdemodir, 'HTS-demo_CMU-ARCTIC-SLT')):
        logger.log('critical', '[%s] does not contain an HTS demo' % (htsdemodir))
    if not build_conf.spk == 'slt':
        logger.log('critical', 'This test currently only setup to work with en/ga/slt')
    # get info to cut slt raw audio in HTSDEMO into spurts
    spttimesfile = os.path.join(cexdir, 'spt_times.dat')
    # get directory for full model files
    htsmodeldir = os.path.join(cexdir, 'htslab')
    # new question set for data
    qsetfile = os.path.join(cexdir, 'questions-kaldi-en-ga.hed')
    # process data

    # create or replace label file directories
    htsdatadir = os.path.join(htsdemodir, 'HTS-demo_CMU-ARCTIC-SLT', 'data')
    # full models
    if os.path.isdir(os.path.join(htsdatadir, 'labels', 'full')):
        if not os.path.isdir(os.path.join(htsdatadir, 'labels', 'full_orig')):
            os.system('mv %s %s' % (os.path.join(htsdatadir, 'labels', 'full'),
                                    os.path.join(htsdatadir, 'labels', 'full_orig')))
            os.mkdir(os.path.join(htsdatadir, 'labels', 'full'))
    else:
        os.mkdir(os.path.join(htsdatadir, 'labels', 'full'))
    # mono models
    if os.path.isdir(os.path.join(htsdatadir, 'labels', 'mono')):
        if not os.path.isdir(os.path.join(htsdatadir, 'labels', 'mono_orig')):
            os.system('mv %s %s' % (os.path.join(htsdatadir, 'labels', 'mono'),
                                    os.path.join(htsdatadir, 'labels', 'mono_orig')))
            os.mkdir(os.path.join(htsdatadir, 'labels', 'mono'))
    else:
        os.mkdir(os.path.join(htsdatadir, 'labels', 'mono'))
    # label files compatible with wavesurfer
    if not os.path.isdir(os.path.join(htsdatadir, 'labels', 'wsurf')):
        os.mkdir(os.path.join(htsdatadir, 'labels', 'wsurf'))
    # create full, mono and wavesurfer label files
    labfiles = glob.glob(htsmodeldir + "/*.lab")
    labfiles.sort()
    for f in labfiles:
        stem = os.path.split(f)[1]
        fp1 = open(os.path.join(htsdatadir, 'labels', 'full', 'cmu_us_arctic_' + stem), 'w')
        fp2 = open(os.path.join(htsdatadir, 'labels', 'mono', 'cmu_us_arctic_' + stem), 'w')
        fp3 = open(os.path.join(htsdatadir, 'labels', 'wsurf', 'cmu_us_arctic_' + stem), 'w')
        for l in open(f).readlines():
            fp1.write(l)
            pat = re.match('^([0-9]+)\s+([0-9]+)\s\S+\-(.*?)\+.*$', l)
            fp2.write("%s %s %s\n" % pat.groups())
            fp3.write("%.3f %.3f %s\n" % (float(pat.group(1))/10000000.0,
                                         float(pat.group(2))/10000000.0,
                                         pat.group(3)))
        fp1.close()
        fp2.close()
        fp3.close()
    # copy question file
    oldqset = os.path.join(htsdatadir, 'questions', 'questions_qst001.hed')
    olduttqset = os.path.join(htsdatadir, 'questions', 'questions_utt_qst001.hed')
    if not os.path.isfile(oldqset + '.orig'):
        os.system('mv %s %s.orig' % (oldqset, oldqset))
    if not os.path.isfile(olduttqset + '.orig'):
        os.system('mv %s %s.orig' % (olduttqset, olduttqset))
    os.system('cp %s %s' % (qsetfile, oldqset))
    # construct utterance qset from qset
    lines = open(qsetfile).readlines()
    fp = open(olduttqset, 'w')
    for l in lines:
        uttqs = False
        for name in UTTQSET:
            if l.find(name) > -1:
                uttqs = True
                break
        if uttqs:
            fp.write(l)
    # cut up audio to correct spt sized chunks
    if not os.path.isdir(os.path.join(htsdatadir, 'kaldiraw')):
        os.system('mv %s %s.orig' % (os.path.join(htsdatadir, 'raw'),
                                     os.path.join(htsdatadir, 'raw')))
        
        os.mkdir(os.path.join(htsdatadir, 'kaldiraw'))
        os.system('ln -s %s %s' % (os.path.join(htsdatadir, 'kaldiraw'),
                                   os.path.join(htsdatadir, 'raw')))
    # load lookup between arcti ids and kaldi ids
    idlak2corpus = {}
    lines = open(corpusid2idlakidfile).readlines()
    for l in lines:
        ll = l.split()
        idlak2corpus[ll[1]] = ll[0]
    # open spt times
    lines = open(spttimesfile).readlines()
    for l in lines:
        ll = l.split()
        origwav = 'cmu_us_arctic_slt_' +  idlak2corpus[ll[0][4:-8]].split('_')[1]
        # currently use ch_wave change to kaldi style MA070314
        cmd = 'ch_wave -o %s/cmu_us_arctic_%s.raw -f 48000 -itype raw -otype raw -start %s -end %s %s/%s.raw' % (
            os.path.join(htsdatadir, 'kaldiraw'),
            ll[0][:-4],
            ll[1], ll[2], os.path.join(htsdatadir, 'raw.orig'), origwav)
        print cmd
        os.system(cmd)
    #TODO create gen labels using script in utils
    # END OF MODULE SPECIFIC CODE
    
    build_conf.end_processing(SCRIPT_NAME)
    
if __name__ == '__main__':
    main()
    

