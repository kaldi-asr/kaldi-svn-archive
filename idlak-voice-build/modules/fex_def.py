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

# Template for creating modules for the idlak build system

import sys, os.path, time, subprocess
from xml.dom.minidom import parse, parseString

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
SCRIPT_NAME = os.path.splitext(os.path.split(__file__)[1])[0]
DESCRIPTION = 'Full model context creation (linguistic feature extraction)'

# Add to path
sys.path = sys.path + [SCRIPT_DIR + '/../utils']
sys.path = sys.path + [SCRIPT_DIR]

# import voice build utilities
import build_configuration

def main():
    # process the options based on the default build configuration
    build_conf, parser = build_configuration.get_config(SCRIPT_NAME, DESCRIPTION, SCRIPT_NAME)
    #print 'SEQ', build_conf.dataseq
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
    build_conf.set_build_environment(SCRIPT_NAME)

    # ADD MODULE SPECIFIC CODE HERE
    # get required input files from idlak-data
    tpdbdir = os.path.join(build_conf.idlakdata, build_conf.lang, build_conf.acc)
    # get required directories from dependent modules
    aligndir = build_conf.get_input_dir('align_def')
    # examine modulespecific settings and set as appropriate
    # process data
    # get path to fexbin and to txpbin
    pathlist = [os.path.join(build_conf.kaldidir, 'src', 'idlakfexbin'),
                os.path.join(build_conf.kaldidir, 'src', 'idlaktxpbin')]
    os.environ["PATH"] += os.pathsep + os.pathsep.join(pathlist)
    # open the aligner xml output with minidom
    dom = parse(os.path.join(aligndir, "text.xml" ))
    # open output file
    fexfp = open(os.path.join(build_conf.outdir, 'output', 'fex.dat'), 'w')
    # get each utterance
    for node in dom.getElementsByTagName('fileid'):
        fexfp.write(node.getAttribute('id').encode('utf8') + '\n')
        normpipe = subprocess.Popen(["idlaktxp", "--pretty", "--tpdb=%s" % (tpdbdir), "-", "-"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        normtext = normpipe.communicate(input=node.toxml())[0]
        normpipe.stdout.close()
        fexpipe = subprocess.Popen(["idlakfex", "--tpdb=%s" % (tpdbdir), "-", "-"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        fex = fexpipe.communicate(input=normtext)[0]
        fexpipe.stdout.close()
        #output = fexpipe.communicate()[0]
        fexfp.write(fex.encode('utf8'))
    # END OF MODULE SPECIFIC CODE
    
    build_conf.end_processing(SCRIPT_NAME)

if __name__ == '__main__':
    main()
    
