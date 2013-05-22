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

import sys, os.path, re

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
SCRIPT_NAME = os.path.splitext(os.path.split(__file__)[1])[0]
DESCRIPTION = 'Template for idlak build module'

# Add to path
sys.path = sys.path + [SCRIPT_DIR + '/../utils']
sys.path = sys.path + [SCRIPT_DIR]

# import utilities
import build_configuration


def main():
    # process the options based on the default build configuration
    build_conf, parser = build_configuration.get_config(SCRIPT_NAME, DESCRIPTION, ['fex_def'])
    #print 'SEQ', build_conf.dataseq
    # parse commamd line
    if __name__ == '__main__':
        opts, args = parser.parse_args()
        # and load custom configurations
        #if opts.bldconf:
        #    build_conf.parse(opts.bldconf)
        #if opts.spkconf:
        #    build_conf.parse(opts.spkconf)
        #build_conf.updatefromopts(parser)
    
    #print build_conf.dataseq
    # check dependencies and build as required
    # get required input files from idlak-data
    # get required directories from dependent modules
    # examine general settings and set as appropriate
    # process data
    # cleanup
    


if __name__ == '__main__':
    main()
    
