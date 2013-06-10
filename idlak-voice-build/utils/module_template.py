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

# Template for creating modules for the idlak build system - CHANGE ME

import sys, os.path, time

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
SCRIPT_NAME = os.path.splitext(os.path.split(__file__)[1])[0]
DESCRIPTION = 'Template for idlak build module - CHANGE ME'

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
    build_conf.set_build_environment(SCRIPT_NAME)

    # MODULE SPECIFIC CODE
    # get required input files from idlak-data
    # get required directories from dependent modules
    # examine general settings and set as appropriate
    # process data
    # END OF MODULE SPECIFIC CODE
    
    build_conf.end_processing(SCRIPT_NAME)
    
if __name__ == '__main__':
    main()
    
