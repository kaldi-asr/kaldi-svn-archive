#!/usr/bin/python

# Copyright 2013 University of Edinburgh  (Author: Richard Williams)

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

# duration_def.py - Provides phone duration

import sys, os.path, time, glob

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
SCRIPT_NAME = os.path.splitext(os.path.split(__file__)[1])[0]
DESCRIPTION = 'Provides phone duration'

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
    aligndir = build_conf.get_input_dir('align_def')
    outdir = os.path.join(build_conf.outdir, 'output')
    # examine general settings and set as appropriate
    # process data
    wrds_dir = os.path.join(aligndir, 'wrds')

    file_list = glob.glob('%s/*.wrd' % (wrds_dir))

    for f in file_list:
        wrd_file = open(f, 'r')
        # Get the input file's name stem so we can use it for the output filename
        filename_stem = os.path.split(os.path.splitext(f)[0])[1]
        output_filename = os.path.join(outdir, '%s.dur' % (filename_stem))
        output_file = open(output_filename, 'w')

        for line in wrd_file:
          columns = line.split()
          # column #0 is the alotted time prior to the given phone.
          # column #1 is the alotted time after the given phone.
          phone_dur = float(columns[1]) - float(columns[0])
          output_file.write(str(phone_dur) + '\n')
        wrd_file.close()
        output_file.close()
    # END OF MODULE SPECIFIC CODE
    
    build_conf.end_processing(SCRIPT_NAME)
    
if __name__ == '__main__':
    main()
    
