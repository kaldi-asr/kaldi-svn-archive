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

# Marshals the build options and configuration files for a module

import sys, os.path, xml.sax
from optparse import OptionParser
from optparse import OptionGroup

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

#sort arguments by length and then by alpha
def sort_length_first(a, b):
    if len(a) > len(b):
        return 1
    elif len(a) < len(b):
        return -1
    else:
        if a > b:
            return 1
        elif a < b:
            return -1
        else:
            return 0

class BuildConfig(xml.sax.ContentHandler):
    def __init__(self):
        self.p = xml.sax.make_parser()
        self.data = {}
        self.dataseq = []

    def parse(self, filename):
        if not os.path.split(filename)[0]:
            filename = os.path.join(SCRIPT_DIR, "..", "build_conf", filename)
        self.p.setContentHandler(self)
        self.p.parse(open(filename, "r"))
        
    def startElement(self, name, attrs):
        if name == 'build_config':
            return
        if not self.data.has_key(name):
            self.data[name] = {}
            self.dataseq.append([name, []])
        keys = attrs.keys()
        keys.sort(sort_length_first)
        for k in keys:
            #regularise boolean
            if attrs[k] in ['true', 'TRUE', 'True']:
                att = 'True'
            elif attrs[k] in ['false', 'FALSE', 'False']:
                att = 'False'
            else:
                att = attrs[k]
            self.data[name][k] = att
            self.dataseq[-1][1].append([k, att])

    def getval(self, name, key):
        if self.data.has_key(name):
            if self.data[name].has_key(key):
                return self.data[name][key]
            else:
                return None
        else:
            return None

    # only add options for modules in module list
    def add_opts(self, parser, module_list = None):
        if not module_list:
            module_list = ['general', 'logging']
        else:
            module_list = ['general', 'logging'] + module_list
        for m in self.dataseq:
            if not m[0] in module_list:
                continue
            options = self.getval(m[0], 'options')
            if not options: options = []
            else: options = options.split(',')
            group = None
            if len(options):
                group = OptionGroup(parser, m[0])
            for k, v in m[1]:
                if k[-5:] == '_desc':
                    continue
                if not (k in options):
                    continue
                desc = self.getval(m[0], k + '_desc')
                if not desc:
                    desc = 'Refer to Idlak voice build documentation'
                if v == 'True' or v == 'False':
                    #print v, k
                    group.add_option("--" + k, dest=k, action="store_true",
                                      help=desc)
                else:
                    #print v, k
                    group.add_option("--" + k, dest=k, default=None,
                                      help=desc)
            if group:
                parser.add_option_group(group)
    
    def dependent_modules(self, module):
        result = {}
        val = self.getval(module, "depend")
        if val:
            for m in val.split(','):
                result[m] = 1
            return result
        else:
            return {}
        
    def dependent_modules_all(self, module):
        dep =  self.dependent_modules(module)
        if not dep:
            return {}
        else:
            result = {}
            for m in dep.keys():
                result[m] = 1
                new_deps = self.dependent_modules_all(m)
                for ndep in new_deps.keys():
                    result[ndep] = 1
            return result
        
    def dependent_modules_list(self, module_list):
        result = {}
        for m in module_list:
            mresult = self.dependent_modules_all(m)
            for ndep in mresult.keys():
                result[ndep] = 1
        return result        

def get_config(scriptname, description, module_list):
    build_config = BuildConfig()
    build_config.parse(os.path.join(SCRIPT_DIR, "..", "build_conf",
                                    "default.xml"))
    
    # check to see if dependencies are switched on
    depend = False
    for arg in sys.argv:
        if arg == '--depend' or arg == '--force':
            depend = True
    # if depend on get all dependent modules for options
    depstring = ""
    dep_modules = []
    if depend:
        dep_modules = build_config.dependent_modules_list(module_list).keys()
        dep_modules.sort()
        if dep_modules:
            depstring = "Dependencies: %s\n" % ','.join(dep_modules)
        else:
            depstring = "Dependencies: None\n"
    usage = "Usage: %s [-h]\n\n%s" % (scriptname, depstring + description)
    parser = OptionParser(usage=usage)

    # Add all options
    build_config.add_opts(parser, module_list + dep_modules)
    return build_config, parser

    
