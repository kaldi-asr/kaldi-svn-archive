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

import sys, os.path, xml.sax, time, glob
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
    # set up sax parser
    def __init__(self):
        self.p = xml.sax.make_parser()
        self.data = {}
        self.dataseq = []
        self.idlakvoicebuild = os.path.split(SCRIPT_DIR)[0]
        self.logger = None
        self.idlakscratch = None
        self.idlakdata = None
        self.idlakwav = None
        self.kaldidir = None
        self.lang = None
        self.acc = None
        self.spk = None
        self.buildid = None
        self.module = None
        self.outdir = None
    # parse a configuration file
    def parse(self, filename):
        if not os.path.split(filename)[0]:
            filename = os.path.join(SCRIPT_DIR, "..", "build_conf", filename)
        self.p.setContentHandler(self)
        self.p.parse(open(filename, "r"))
    # sax parser call back on start element. Builds data which is dictionary
    # lookup of settings and dataseq which keeps the start elements (modules)
    # in correct order for option information
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
    # get a module key/val pair from configuration
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
    # return directly dependent modules for a module
    def dependent_modules(self, module):
        result = {}
        val = self.getval(module, "depend")
        if val:
            for m in val.split(','):
                result[m] = 1
            return result
        else:
            return {}
    # return all dependent modules for a module   
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
    # update data from command line options (dataseq is left unchanged)
    def updatefromopts(self, opts):
        for m in self.data.keys():
            if self.data[m].has_key('options'):
                for k in self.data[m]['options'].split(','):
                    try:
                        v = eval('opts.' + k)
                    except AttributeError:
                        v = None
                    if v:
                        if type(v) == type(True):
                            self.data[m][k] = 'True'
                        else:
                            self.data[m][k] = v
    # set up logging, check idalk-scratch, run dependent modules as required
    def set_build_environment(self, module):
        # set up logging
        self.logger = Logger(module, self.data)
        # set up useful paths and information
        # scratch
        if not self.data['general']['idlakscratch']:
            self.idlakscratch =  os.path.join(self.idlakvoicebuild, 'idlak-scratch')
        else:
            self.idlakscratch = self.data['general']['idlakscratch']
        # data
        if not self.data['general']['idlakdata']:
            self.idlakdata = os.path.join(self.idlakvoicebuild, '..', 'idlak-data')
        else:
            self.idlakdata = self.data['general']['idlakdata']
        # wavs
        if not self.data['general']['idlakwav']:
            self.idlakwav = os.path.join(self.idlakvoicebuild, '..', 'idlak-data')
        else:
            self.idlakwav = self.data['general']['idlakwav']
        # kaldi root for command line untilities
        if not self.data['general']['kaldidir']:
            self.kaldidir = os.path.join(self.idlakvoicebuild, '..')
        else:
            self.kaldidir = self.data['general']['kaldidir']
        # extract basic speaker configuration
        self.lang = self.data['general']['lang']
        self.acc = self.data['general']['acc']
        self.spk = self.data['general']['spk']
        self.buildid = self.data['general']['buildid']
        # check idlak-scratch and create if required
        self.outdir = os.path.join(self.idlakscratch, self.lang, self.acc, self.spk, module, self.buildid)
        if not os.path.isdir(self.outdir):
            os.makedirs(self.outdir)
        if not os.path.isdir(os.path.join(self.outdir, 'output')):
            os.mkdir(os.path.join(self.outdir, 'output'))
        # if building dependent modules recursively call them
        if self.data['general']['depend'] == 'True':
            dep_modules = self.dependent_modules(module).keys()
            dep_modules.sort()
            for dep in dep_modules:
                if not os.path.isdir(os.path.join(self.idlakscratch, self.lang, self. acc,
                                                  self.spk, dep, self.buildid, 'complete')):
                    com = os.path.join(self.idlakvoicebuild, 'modules', dep + '.py') + ' ' + ' '.join(sys.argv[1:])
                    self.logger.log('info', 'Running dependent: %s' % com)
                    os.system(com)
        # return logger for module code to use
        return self.logger
    # run after the module has completed
    def end_processing(self, module):
        # clean is set to True it will delete ALL contents
        # of module output directory except anything in subdir 'output'
        if self.data['general']['clean'] == 'True':
            contents = glob.glob(os.path.join(self.outdir, '*'))
            contents.sort()
            for c in contents:
                if not os.path.split(c)[1] in ['output', 'complete']:
                    com = 'rm -rf %s' % c
                    self.logger.log('info', 'Cleaning: %s' % c)
                    os.system(com)
        fp = open(os.path.join(self.outdir, 'complete'), 'w')
        fp.write(time.asctime() + '\n')
         
class Logger:
    loglevels = {'none':5, 'critical':4, 'error':3, 'warn':2, 'info':1, 'debug':0}
    curlevel = 0
    module = None
    logfname = None
    nolog = False
    logtofile = False
    logtostderr = False
    # set up a logger
    def __init__(self, module, data):
        self.module = module
        if data['logging']['loglevel']:
            if not self.loglevels.has_key(data['logging']['loglevel'].lower()):
                self.log('Warning', 'Bad log level in configuration: %s' % (level))
            else:
                self.curlevel = self.loglevels[data['logging']['loglevel'].lower()]
        if data['logging']['logtofile'] == 'True':
            if data['logging']['logname']:
                self.logfname = os.path.join(data['logging']['logdir'], data['logging']['logname'])
            else:
                self.logfname = os.path.join(data['logging']['logdir'], data['general']['buildid'] + '.log')
            # try to open file to force an error if not writeable
            fp = open(self.logfname, 'a')
            fp.close()
            self.logtofile = True
        if data['logging']['nolog'] == 'True':
            self.nolog = True
        if data['logging']['logtostderr'] == 'True':
            self.logtostderr = True
        self.log('INFO', 'Started Logging')
    # log a message
    def log(self, level, message):
        # check logging is switched on
        if self.nolog:
            return
        # check valid logging level
        if self.loglevels.has_key(level.lower()):
            # check logging at this level
            if self.loglevels[level.lower()] >= self.curlevel:
                msg = self.module.upper() + '[' + time.asctime() + '] ' + message + '\n'
                if self.logtofile:
                    fp = open(self.logfname, 'a')
                    fp.write(msg)
                    fp.close()
                if self.logtostderr:
                    sys.stderr.write(msg)
        else:
            self.log('Warn', 'Bad log level: %s Message: %s' % (level, message))
            
def get_config(scriptname, description, module):
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
        dep_modules = build_config.dependent_modules(module).keys()
        dep_modules.sort()
        if dep_modules:
            depstring = "Dependencies: %s\n" % ','.join(dep_modules)
        else:
            depstring = "Dependencies: None\n"
    usage = "Usage: %s [-h]\n\n%s" % (scriptname, depstring + description)
    parser = OptionParser(usage=usage)

    # Add all options
    build_config.add_opts(parser, [module] + dep_modules)
    return build_config, parser

    
