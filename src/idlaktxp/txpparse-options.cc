// idlaktxp/txpconfig.cc

// Copyright 2013 CereProc Ltd.  (Author: Matthew Aylett)

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.
//

#include "idlaktxp/txpparse-options.h"
#include "util/text-utils.h"
#include "base/kaldi-common.h"

namespace kaldi {

/// This is the full default configuration for txp. Only keys that
/// have a default value here can be specified in module setup
/// \todo currently nonsense values and modules
const std::string txpconfigdefault =
    "--general-lang=en\n"
    "--general-region=us\n"
    "--general-acc=ga\n"
    "--general-spk=/>\n"
    "--tokenise-processing-mode=lax/>\n"
    "--pauses-hzone=True\n"
    "--pauses hzone-start=45\n"
    "--pauses-hzone-end=100/>\n"
    "--toxparselite max-token-length=30/>\n"
    "--phrasing-max-phrase-length=30\n"
    "--phrasing-by-utterance=True\n"
    "--phrasing-max-utterance-length=10\n"
    "--phrasing-phrase-length-window=10/>\n"
    "--normalise-trace=0\n"
    "--normalise-active=True/>\n"
    "--pronounce-novowel-spell=True/>\n"
    "--syllabify-slang=/>\n"
    "--archiphone-match-all-phones=False/>\n";

TxpParseOptions::TxpParseOptions(const char *usage)
    : ParseOptions(usage) {
  std::istringstream is(txpconfigdefault);
  std::string line, key, value, *stringptr;
  LookupMapItem item;
  RegisterStandard("tpdb", &tpdb_,
                   "Text processing database (directory XML language/speaker files)"); //NOLINT
  while (std::getline(is, line)) {
    // trim out the comments
    size_t pos;
    if ((pos = line.find_first_of('#')) != std::string::npos) {
      line.erase(pos);
    }
    // skip empty lines
    Trim(&line);
    if (line.length() == 0) continue;

    // parse option
    SplitLongArg(line, &key, &value);
    NormalizeArgName(&key);
    Trim(&value);
    stringptr = new std::string(value);
    txpoptions_.insert(LookupItemPtr(key, stringptr));
    Register(key.c_str(), stringptr,
             "Idlak Text Processing Option : See Idlak Documentation");
  }
}

TxpParseOptions::~TxpParseOptions() {
  for (LookupMapPtr::iterator iter = txpoptions_.begin();
      iter != txpoptions_.end(); iter++) {
    delete iter->second;
  }
}

int TxpParseOptions::Read(int argc, const char* argv[]) {
  std::string key, value;
  int i;
  // first pass: look for tpdb parameter
  for (i = 1; i < argc; i++) {
    if (std::strncmp(argv[i], "--", 2) == 0) {
      SplitLongArg(argv[i], &key, &value);
      NormalizeArgName(&key);
      Trim(&value);
      if (key.compare("tpdb") == 0) {
        ReadConfigFile(value + "/default.conf");
      }
      if (key.compare("help") == 0) {
        PrintUsage();
        exit(0);
      }
    }
  }
  return ParseOptions::Read(argc, argv);
}

const char* TxpParseOptions::GetValue(const char* module, const char* key) {
  LookupMapPtr::iterator lookup;
  std::string optkey(module);
  optkey = optkey + "-" + key;
  lookup = txpoptions_.find(optkey);
  if (lookup == txpoptions_.end()) return NULL;
  return (lookup->second)->c_str();
}
const char* TxpParseOptions::GetTpdb() {
  LookupMapPtr::iterator lookup;
  lookup = txpoptions_.find(std::string("tpdb"));
  return (lookup->second)->c_str();
}

}  // namespace kaldi
