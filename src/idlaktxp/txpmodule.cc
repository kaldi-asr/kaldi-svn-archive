// idlaktxp/toxmodule.cc

// Copyright 2012 CereProc Ltd.  (Author: Matthew Aylett)

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

#include "./txpmodule.h"

namespace kaldi {

TxpModule::TxpModule(const std::string &name, const std::string &tpdb,
                     const std::string &configf)
    : name_(name), tpdb_(tpdb), configf_(configf) {
  // Load configuration
  config_.Parse(TXPCONFIG_LVL_SYSTEM, tpdb.c_str());
  if (!configf.empty()) config_.Parse(TXPCONFIG_LVL_USER, configf.c_str());
}

const char * TxpModule::GetConfigValue(const char * key) {
  return config_.GetValue(name_.c_str(), key);
}

// True/true/TRUE -> true, False, false, FALSE, anything else -> false
bool TxpModule::GetConfigValueBool(const char * key) {
  const char * val;
  val = config_.GetValue(name_.c_str(), key);
  if (!strcmp(val, "True") || !strcmp(val, "true") || !strcmp(val, "TRUE"))
    return true;
  return false;
}

}  // namespace kaldi
