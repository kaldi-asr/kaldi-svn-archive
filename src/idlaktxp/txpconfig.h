// idlaktxp/txpconfig.h

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

#ifndef SRC_IDLAKTXP_TXPCONFIG_H_
#define SRC_IDLAKTXP_TXPCONFIG_H_

#include <string>
#include "pugixml.hpp"
#include "base/kaldi-common.h"
#include "./idlak-common.h"
//#include "./txpnrules.h"

/// Name of configuration tag in XML source text
#define TXPCONFIG_SECTION "tpconfig"

namespace kaldi {

/// \addtogroup idlak_utils
/// @{

/// Configurations are merged together with
/// User configurations overriding system configurations
enum TXPCONFIG_LVL {TXPCONFIG_LVL_SYSTEM = 0,
                    TXPCONFIG_LVL_USER = 1};

/// Manages XML configuration files for the idlaktxp system
/// Configurations are in a section tag TXPCONFIG_SECTION
/// And are divided by module into key value pairs with the module
/// as a tag and the key value pairs as name/attribute pairs
///
/// In the prototype very no options are supported but as the system
/// is extended more will be required. An option must exist with a default
/// value specified in txpconfig.cc to be present in either a system wide
/// configuration file or a user configuration file.
class TxpConfig {
 public:
  /// Creates a txp config object with default values for all configuration
  /// settings
  explicit TxpConfig();
  ~TxpConfig() {}
  /// Loads either a system configuration or a user configuration
  bool Parse(enum TXPCONFIG_LVL lvl, const char * config);
  /// Looks for a module/key value, first in the user config,
  /// then the system config and finally in the default config.
  const char * GetValue(const char * module, const char * key);

 private:
  /// Default values hard coded in txpconfig.cc
  pugi::xml_document default_;
  /// System values typically specified in voice data
  pugi::xml_document system_;
  /// User specific values
  pugi::xml_document user_;
};

/// @} end of \addtogroup idlak_utils

}  // namespace kaldi

#endif  // SRC_IDLAKTXP_TXPCONFIG_H_
