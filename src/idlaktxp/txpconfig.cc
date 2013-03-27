// idlaktxp/txpconfig.cc

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

#include "./txpconfig.h"

namespace kaldi {

/// This is the full default configuration for txp. Only keys that
/// have a default value here can be specified in module setup
/// \todo currently nonsense values and modules
const char * txpconfigdefault =
    "<tpconfig>\n"
    "    <general lang='en'\n"
    "         region='us'\n"
    "         acc='ga'\n"
    "         spk=''/>\n"
    "    <tokenise processing_mode='lax'/>\n"
    "    <pauses hzone='True'\n"
    "         hzone_start='45'\n"
    "         hzone_end='100'/>\n"
    "    <toxparselite max_token_length='30'/>\n"
    "    <phrasing max_phrase_length='30'\n"
    "        by_utterance='True'\n"
    "        max_utterance_length='10'\n"
    "        phrase_length_window='10'/>\n"
    "    <normalise trace='0'\n"
    "        active='True'/>\n"
    "    <pronounce novowel_spell='True'/>\n"
    "    <syllabify slang=''/>\n"
    "    <archiphone match_all_phones='False'/>\n"
    "</tpconfig>\n";

TxpConfig::TxpConfig() {
  pugi::xml_parse_result r;
  r = default_.load(txpconfigdefault);
  if (!r) KALDI_ERR << "PugiXML Error parsing internal txp configuration file";
}

bool TxpConfig::Parse(enum TXPCONFIG_LVL lvl, const char * config) {
  std::string fname;
  pugi::xml_parse_result r;
  // input is the system wide configuration XML input file
  if (lvl == TXPCONFIG_LVL_SYSTEM) {
    fname.append(config);
    fname.append("/config.xml");
    r = system_.load_file(fname.c_str());
    if (!r) KALDI_ERR << "PugiXML Error parsing tpdb txp configuration file: "
                      << config;
  // input is a user defined XML input file
  } else {
    r = user_.load_file(config);
    if (!r) KALDI_WARN << "PugiXML Error parsing user txp configuration file: "
                       << config;
  }
  if (!r) return false;
  return true;
}

const char * TxpConfig::GetValue(const char * module, const char * key) {
  std::string xpath_string;
  pugi::xml_node node;
  pugi::xpath_node_set nodeset;
  pugi::xpath_node_set::const_iterator it;
  xpath_string = std::string("/tpconfig/") + module + "[@"+ key + "]" +
      " | tpconfig/" + module + "[@"+ key + "]";
  // Check we have a valid default
  nodeset = default_.document_element().select_nodes(xpath_string.c_str());
  if (nodeset.begin() == nodeset.end()) {
    KALDI_WARN << "Invalid configuration key: " << key << " for module: "
               << module;
    return NULL;
  }
  // Look in user config
  nodeset = user_.document_element().select_nodes(xpath_string.c_str());
  it = nodeset.begin();
  if (it != nodeset.end()) {
    node = (*it).node();
    return node.attribute(key).value();
  }
  // Try system config
  nodeset = system_.document_element().select_nodes(xpath_string.c_str());
  it = nodeset.begin();
  if (it != nodeset.end()) {
    node = (*it).node();
    return node.attribute(key).value();
  }
  // Try default config
  nodeset = default_.document_element().select_nodes(xpath_string.c_str());
  it = nodeset.begin();
  if (it != nodeset.end()) {
    node = (*it).node();
    return node.attribute(key).value();
  }
  return NULL;
}

}  // namespace kaldi
