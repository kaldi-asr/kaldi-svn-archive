// idlaktxp/mod-syllabify.cc

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

TxpSyllabify::TxpSyllabify(const std::string &tpdb, const std::string &configf)
    : TxpModule("syllabify", tpdb, configf), sylmax_("sylmax", "default") {
  sylmax_.Parse(tpdb.c_str());
}

TxpSyllabify::~TxpSyllabify() {
}

bool TxpSyllabify::Process(pugi::xml_document * input) {
  std::string sylpron;
  PhoneVector pvector;
  pugi::xpath_node_set spts = input->document_element().select_nodes("//spt");
  spts.sort();
  for (pugi::xpath_node_set::const_iterator it = spts.begin();
       it != spts.end();
       ++it) {
    pugi::xml_node pre_node;
    pugi::xml_node spt = (*it).node();
    pugi::xpath_node_set tks = spt.select_nodes("descendant::tk");
    tks.sort();
    for (pugi::xpath_node_set::const_iterator it2 = tks.begin();
         it2 != tks.end();
         ++it2) {
      pugi::xml_node node = (*it2).node();
      if (!node.attribute("pron").empty())
        sylmax_.GetPhoneVector(node.attribute("pron").value(), &pvector);
      if (it2 == tks.end() - 1) pvector[pvector.size() - 1].cross_word = false;
      sylmax_.Maxonset(&pvector);
      if (it2 != tks.begin()) {
        sylmax_.Writespron(&pvector, &sylpron);
        pre_node.append_attribute("spron").set_value(sylpron.c_str());
      }
      pre_node = node;
    }
    if (!pre_node.empty()) {
      sylmax_.Writespron(&pvector, &sylpron);
      pre_node.append_attribute("spron").set_value(sylpron.c_str());
    }
  }
  return true;
}

}  // namespace kaldi
