// idlaktxp/mod-postag.cc

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

TxpPosTag::TxpPosTag(const std::string &tpdb, const std::string &configf)
    : TxpModule("postag", tpdb, configf), tagger_("postags", "default") {
  tagger_.Parse(tpdb.c_str());
}

TxpPosTag::~TxpPosTag() {
}

bool TxpPosTag::Process(pugi::xml_document * input) {
  const char * ptag = "", * tag;
  pugi::xpath_node_set tks = input->document_element().select_nodes("//tk");
  tks.sort();
  for (pugi::xpath_node_set::const_iterator it = tks.begin();
       it != tks.end();
       ++it) {
    pugi::xml_node node = (*it).node();
    if (it == tks.begin()) {
      ptag = "#";
    } else {
      pugi::xml_node pnode = (*(it - 1)).node();
      ptag = pnode.attribute("pos").value();
    }
    tag = tagger_.GetPos(ptag, node.attribute("norm").value());
    if (!node.attribute("pos")) node.append_attribute("pos");
    node.attribute("pos").set_value(tag);
  }
  return true;
}

}  // namespace kaldi
