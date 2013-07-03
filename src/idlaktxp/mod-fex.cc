// idlaktxp/mod-fex.cc

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

TxpFex::TxpFex(const std::string &tpdb, const std::string &configf)
    : TxpModule("fex", tpdb, configf) {
  fexspec_.Init(&config_, "fex", "default");
  fexspec_.Parse(tpdb.c_str());
}

TxpFex::~TxpFex() {
}

bool TxpFex::Process(pugi::xml_document * input) {
  int modellen;
  char * model;
  modellen = fexspec_.MaxFeatSz() + 1;
  model =  new char[modellen];
  fexspec_.AddPauseNodes(input);
  pugi::xpath_node_set tks =
      input->document_element().select_nodes("//phon");
  tks.sort();
  TxpFexspecContext context(*input,  fexspec_.GetPauseHandling());
  kaldi::int32 i = 0;
  for (pugi::xpath_node_set::const_iterator it = tks.begin();
       it != tks.end();
       ++it, i++, context.next()) {
    pugi::xml_node phon = (*it).node();
    memset(model, 0, modellen);
    fexspec_.ExtractFeatures(context, model);
    phon.text() = model;
  }
  delete [] model;
  // should set to error status
  return true;  
}

bool TxpFex::IsSptPauseHandling() {
  if (fexspec_.GetPauseHandling() == FEXSPECPAU_HAND_SPT) return true;
  return false;
}

}  // namespace kaldi
