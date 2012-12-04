// idlaktxp/txplexicon.h

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

#include "./txplexicon.h"

namespace kaldi {

void TxpLexicon::StartElement(const char * name, const char ** atts) {
  if (!strcmp(name, "lex")) {
    inlex_ = true;
    SetAtt("entry", atts, &entry_);
    SetAtt("pron", atts, &pron_);
    SetAtt("default", atts, &isdefault_);
    word_ = "";
  }
}

void TxpLexicon::CharHandler(const char * data, int32 len) {
  if (inlex_) {
    word_ = word_ + std::string(data, len);
  }
}

void TxpLexicon:: EndElement(const char * name) {
  LookupMap::iterator it;
  if (!strcmp(name, "lex")) {
    inlex_ = false;
    if (isdefault_ == "true") {
      lookup_.insert(LookupItem(std::string("default:") + word_, pron_));
      lookup_.insert(LookupItem(entry_ + std::string(":") + word_, pron_));
    } else {
      it =  lookup_.find(entry_ + std::string(":") + word_);
      if (it == lookup_.end()) {
        lookup_.insert(LookupItem(entry_ + std::string(":") + word_, pron_));
      }
    }
  }
}

int TxpLexicon::GetPron(const std::string &word,
                        const std::string &entry,
                        TxpLexiconLkp &lkp) {
  LookupMap::iterator it;
  if (!entry.empty())
    it =  lookup_.find(entry + std::string(":") + word);
  else
    it =  lookup_.find(std::string("default:") + word);
  if (it != lookup_.end()) {
    lkp.pron += it->second;
    return true;
  } else {
    return false;
  }
}

}  // namespace kaldi
