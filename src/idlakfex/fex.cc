// idlakfex/fex.cc

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

#include "./fex.h"
#include "./fexfunctions.h"

namespace kaldi {

// constructor takes tpdb loads and sets up features
Fex::Fex(const char * tpdb, const char * architecture)
    : TxpXmlData("fex", architecture),
      fex_maxfieldlen_(FEX_MAXFIELDLEN),
      pauhand_(FEXPAU_HAND_SPT) {
  if (!strcmp(FEX_PAUSEHANDLING, "UTT")) pauhand_ = FEXPAU_HAND_UTT;
  Parse(tpdb);
}

// parse file into Fex class adding feature specification and
// feature functions to architecture
void Fex::StartElement(const char * name, const char ** atts) {
  std::string att, att2;
  StringSet * set;
  FexFeat feat;
  int32 featidx;
  
  // add features and other definitions
  // top level behaviour
  if (!strcmp(name, "fex")) {
    SetAtt("maxfieldlen", atts, &att);
    if (!att.empty()) fex_maxfieldlen_ = atoi(att.c_str());
    SetAtt("pausehandling", atts, &att);
    if (!att.empty()) {
      if (!strcmp(att.c_str(), "UTT")) pauhand_ = FEXPAU_HAND_UTT;
    }
    // sets for string based feature functions
  } else if (!strcmp(name, "set")) {
    SetAtt("name", atts, &att);
    if (att.empty()) {
      // throw an error
      KALDI_WARN << "Badly formed fex set: "
                 << " Line: " << ::XML_GetCurrentLineNumber(parser_)
                 << " Col:" << XML_GetCurrentColumnNumber(parser_);
      return;
    }
    curset_ = att;
    SetAtt("null", atts, &att);
    if (!att2.empty()) att = FEX_NULL;
    sets_.insert(LookupMapSetItem(curset_, StringSet()));
    setnull_.insert(LookupItem(curset_, att));
    // features
  } else if (!strcmp(name, "item")) {
    SetAtt("name", atts, &att);
    if (att.empty()) {
      // throw an error
      KALDI_WARN << "Badly formed fex set item: "
                 << " Line: " << ::XML_GetCurrentLineNumber(parser_)
                 << " Col:" << XML_GetCurrentColumnNumber(parser_);
      return;
    }
    set = &(sets_.find(curset_)->second);
    set->insert(att);
  } else if (!strcmp(name, "feat")) {
    SetAtt("name", atts, &att);
    if (att.empty()) {
      // throw an error
      KALDI_WARN << "Badly formed fex set item: "
                 << " Line: " << ::XML_GetCurrentLineNumber(parser_)
                 << " Col:" << XML_GetCurrentColumnNumber(parser_);
      return;
    }
    featidx = GetFeatIndex(att);
    if (featidx == NO_INDEX) {
      curfunc_ = "";
      // throw an error
      KALDI_WARN << "Missing feature in architecture: " << att
                 << " Line: " << ::XML_GetCurrentLineNumber(parser_)
                 << " Col:" << XML_GetCurrentColumnNumber(parser_);
      return;
    }
    // valid function try to add rest of the features specification
    curfunc_ = att;
    SetAtt("htsname", atts, &att);
    feat.htsname = att;
    SetAtt("desc", atts, &att);
    feat.desc = att;
    SetAtt("delim", atts, &att);
    if (att.empty() && fexfeats_.size()) {
      // throw an error
      KALDI_WARN << "Missing delimiter for feature architecture: " << curfunc_
                 << " Line: " << ::XML_GetCurrentLineNumber(parser_)
                 << " Col:" << XML_GetCurrentColumnNumber(parser_);
      return;
    }
    feat.delim = att;
    feat.func = FEXFUNC[featidx];
    feat.pauctx = false;
    SetAtt("pauctx", atts, &att);
    if (att == "true" || att == "True" || att == "TRUE") feat.pauctx = true;
    SetAtt("pauctx", atts, &att2);
    feat.pautype = FEXFUNCPAUTYPE[featidx];
    feat.type = FEXFUNCTYPE[featidx];
    if (feat.type == FEX_TYPE_STR) {
      SetAtt("set", atts, &att);
      if (att.empty()) {
        // throw an error
        KALDI_WARN << "Missing set name for string feature architecture: " << curfunc_
                   << " Line: " << ::XML_GetCurrentLineNumber(parser_)
                   << " Col:" << XML_GetCurrentColumnNumber(parser_);
        return;
      }
      // check set has been added
      if (sets_.find(att) == sets_.end()) {
        // throw an error
        KALDI_WARN << "Missing set for string feature architecture: " << curfunc_
                   << " Line: " << ::XML_GetCurrentLineNumber(parser_)
                   << " Col:" << XML_GetCurrentColumnNumber(parser_)
                   << " (must define before function)";
        return;
      }
      feat.set = att;
      feat.nullvalue = setnull_.find(att)->second;
    } else if (feat.type == FEX_TYPE_INT) {
      SetAtt("min", atts, &att);
      if (!att.empty()) feat.min = atoi(att.c_str());
      SetAtt("max", atts, &att);
      if (!att.empty()) {
        // throw an error
        KALDI_WARN << "Missing maximum value for integer feature architecture: " << curfunc_
                   << " Line: " << ::XML_GetCurrentLineNumber(parser_)
                   << " Col:" << XML_GetCurrentColumnNumber(parser_);
        return;
      }
      feat.max = atoi(att.c_str());
    }
    // if we have got to here the feature is valid add it to the architecture
    fexfeatlkp_.insert(LookupIntItem(curfunc_, fexfeats_.size()));
    fexfeats_.push_back(feat);
  } else if (!strcmp(name, "mapping")) {
    SetAtt("fromstr", atts, &att);
    SetAtt("tostr", atts, &att2);
    if (att.empty() || att2.empty()) {
      // throw an error
      KALDI_WARN << "bad mapping item: " << curfunc_
                 << " Line: " << ::XML_GetCurrentLineNumber(parser_)
                 << " Col:" << XML_GetCurrentColumnNumber(parser_);
      return;
    }
    if (!curfunc_.empty()) {
      LookupInt::iterator iter;
      iter = fexfeatlkp_.find(curfunc_);
      if (iter != fexfeatlkp_.end()) {
        fexfeats_[iter->second].mapping.insert(LookupItem(att, att2));
      }
    }
  }
}

/// return maximum width in bytes of feature string
int32 Fex::MaxFeatSz() {
  int32 maxsz = 0;
  FexFeatVector::iterator iter;
  // iterate through features and add max size and delimiters
  for(iter = fexfeats_.begin(); iter != fexfeats_.end(); iter++) {
    maxsz += FEX_MAXFIELDLEN;
    maxsz += (*iter).delim.size();
  }
  return maxsz;
}

// TODO set up a context class for the feature functions to access
int32 Fex::GetModels(const pugi::xml_document &doc, FexModels * models) {
  pugi::xpath_node_set tks =
      doc.document_element().select_nodes("//phon|//break");
  tks.sort();
  kaldi::int32 i = 0;
  for (pugi::xpath_node_set::const_iterator it = tks.begin();
       it != tks.end();
       ++it, i++) {
    pugi::xml_node tk = (*it).node();
    ExtractFeatures(tks, tk, i, models->Append());
  }
  // should set to error status
  return true;
}

// call feature functions and deal with pause behaviour
bool Fex::ExtractFeatures(pugi::xpath_node_set tks,  pugi::xml_node tk,
                          int32 idx, char * buf) {
  FexFeatVector::iterator iter;
  struct FexFeat feat;
  bool rval = true;;
  bool endbreak;
  bool internalbreak;
  // check pause status
  if (!strcmp(tk.name(), "break")) {
    if (!idx) {
      endbreak = false;
      internalbreak = false;
    } else if (idx + 1 == tks.size()) {
      endbreak = true;
      internalbreak = false;
    } else if (!strcmp(tks[idx - 1].node().name(), "break")) {
      endbreak = false;
      internalbreak = true;
    } else {
      endbreak = true;
      internalbreak = true;
    } 
  }
  // iterate through features inserting nulls when required
  for(iter = fexfeats_.begin(); iter != fexfeats_.end(); iter++) {
    feat = *iter;
    if (!strcmp(tk.name(), "phon")) {
      if (!feat.func(this, feat, tks, idx, tk, buf)) rval = false;
    } else if (!strcmp(tk.name(), "break")) {
      // if ! break index 4 and pause handling utterance based
      // ignore document internal second break
      if (feat.pautype == FEXPAU_TYPE_CUR) {
        AppendNull(feat, buf);
      } else if (feat.pautype == FEXPAU_TYPE_PRE) {
        if (endbreak && feat.pauctx) {
          if (!feat.func(this, feat, tks, idx, tk, buf)) rval = false;
        } else {
        AppendNull(feat, buf);
        }
      } else if (feat.pautype == FEXPAU_TYPE_PST) {
        if (!endbreak && feat.pauctx) {
          if (!feat.func(this, feat, tks, idx, tk, buf)) rval = false;
        } else {
            // if pause handling is by utterance use the subsequent
            // internal break to get pst context features
            if (pauhand_ == FEXPAU_HAND_UTT &&
                strcmp(tk.attribute("type").value(), "4") &&
                internalbreak) {
              if (!feat.func(this, feat, tks, idx, tks[idx + 1].node(), buf))
                rval = false;
            } else {
              AppendNull(feat, buf);
            }
        }
      }
    }
  }  
  return rval;
}

/// utility to return index of a feature function
int32 Fex::GetFeatIndex(const std::string &name) {
  for(int i = 0; i < FEX_NO_FEATURES; i++) {
    if (!strcmp(FEXFUNCLBL[i], name.c_str())) return i;
  }
  return NO_INDEX;
}

bool Fex::AppendValue(const FexFeat &feat, bool error,
                      const char * s, char * buf) {
  StringSet * set;
  StringSet::iterator i;
  set = &(sets_.find(feat.set)->second);
  if (set->find(std::string(s)) == set->end()) {
    AppendError(feat, buf);
    return false;
  }
  strncat(buf, feat.delim.c_str(), fex_maxfieldlen_);
  strncat(buf, Mapping(feat, s), fex_maxfieldlen_);  
  return true;
}

bool Fex::AppendValue(const FexFeat &feat, bool error,
                      int32 i, char * buf) {
  std::stringstream stream;
  if (feat.type != FEX_TYPE_INT) {
    AppendError(feat, buf);
    return false;
  }
  if (i < feat.min) i = feat.min;
  if (i > feat.max) i = feat.max;
  stream << i;
  strncat(buf, feat.delim.c_str(), fex_maxfieldlen_);
  strncat(buf, Mapping(feat, stream.str().c_str()), fex_maxfieldlen_);
  return true;
}

bool Fex::AppendNull(const FexFeat &feat, char * buf) {
  strncat(buf, feat.delim.c_str(), fex_maxfieldlen_);
  strncat(buf, Mapping(feat, feat.nullvalue.c_str()), fex_maxfieldlen_);
  return true;
}

bool Fex::AppendError(const FexFeat &feat, char * buf) {
  strncat(buf, feat.delim.c_str(), fex_maxfieldlen_);
  strncat(buf, FEX_ERROR, fex_maxfieldlen_);
  return true;  
}

const char * Fex::Mapping(const FexFeat &feat, const char * instr) {
  LookupMap::const_iterator i;
  i = feat.mapping.find(std::string(instr));
  if (i != feat.mapping.end()) return i->second.c_str();
  else return instr;
}

FexModels::FexModels(Fex *fex) {
  buflen_ = fex->MaxFeatSz() + 1;
}

FexModels::~FexModels() {
  int32 i;
  for(i = 0; i < models_.size(); i++) {
    delete models_[i];
  }
}

void FexModels::Clear() {
  int32 i;
  for(i = 0; i < models_.size(); i++) {
    delete models_[i];
  }
  models_.clear();
}

char * FexModels::Append() {
  char * buf = new char[buflen_];
  memset(buf, 0, buflen_);
  models_.push_back(buf);
  return buf;
}

}  // namespace kaldi
