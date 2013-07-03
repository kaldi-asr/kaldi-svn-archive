// idlaktxp/txpfexspec.cc

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

#include "./txpfexspec.h"
#include "./fexfunctions.h"

namespace kaldi {

// constructor takes tpdb, loads and sets up features
TxpFexspec::TxpFexspec(TxpConfig * config, const char * type, const char * name)
    : TxpXmlData(config, type, name),
      fexspec_maxfieldlen_(FEXSPEC_MAXFIELDLEN),
      pauhand_(FEXSPECPAU_HAND_SPT) {
  if (!strcmp(FEXSPEC_PAUSEHANDLING, "UTT")) pauhand_ = FEXSPECPAU_HAND_UTT;
}

// parse file into Fexspec class adding feature specification and
// feature functions to architecture
void TxpFexspec::StartElement(const char * name, const char ** atts) {
  std::string att, att2;
  StringSet * set;
  TxpFexspecFeat feat;
  int32 featidx;
  
  // add features and other definitions
  // top level behaviour
  if (!strcmp(name, "fex")) {
    SetAtt("maxfieldlen", atts, &att);
    if (!att.empty()) fexspec_maxfieldlen_ = atoi(att.c_str());
    SetAtt("pausehandling", atts, &att);
    if (!att.empty()) {
      if (!strcmp(att.c_str(), "UTT")) pauhand_ = FEXSPECPAU_HAND_UTT;
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
    if (!att2.empty()) att = FEXSPEC_NULL;
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
    if (att.empty() && fexspecfeats_.size()) {
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
    if (feat.type == FEXSPEC_TYPE_STR) {
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
    } else if (feat.type == FEXSPEC_TYPE_INT) {
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
    fexspecfeatlkp_.insert(LookupIntItem(curfunc_, fexspecfeats_.size()));
    fexspecfeats_.push_back(feat);
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
      iter = fexspecfeatlkp_.find(curfunc_);
      if (iter != fexspecfeatlkp_.end()) {
        fexspecfeats_[iter->second].mapping.insert(LookupItem(att, att2));
      }
    }
  }
}

/// return maximum width in bytes of feature string
int32 TxpFexspec::MaxFeatSz() {
  int32 maxsz = 0;
  TxpFexspecFeatVector::iterator iter;
  // iterate through features and add max size and delimiters
  for(iter = fexspecfeats_.begin(); iter != fexspecfeats_.end(); iter++) {
    maxsz += FEXSPEC_MAXFIELDLEN;
    maxsz += (*iter).delim.size();
  }
  return maxsz;
}

// Add pause tk, syl, phon to break tags and information
// on pause type
// This makes interation for feature extraction more
// homogeneous
int32 TxpFexspec::AddPauseNodes(pugi::xml_document * doc) {
  bool uttbreak, endbreak, internalbreak;
  pugi::xml_node node, childnode;
  pugi::xpath_node_set nodes =
      doc->document_element().select_nodes("//phon|//break");
  nodes.sort();
  for (pugi::xpath_node_set::const_iterator it = nodes.begin();
       it != nodes.end();
       ++it) {
    node = it->node();
    if (!strcmp(node.name(), "break")) {
      // determine break type
      if (!strcmp(node.attribute("type").value(), "4")) uttbreak = true;
      else uttbreak = false;
      if (it == nodes.begin()) {
        endbreak = false;
        internalbreak = false;
      } else if (it + 1 == nodes.end()) {
        endbreak = true;
        internalbreak = false;
      } else if (!strcmp((it - 1)->node().name(), "break")) {
        endbreak = false;
        internalbreak = true;
      } else {
        endbreak = true;
        internalbreak = true;
      }
      if (pauhand_ == FEXSPECPAU_HAND_UTT && !uttbreak &&
          internalbreak && !endbreak) continue;
      childnode = node.append_child("tk");
      childnode.append_attribute("pron").set_value("pau");
      childnode = childnode.append_child("syl");
      childnode.append_attribute("val").set_value("pau");
      childnode = childnode.append_child("phon");
      childnode.append_attribute("val").set_value("pau");
    }
  }
  Output kio("./temp.xml", false);
  doc->save(kio.Stream(), "\t");
  kio.Stream().flush();
  return true;
}

// call the feature functions
bool TxpFexspec::ExtractFeatures(const TxpFexspecContext &context, char * buf) {
  TxpFexspecFeatVector::iterator iter;
  struct TxpFexspecFeat feat;
  bool rval = true;
  // iterate through features inserting nulls when required
  for(iter = fexspecfeats_.begin(); iter != fexspecfeats_.end(); iter++) {
    feat = *iter;
    if (!feat.func(this, &feat, &context, buf))
      rval = false;
  }  
  return rval;
}

/// utility to return index of a feature function
int32 TxpFexspec::GetFeatIndex(const std::string &name) {
  for(int i = 0; i < FEX_NO_FEATURES; i++) {
    if (!strcmp(FEXFUNCLBL[i], name.c_str())) return i;
  }
  return NO_INDEX;
}

bool TxpFexspec::AppendValue(const TxpFexspecFeat &feat, bool error,
                      const char * s, char * buf) const {
  const StringSet * set;
  StringSet::iterator i;
  set = &(sets_.find(feat.set)->second);
  if (set->find(std::string(s)) == set->end()) {
    AppendError(feat, buf);
    return false;
  }
  strncat(buf, feat.delim.c_str(), fexspec_maxfieldlen_);
  strncat(buf, Mapping(feat, s), fexspec_maxfieldlen_);  
  return true;
}

bool TxpFexspec::AppendValue(const TxpFexspecFeat &feat, bool error,
                      int32 i, char * buf) const {
  std::stringstream stream;
  if (feat.type != FEXSPEC_TYPE_INT) {
    AppendError(feat, buf);
    return false;
  }
  if (i < feat.min) i = feat.min;
  if (i > feat.max) i = feat.max;
  stream << i;
  strncat(buf, feat.delim.c_str(), fexspec_maxfieldlen_);
  strncat(buf, Mapping(feat, stream.str().c_str()), fexspec_maxfieldlen_);
  return true;
}

bool TxpFexspec::AppendNull(const TxpFexspecFeat &feat, char * buf) const {
  strncat(buf, feat.delim.c_str(), fexspec_maxfieldlen_);
  strncat(buf, Mapping(feat, feat.nullvalue.c_str()), fexspec_maxfieldlen_);
  return true;
}

bool TxpFexspec::AppendError(const TxpFexspecFeat &feat, char * buf) const {
  strncat(buf, feat.delim.c_str(), fexspec_maxfieldlen_);
  strncat(buf, FEXSPEC_ERROR, fexspec_maxfieldlen_);
  return true;  
}

const char * TxpFexspec::Mapping(const TxpFexspecFeat &feat, const char * instr) const {
  LookupMap::const_iterator i;
  i = feat.mapping.find(std::string(instr));
  if (i != feat.mapping.end()) return i->second.c_str();
  else return instr;
}

TxpFexspecModels::~TxpFexspecModels() {
  int32 i;
  for(i = 0; i < models_.size(); i++) {
    delete models_[i];
  }
}

void TxpFexspecModels::Init(TxpFexspec *fexspec) {
  buflen_ = fexspec->MaxFeatSz() + 1;
}

void TxpFexspecModels::Clear() {
  int32 i;
  for(i = 0; i < models_.size(); i++) {
    delete models_[i];
  }
  models_.clear();
}

char * TxpFexspecModels::Append() {
  char * buf = new char[buflen_];
  memset(buf, 0, buflen_);
  models_.push_back(buf);
  return buf;
}

TxpFexspecContext::TxpFexspecContext(const pugi::xml_document &doc,
                       enum FEXSPECPAU_HAND pauhand) : pauhand_(pauhand) {
  phons_ = doc.document_element().select_nodes("//phon");
  syls_ = doc.document_element().select_nodes("//syl");
  wrds_ = doc.document_element().select_nodes("//tk");
  spts_ = doc.document_element().select_nodes("//spt");
  utts_ = doc.document_element().select_nodes("//utt");
  phons_.sort();
  syls_.sort();
  wrds_.sort();
  spts_.sort();
  utts_.sort();
  cur_phon_ = phons_.begin();
  cur_syl_ = syls_.begin();
  cur_wrd_ = wrds_.begin();
  cur_spt_ = spts_.begin();
  cur_utt_ = utts_.begin();
}

bool TxpFexspecContext::next() {
  pugi::xml_node node, phon, empty;
  
  // std::cout << (cur_phon_->node()).attribute("val").value() << ":"
  //           << (cur_syl_->node()).attribute("val").value() << ":"
  //           << (cur_wrd_->node()).attribute("norm").value() << ":"
  //           << (cur_spt_->node()).attribute("phraseid").value() << ":"
  //           << (cur_utt_->node()).attribute("uttid").value() << " "
  //           << (cur_phon_->node()).attribute("type").value() << "\n";
  // iterate over phone/break items
  cur_phon_ = cur_phon_++;
  phon = cur_phon_->node();
  // update other iterators as required
  // dummy pau items already added for tk and syl levels
  node = getContextUp(phon, "syl");
  while(node != cur_syl_->node()) cur_syl_++;
  node = getContextUp(phon, "tk");
  while(node != cur_wrd_->node()) cur_wrd_++;
  node = getContextUp(phon, "spt");
  while(node != cur_spt_->node()) cur_spt_++;
  node = getContextUp(phon, "utt");
  while(node != cur_utt_->node()) cur_utt_++;
  
  return true;
}

// look up from the node until we find the correct current context node
pugi::xml_node TxpFexspecContext::getContextUp(const pugi::xml_node &node,
                                          const char * name) {
  pugi::xml_node parent;
  parent = node.parent();
  while(!parent.empty()) {
    if (!strcmp(parent.name(), name)) return parent;
    parent = parent.parent();
  }
  return parent;
}

// return phon back or forwards from current phone
pugi::xml_node TxpFexspecContext::getPhon(int32 idx, bool pauctx) const {
  int32 i;
  int32 pau_found = 0;
  pugi::xml_node empty;
  if (idx >= 0) {
    for(i = 0; i < idx; i++) {
      if ((cur_phon_ + i) == phons_.end()) return empty;
      if (!strcmp("pau", (cur_phon_ + i)->node().attribute("val").value()))
        pau_found++;
    }
  }
  else {
    for(i = 0; i > idx; i--) {
      if ((cur_phon_ + i) == phons_.begin()) return empty;
      if (!strcmp("pau", (cur_phon_ + i)->node().attribute("val").value()))
        pau_found++;
    }
  }
  if ((cur_phon_ + i) == phons_.end()) return empty;
  if (pau_found == 2 && !pauctx) return empty;
  return (cur_phon_ + i)->node();
}

}  // namespace kaldi
