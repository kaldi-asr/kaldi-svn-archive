// idlaktxp/txpxmldata.cc

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

#include "./txpxmldata.h"

namespace kaldi {

TxpXmlData::TxpXmlData(const char * type, const char * name)
    : type_(type), name_(name) {
  parser_ = XML_ParserCreate("UTF-8");
  ::XML_SetUserData(parser_, this);
  ::XML_SetElementHandler(parser_, StartElementCB, EndElementCB);
  ::XML_SetCharacterDataHandler(parser_, CharHandlerCB);
  ::XML_SetStartCdataSectionHandler(parser_, StartCDataCB);
  ::XML_SetEndCdataSectionHandler(parser_, EndCDataCB);
}

TxpXmlData::~TxpXmlData() {
  XML_ParserFree(parser_);
}

void TxpXmlData::StartElementCB(void *userData, const char *name,
                            const char **atts) {
  reinterpret_cast<TxpXmlData*>(userData)->StartElement(name, atts);
}

void TxpXmlData::EndElementCB(void *userData, const char *name) {
  reinterpret_cast<TxpXmlData*>(userData)->EndElement(name);
}

void TxpXmlData::CharHandlerCB(void *userData, const char * data, int32 len) {
  reinterpret_cast<TxpXmlData*>(userData)->CharHandler(data, len);
}

void TxpXmlData::StartCDataCB(void *userData) {
  reinterpret_cast<TxpXmlData*>(userData)->StartCData();
}

void TxpXmlData::EndCDataCB(void *userData) {
  reinterpret_cast<TxpXmlData*>(userData)->EndCData();
}

bool TxpXmlData::Parse(const std::string &tpdb) {
  bool binary;
  enum XML_Status r;
  tpdb_.clear();
  tpdb_.append(tpdb);
  fname_.clear();
  fname_ = tpdb + "/" + type_ + "-" + name_ + ".xml";
  Input ki(fname_.c_str(), &binary);
  if (binary) return false;
  while (getline(ki.Stream(), buffer_)) {
    buffer_.append("\n");  // Reappend line break to get correct error reporting
    r = XML_Parse(parser_, buffer_.c_str(), buffer_.length(), false);
    if (r == XML_STATUS_ERROR) {
      KALDI_WARN << "Expat XML Parse error: " <<
          XML_ErrorString(::XML_GetErrorCode(parser_))
                 << " Line: " << ::XML_GetCurrentLineNumber(parser_)
                 << " Col:" << XML_GetCurrentColumnNumber(parser_);
      return false;
    }
  }
  XML_Parse(parser_, "", 0, true);
  return true;
}

int32 TxpXmlData::SetAtt(const char * name, const char ** atts,
                         std::string *val) {
  int32 i = 0;
  val->clear();
  while (atts[i]) {
    if (!strcmp(atts[i], name)) *val = atts[i + 1];
    i += 2;
  }
  return i / 2;
}

}  // namespace kaldi
