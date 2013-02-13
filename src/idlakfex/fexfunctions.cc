// idlakfex/fexfunctions.cc

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

// Each function fills a feature extraction buffer up with a feature based on
// an XML node and context.

// Function names are of the form FexFunc[CUR|PRE|PST][INT|STR][feature name]
// i.e. FexFuncCURINTp0

// if the function returns false fex appends the NULL value defined in the
// architecture

// if the function throws an error fex appends the ERROR value, reports and
// tries to continue

#include "./fex.h"

namespace kaldi {


// previous previous phone name
bool FexFuncPRESTRbbp(const Fex * fex,
                     const FexFeat * feat,
                     const FexContext * context,
                     char * buffer) {
  bool okay = true;
  pugi::xml_node node;
  const char * phonename;
  // get node from correct context
  node = context->getPhon(-2, feat->pauctx);
  // check for NULL value
  if (node.empty()) fex->AppendNull(*feat, buffer);
  else {
    // extract value from XML
    phonename = node.attribute("val").value();
    // check and append value
    okay = fex->AppendValue(*feat, okay, phonename, buffer);
  }
  // return error status
  return okay;
}

// previous phone name
bool FexFuncPRESTRbp(const Fex * fex,
                     const FexFeat * feat,
                     const FexContext * context,
                     char * buffer) {
  bool okay = true;
  pugi::xml_node node;
  const char * phonename;
  // get node from correct context
  node = context->getPhon(-1, feat->pauctx);
  // check for NULL value
  if (node.empty()) fex->AppendNull(*feat, buffer);
  else {
    // extract value from XML
    phonename = node.attribute("val").value();
    // check and append value
    okay = fex->AppendValue(*feat, okay, phonename, buffer);
  }
  // return error status
  return okay;
}

// current phone name
bool FexFuncCURSTRp(const Fex * fex,
                     const FexFeat * feat,
                     const FexContext * context,
                     char * buffer) {
  bool okay = true;
  pugi::xml_node node;
  const char * phonename;
  // get node from correct context
  node = context->getPhon(0, feat->pauctx);
  // check for NULL value
  if (node.empty()) fex->AppendNull(*feat, buffer);
  else {
    // extract value from XML
    phonename = node.attribute("val").value();
    // check and append value
    okay = fex->AppendValue(*feat, okay, phonename, buffer);
  }
  // return error status
  return okay;
}

// next phone name
bool FexFuncPSTSTRfp(const Fex * fex,
                     const FexFeat * feat,
                     const FexContext * context,
                     char * buffer) {
  bool okay = true;
  pugi::xml_node node;
  const char * phonename;
  // get node from correct context
  node = context->getPhon(1, feat->pauctx);
  // check for NULL value
  if (node.empty()) fex->AppendNull(*feat, buffer);
  else {
    // extract value from XML
    phonename = node.attribute("val").value();
    // check and append value
    okay = fex->AppendValue(*feat, okay, phonename, buffer);
  }
  // return error status
  return okay;
}

// next next phone name
bool FexFuncPSTSTRffp(const Fex * fex,
                     const FexFeat * feat,
                     const FexContext * context,
                     char * buffer) {
  bool okay = true;
  pugi::xml_node node;
  const char * phonename;
  // get node from correct context
  node = context->getPhon(2, feat->pauctx);
  // check for NULL value
  if (node.empty()) fex->AppendNull(*feat, buffer);
  else {
    // extract value from XML
    phonename = node.attribute("val").value();
    // check and append value
    okay = fex->AppendValue(*feat, okay, phonename, buffer);
  }
  // return error status
  return okay;
}



}  // namespace kaldi
