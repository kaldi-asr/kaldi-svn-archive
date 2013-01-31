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

// current phone name
bool FexFuncCURSTRp(Fex * fex, const FexFeat &feat,
                    const pugi::xpath_node_set &nodes,
                    int32 idx,
                    const pugi::xml_node &p,
                    char * buffer) {
  bool okay = true;
  const char * phonename;
  // extract value from XML
  phonename = p.attribute("val").value();
  // error on empty string
  if (!*phonename) okay = false;
  // check and append value
  okay = fex->AppendValue(feat, okay, phonename, buffer);
  // return error status
  return okay;
}

// previous current phone name
bool FexFuncPRESTRbp(Fex * fex, const FexFeat &feat,
                     const pugi::xpath_node_set &nodes,
                     int32 idx,
                     const pugi::xml_node &p,
                     char * buffer) {
  bool okay = true;
  const char * phonename;
  // extract value from XML
  phonename = nodes[idx - 1].node().attribute("val").value();
  // error on empty string
  if (!*phonename) okay = false;
  // check and append value
  okay = fex->AppendValue(feat, okay, phonename, buffer);
  // return error status
  return okay;
}

}  // namespace kaldi
