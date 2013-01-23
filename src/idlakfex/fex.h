// idlakfex/fex.h

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

#ifndef SRC_IDLAKFEX_FEX_H
#define SRC_IDLAKFEX_FEX_H

// This file defines the feature extraction system

#include <pugixml.hpp>
#include "base/kaldi-common.h"
#include "idlaktxp/idlak-common.h"
#include "idlaktxp/txpxmldata.h"

namespace kaldi {

// Relationship between a featuere extraction function and current/ pre or post
// context of phone. Used to dictate behaviour in pauses.
enum FEXPAU_TYPE {FEXPAU_TYPE_CUR = 0,
                  FEXPAU_TYPE_PRE = 1,
                  FEXPAU_TYPE_PST = 2};

// Is the feature result a string or an integer
enum FEX_TYPE {FEX_TYPE_STR = 0,
               FEX_TYPE_INT = 1};


// Default maximum size of a feature in bytes
// (can be set in fex-<architecture>.xml)
#define FEX_MAXFIELDLEN 5
// Default error code
#define FEX_ERROR "ERROR"
// Default pause handling - SPT means have two sil models between
// every phrase - HTS menas use a single sil model within utterances
#define PAUSEHANDLING "SPT"

class Fex;
struct FexFeat;

typedef bool (* fexfunction) 
     (Fex *, const pugi::xpath_node_set *, const pugi::xml_node *,
      const char *);

class Fex: public TxpXmlData {
 public:
  explicit Fex(const char * name)
      : TxpXmlData("fex", name) {}
  ~Fex() {}
  //check and append value - function for int and str
  //call feature function and deal with pause behaviour
 private:
  void StartElement(const char * name, const char ** atts);
};

// hold information on a feature function defined in fex-<architecture>.xml
struct FexFeat {
  const fexfunction func;
  bool pauctx;
  enum FEXPAU_TYPE pautype;
  enum FEX_TYPE type;
};

}  // namespace kaldi

#endif  // SRC_IDLAKFEX_FEX_H_
