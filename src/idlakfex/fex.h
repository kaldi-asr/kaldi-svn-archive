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

// whether to allow cross pause context across breaks < 4
// SPT (default) no, UTT yes.
enum FEXPAU_HAND {FEXPAU_HAND_SPT = 0,
                  FEXPAU_HAND_UTT = 1};

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
#define FEX_PAUSEHANDLING "SPT"
// Default null value for features
#define FEX_NULL "xx"

// TODO(MPA): add padding control so that all features value strings can be
// set to the same length so that it is easier to read and compare them visually


class Fex;
struct FexFeat;
class FexModels;

typedef bool (* fexfunction) 
     (Fex *, const FexFeat &, const pugi::xpath_node_set &,
      int32, const pugi::xml_node &, char *);

// array of feature functiion names
extern const char * const FEXFUNCLBL[];
// array of feature functiion pointers
extern const fexfunction FEXFUNC[];
// array of feature function pause handling type
extern const enum FEXPAU_TYPE FEXFUNCPAUTYPE[];
// array of feature function types
extern const enum FEX_TYPE FEXFUNCTYPE[];

/// lookup valid values from set name
typedef std::map<std::string, StringSet> LookupMapSet;
/// valid values/ set name pair
typedef std::pair<std::string, StringSet> LookupMapSetItem;
/// vector feature structures in architecture
typedef std::vector<FexFeat> FexFeatVector;

class Fex: public TxpXmlData {
 public:
  explicit Fex(const char * tpdb, const char * architecture);
  ~Fex() {}
  // calculate biggest buffer required for feature output
  int32 MaxFeatSz();
  // process an XML document
  int GetModels(const pugi::xml_document &doc, FexModels * models);
  // call feature function and deal with pause behaviour
  bool ExtractFeatures(pugi::xpath_node_set tks,  pugi::xml_node tk,
  		       int32 idx, char * buf);
  // check and append value - function string
  bool AppendValue(const FexFeat &feat, bool error, const char * s, char * buf);
  // check and append value - function integer
  bool AppendValue(const FexFeat &feat, bool error, int32 i, char * buf);
  // append a null value
  bool AppendNull(const FexFeat &feat, char * buf);
  // append an error value
  bool AppendError(const FexFeat &feat, char * buf);
  // return feature specific mapping between fex value and desired value
  const char * Mapping(const FexFeat &feat, const char * instr);
 private:
  void StartElement(const char * name, const char ** atts);
  // return index of a feature function by name
  int32 GetFeatIndex(const std::string &name);
  // stores valid values for string based features
  LookupMapSet sets_;
  // stores null values for string based features
  LookupMap setnull_;
  // stores information on current feature architecture
  FexFeatVector fexfeats_;
  // lookup for feature name to index of fexfeats_
  LookupInt fexfeatlkp_;
  // maximum feature field length
  int32 fex_maxfieldlen_;
  // pause handling
  enum FEXPAU_HAND pauhand_;
  // used while parsing input XML to keep track of current set
  std::string curset_;
  // used while parsing input XML to keep track of current fex function
  std::string curfunc_;
  
};

// hold information on a feature function defined in fex-<architecture>.xml
struct FexFeat {
  // name of feature
  std::string name;
  // htsname of feature (for information only)
  std::string htsname;
  // description of function (for information only)
  std::string desc;
  // delimiter used before feature in model name
  std::string delim;
  // value when no feature value is meaningful
  std::string nullvalue;
  // pointer to the extraction function
  fexfunction func;
  // whether to allow cross silence context on break < 4
  bool pauctx;
  // how the extraction function behaves on silence  
  enum FEXPAU_TYPE pautype;
  // the type of fuction (string or integer)
  enum FEX_TYPE type;
  // name of the valid set of values if a string type fucntion
  std::string set;
  // maximum value if an integer type value
  int32 max;
  // minimum value if an integer type value
  int32 min;
  // mapping from specific feature extraction values to architecture specific values
  LookupMap mapping;
};

// container for a feature output full context HMM modelnames
class FexModels {
 public:
  explicit FexModels(Fex *fex);
  ~FexModels();
  // clear container for reuse
  void Clear();
  // append an empty model
  char * Append();
  // return total number of phone models produces by XML input
  int GetNoModels() {return models_.size();}
  // return a model name
  const char * GetModel(int idx) {return models_[idx];}
 private:
  // vector or locally allocated buffers each for a model
  CharPtrVector models_;
  // maximum buffer length required based on feature achitecture
  int32 buflen_;
};


}  // namespace kaldi

#endif  // SRC_IDLAKFEX_FEX_H_
