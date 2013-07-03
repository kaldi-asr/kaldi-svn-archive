// idlaktxp/txpfexspec.h

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

#ifndef SRC_IDLAKTXP_FEXSPEC_H
#define SRC_IDLAKTXP_FEXSPEC_H

// This file defines the feature extraction system

#include <pugixml.hpp>
#include "base/kaldi-common.h"
#include "idlaktxp/idlak-common.h"
#include "idlaktxp/txpxmldata.h"
#include <deque>

namespace kaldi {

// Relationship between a featuere extraction function and current/ pre or post
// context of phone. Used to dictate behaviour in pauses.
enum FEXSPECPAU_TYPE {FEXSPECPAU_TYPE_CUR = 0,
                      FEXSPECPAU_TYPE_PRE = 1,
                      FEXSPECPAU_TYPE_PST = 2};

// whether to allow cross pause context across breaks < 4
// SPT (default) no, UTT yes.
enum FEXSPECPAU_HAND {FEXSPECPAU_HAND_SPT = 0,
                      FEXSPECPAU_HAND_UTT = 1};

// Is the feature result a string or an integer
enum FEXSPEC_TYPE {FEXSPEC_TYPE_STR = 0,
                   FEXSPEC_TYPE_INT = 1};


// Default maximum size of a feature in bytes
// (can be set in fex-<architecture>.xml)
#define FEXSPEC_MAXFIELDLEN 5
// Default error code
#define FEXSPEC_ERROR "ERROR"
// Default pause handling - SPT means have two sil models between
// every phrase - HTS menas use a single sil model within utterances
#define FEXSPEC_PAUSEHANDLING "SPT"
// Default null value for features
#define FEXSPEC_NULL "xx"

// TODO(MPA): add padding control so that all features value strings can be
// set to the same length so that it is easier to read and compare them visually


class TxpFexspec;
struct TxpFexspecFeat;
class TxpFexspecModels;
class TxpFexspecContext;

// moving vector which keeps context for each context level
typedef std::deque<pugi::xml_node> XmlNodeVector;

// a feature function
typedef bool (* fexfunction)
(const TxpFexspec *, const TxpFexspecFeat *, const TxpFexspecContext *, char *);

// array of feature functiion names
extern const char * const FEXFUNCLBL[];
// array of feature functiion pointers
extern const fexfunction FEXFUNC[];
// array of feature function pause handling type
extern const enum FEXSPECPAU_TYPE FEXFUNCPAUTYPE[];
// array of feature function types
extern const enum FEXSPEC_TYPE FEXFUNCTYPE[];

/// lookup valid values from set name
typedef std::map<std::string, StringSet> LookupMapSet;
/// valid values/ set name pair
typedef std::pair<std::string, StringSet> LookupMapSetItem;
/// vector feature structures in architecture
typedef std::vector<TxpFexspecFeat> TxpFexspecFeatVector;

class TxpFexspec: public TxpXmlData {
 public:
  explicit TxpFexspec(TxpConfig * config, const char * tpdb, const char * architecture);
  explicit TxpFexspec() : TxpXmlData() {};
  ~TxpFexspec() {}
  // calculate biggest buffer required for feature output
  int32 MaxFeatSz();
  // return pause handling strategy
  enum FEXSPECPAU_HAND GetPauseHandling() {return pauhand_;}
  // add pause structure to an XML document
  int32 AddPauseNodes(pugi::xml_document  * doc);
  // call feature function and deal with pause behaviour
  bool ExtractFeatures(const TxpFexspecContext &context, char * buf);
  // check and append value - function string
  bool AppendValue(const TxpFexspecFeat &feat, bool error,
                   const char * s, char * buf) const;
  // check and append value - function integer
  bool AppendValue(const TxpFexspecFeat &feat, bool error,
                   int32 i, char * buf) const;
  // append a null value
  bool AppendNull(const TxpFexspecFeat &feat, char * buf) const;
  // append an error value
  bool AppendError(const TxpFexspecFeat &feat, char * buf) const;
  // return feature specific mapping between fex value and desired value
  const char * Mapping(const TxpFexspecFeat &feat, const char * instr) const;
 private:
  // configuration file object
  TxpConfig config_;
  // Parser for tpdb xml fex setup
  void StartElement(const char * name, const char ** atts);
  // return index of a feature function by name
  int32 GetFeatIndex(const std::string &name);
  // stores valid values for string based features
  LookupMapSet sets_;
  // stores null values for string based features
  LookupMap setnull_;
  // stores information on current feature architecture
  TxpFexspecFeatVector fexspecfeats_;
  // lookup for feature name to index of fexspecfeats_
  LookupInt fexspecfeatlkp_;
  // maximum feature field length
  int32 fexspec_maxfieldlen_;
  // pause handling
  enum FEXSPECPAU_HAND pauhand_;
  // used while parsing input XML to keep track of current set
  std::string curset_;
  // used while parsing input XML to keep track of current fexspec function
  std::string curfunc_;
  
};

// hold information on a feature function defined in fex-<architecture>.xml
struct TxpFexspecFeat {
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
  // whether to allow cross silence context pause
  bool pauctx;
  // how the extraction function behaves on silence  
  enum FEXSPECPAU_TYPE pautype;
  // the type of fuction (string or integer)
  enum FEXSPEC_TYPE type;
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
class TxpFexspecModels {
 public:
  explicit TxpFexspecModels() {}
  ~TxpFexspecModels();
  // initialise model name output
  void Init(TxpFexspec *fexspec);
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

// iterator for accessing linguistic structure in the XML document
class TxpFexspecContext {
 public:
  explicit TxpFexspecContext(const pugi::xml_document &doc,
                      enum FEXSPECPAU_HAND pauhand);
  ~TxpFexspecContext() {};
  // iterate to next item
  bool next();
  // are we in a silence
  bool isBreak() {return isbreak_;}
  // is the silence doucument internal
  bool isBreakInternal() {return internalbreak_;}
  // is the break at the end or begining of a spt
  bool isEndBreak() {return endbreak_;}
  // is the break between sentences
  bool isUttBreak() {return uttbreak_;}
  // return phon back or forwards from current phone
  pugi::xml_node getPhon(int32 idx, bool pauctx) const;
 private:
  // look up from the node until we find the correct current context node
  pugi::xml_node getContextUp(const pugi::xml_node &node,
                                const char * name);
  bool isbreak_;
  bool endbreak_;
  bool internalbreak_;
  bool uttbreak_;
  enum FEXSPECPAU_HAND pauhand_;
  pugi::xpath_node_set phons_;
  pugi::xpath_node_set syls_;
  pugi::xpath_node_set wrds_;
  pugi::xpath_node_set spts_;
  pugi::xpath_node_set utts_;
  pugi::xpath_node_set::const_iterator cur_phon_;
  pugi::xpath_node_set::const_iterator cur_syl_;
  pugi::xpath_node_set::const_iterator cur_wrd_;
  pugi::xpath_node_set::const_iterator cur_spt_;
  pugi::xpath_node_set::const_iterator cur_utt_;
  // phone contexts
  XmlNodeVector ctxphons_;
  // syl contexts
  XmlNodeVector ctxsyls_;
  // wrd contexts
  XmlNodeVector ctxwrds_;
  // spt contexts
  XmlNodeVector ctxspts_;
  // utt contexts
  XmlNodeVector ctxutts_;
};

}  // namespace kaldi

#endif  // SRC_IDLAKTXP_FEXSPEC_H_
