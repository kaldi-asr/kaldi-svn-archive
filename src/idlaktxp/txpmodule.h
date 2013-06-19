// idlaktxp/txpmodule.h

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

#ifndef KALDI_IDLAKTXP_TXPMODULE_H_
#define KALDI_IDLAKTXP_TXPMODULE_H_

// This file defines the basic txp module which incrementally parses
// either text, tox (token oriented xml) tokens, or spurts (phrases)
// containing tox tokens.

#include <string>
#include "pugixml.hpp"

#include "base/kaldi-common.h"
#include "idlaktxp/idlak-common.h"
#include "idlaktxp/txpconfig.h"
#include "idlaktxp/txpnrules.h"
#include "idlaktxp/txppos.h"
#include "idlaktxp/txppbreak.h"
#include "idlaktxp/txplexicon.h"
#include "idlaktxp/txplts.h"
#include "idlaktxp/txpsylmax.h"

namespace kaldi {

/// Base class for all modules
///
/// Takes puji XML input and modifies it for puji XML output
/// Has a configuration section which can be accessed using utility
/// functions
class TxpModule {
 public:
  /// Construct module, also loads tpdb modules in specific instances
  explicit TxpModule(const std::string &name, const std::string &tpdb,
                     const std::string &configf);
  virtual ~TxpModule() {}
  /// Process the XML, modifying the XML to reflect linguistic
  /// information
  virtual bool Process(pugi::xml_document * input) {return true;}
  /// Return a configuration value for this module as a string
  const char * GetConfigValue(const char * key);
  /// Return a boolean configuration
  /// True/true/TRUE -> true, False, false, FALSE, anything else -> false
  bool GetConfigValueBool(const char * key);
  /// Get the name of the module
  const std::string & GetName() {return name_;}

 protected:
  /// Configuration structure for the module
  TxpConfig config_;

 private:
  /// Name of the module
  std::string name_;
  /// Directory for text processing database (tpdb) files
  /// used by txpxmldata objectys in the module
  std::string tpdb_;
  /// Filename for a user configuration file
  std::string configf_;
};

/// Tokenise input text into tokens and whitespace
/// \ref idlaktxp_token
class TxpTokenise : public TxpModule {
 public:
  explicit TxpTokenise(const std::string &tpdb,
                       const std::string &configf = "");
  ~TxpTokenise();
  bool Process(pugi::xml_document * input);

 private:
  /// Analyses the characters and sets flags giving case, foriegn
  /// charcater info
  int32 SetPuncCaseInfo(std::string *tkin, pugi::xml_node *tk);
  /// A normalisation rule database used to decide case etc.
  /// Currently this data will be loaded muliple times across
  /// multiple modules
  TxpNRules nrules_;
};

/// Convert punctuation and break tags into pauses
/// /ref idlaktxp_pause_insertion
class TxpPauses : public TxpModule {
 public:
  explicit TxpPauses(const std::string &tpdb,
                     const std::string &configf = "");
  ~TxpPauses();
  bool Process(pugi::xml_document * input);

 private:
  /// Object containing lookup between punctuation and break strength
  /// and time
  TxpPbreak pbreak_;
  /// If true tries distinguish real line breaks from those at page left
  /// A hypenation zone (hzone) can then be used to remove phantom line breaks
  /// and reappend soft hypenation
  bool hzone_;
  /// Column start for hzone
  int32 hzone_start_;
  /// Column end for hzone
  int32 hzone_end_;
};

/// Assign part of speech to each token
/// /ref idlaktxp_pos
class TxpPosTag : public TxpModule {
 public:
  explicit TxpPosTag(const std::string &tpdb,
                     const std::string &configf = "");
  ~TxpPosTag();
  bool Process(pugi::xml_document * input);

 private:
  /// Greedy regex and bigram tagger
  TxpPos tagger_;
  /// Tagger set
  TxpPosSet posset_;
};

/// Using pauses contruct a phrase structure in the XML
/// by adding spt elements. /ref idlaktxp_phrase
class TxpPhrasing : public TxpModule {
 public:
  explicit TxpPhrasing(const std::string &tpdb,
                       const std::string &configf = "");
  ~TxpPhrasing();
  bool Process(pugi::xml_document * input);

 private:
};

/// Convert tokens into pronunications based on lexicons and
/// lts rules. Currently only one lexicon is supported. A user lexicon
/// and ability to add bilingual lexicons may be added. /ref idlaktxp_pron
class TxpPronounce : public TxpModule {
 public:
  explicit TxpPronounce(const std::string &tpdb,
                        const std::string &configf = "");
  ~TxpPronounce();
  bool Process(pugi::xml_document * input);

 private:
  /// Checks lexicon and lts to determine pronuciations
  /// and appends it to the lex lookup structure
  void AppendPron(const char * entry, const std::string &word,
                  TxpLexiconLkp &lexlkp);
  /// A normalisation rules object is required to allow the default
  /// pronunciation of symbol and digit characters
  TxpNRules nrules_;
  /// A pronuciation lexicon object
  TxpLexicon lex_;
  /// A cart based letter to sound rul object
  TxpLts lts_;
};

/// Syllabifies pronunciations into onset, nucleus, coda items
/// Allows laison from left to right. /ref idlaktxp_syll
class TxpSyllabify : public TxpModule {
 public:
  explicit TxpSyllabify(const std::string &tpdb,
                        const std::string &configf = "");
  ~TxpSyllabify();
  bool Process(pugi::xml_document * input);

 private:
  /// Object containing specifications of valid nucleus and onset
  /// phone sequences
  TxpSylmax sylmax_;
};

}  // namespace kaldi

#endif  // KALDI_IDLAKTXP_TXPMODULE_H_
