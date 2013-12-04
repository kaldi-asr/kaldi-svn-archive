// tree/build-tree-questions.cc

// Copyright 2009-2011  Microsoft Corporation
//           2013-      Arnab Ghoshal

// See ../../COPYING for clarification regarding multiple authors
//
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

#include <string>

#include "util/stl-utils.h"
#include "util/text-utils.h"
#include "tree/build-tree-questions.h"
#include "tree/build-tree-utils.h"

namespace kaldi {

void QuestionsForKey::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<QuestionsForKey>");
  int32 size = initial_questions.size();
  WriteBasicType(os, binary, size);
  for (int32 i = 0; i < size; i++)
    WriteIntegerVector(os, binary, initial_questions[i]);
  refine_opts.Write(os, binary);
  WriteToken(os, binary, "</QuestionsForKey>");
}

void QuestionsForKey::Read(std::istream &is, bool binary) {
  int32 size;
  ExpectToken(is, binary, "<QuestionsForKey>");
  ReadBasicType(is, binary, &size);
  initial_questions.resize(size);
  for (int32 i = 0; i < size; i++)
    ReadIntegerVector(is, binary, &(initial_questions[i]));
  refine_opts.Read(is, binary);
  ExpectToken(is, binary, "</QuestionsForKey>");
}


const QuestionsForKey& Questions::GetQuestionsOf(EventKeyType key) const {
  std::map<EventKeyType, size_t>::const_iterator iter;
  if ((iter = key_idx_.find(key)) == key_idx_.end()) {
    KALDI_ERR << "Questions: no options for key " << key;
  }
  size_t idx = iter->second;
  assert(idx < key_options_.size());
  key_options_[idx]->Check();
  return *(key_options_[idx]);
}

void Questions::SetQuestionsOf(EventKeyType key,
                               const QuestionsForKey &options_of_key) {
  options_of_key.Check();
  if (key_idx_.count(key) == 0) {
    key_idx_[key] = key_options_.size();
    key_options_.push_back(new QuestionsForKey());
    *(key_options_.back()) = options_of_key;
  } else {
    size_t idx = key_idx_[key];
    assert(idx < key_options_.size());
    *(key_options_[idx]) = options_of_key;
  }
}

void Questions::AddQuestion(EventKeyType key,
                            const std::vector<EventValueType> &question) {
  KALDI_ASSERT(IsSorted(question));
  if (key_idx_.count(key) == 0) {
    key_idx_[key] = key_options_.size();
    key_options_.push_back(new QuestionsForKey());
    key_options_.back()->initial_questions.push_back(question);
  } else {
    size_t idx = key_idx_[key];
    assert(idx < key_options_.size());
    key_options_[idx]->initial_questions.push_back(question);
  }
}


void Questions::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<Questions>");

  std::vector<EventKeyType> keys_with_options;
  this->GetKeysWithQuestions(&keys_with_options);
  for (size_t i = 0; i < keys_with_options.size(); i++) {
    EventKeyType key = keys_with_options[i];
    WriteToken(os, binary, "<Key>");
    WriteBasicType(os, binary, key);
    const QuestionsForKey &opts = GetQuestionsOf(key);
    opts.Write(os, binary);
  }
  WriteToken(os, binary, "</Questions>");
}

void Questions::Read(std::istream &is, bool binary) {
  // First, clear anything present.
  DeletePointers(&key_options_);
  key_options_.clear();
  key_idx_.clear();

  ExpectToken(is, binary, "<Questions>");

  std::vector<EventKeyType> keys_with_options;

  while (1) {
    std::string token;
    ReadToken(is, binary, &token);
    if (token == "</Questions>") {
      return;
    } else {
      if (token != "<Key>")
      KALDI_ERR << "Questions::Read, expecting <Key>, got " << token;
      EventKeyType key;
      ReadBasicType(is, binary, &key);
      QuestionsForKey opts;
      opts.Read(is, binary);
      SetQuestionsOf(key, opts);
    }
  }
}

void Questions::InitRand(const BuildTreeStatsType &stats, int32 num_quest,
                         int32 num_iters_refine, AllKeysType all_keys_type) {
  std::vector<EventKeyType> all_keys;
  FindAllKeys(stats, all_keys_type, &all_keys);  // get all keys.
  if (all_keys_type == kAllKeysUnion) {
    KALDI_WARN << "Using union of all keys.  This may work depending on how "
               << "you are building the tree but may crash (unless you have "
               << "already ensured that stats currently on the same leaf all "
               << "share the same set of keys.)";
  }

  for (size_t i = 0; i < all_keys.size(); i++) {
    EventKeyType key = all_keys[i];
    std::vector<EventValueType> all_values;
    bool b = PossibleValues(key, stats, &all_values);  // get possible values
    if (all_keys_type != kAllKeysUnion)
      KALDI_ASSERT(b);
    // must have some value defined, since key exists in stats
    KALDI_ASSERT(all_values.size() != 0);
    QuestionsForKey q_for_key;
    q_for_key.refine_opts.num_iters = num_iters_refine;
    q_for_key.initial_questions.clear();  // Make sure empty.
    if (all_values.size() == 1) {
      // can't have meaningful questions because only 1 possible value.
      // use empty set of questions. So do nothing.
    } else {
      q_for_key.initial_questions.resize((size_t) num_quest);
      for (size_t i = 0; i < (size_t) num_quest; i++) {
        std::vector<EventValueType> &this_quest = q_for_key.initial_questions[i];
        for (size_t j = 0; j < all_values.size() / 2; j++)
          this_quest.push_back(all_values[RandInt(0, all_values.size() - 1)]);
        SortAndUniq(&this_quest);
        assert(!this_quest.empty());
      }
      SortAndUniq(&q_for_key.initial_questions);  // Ensure unique questions
    }
    q_for_key.Check();
    SetQuestionsOf(key, q_for_key);
  }
}

void ReadKeyedQuestions(std::istream &is, Questions *qst) {
  std::string line;
  std::vector<std::string> stringvector;
  while (std::getline(is, line)) {
    std::istringstream iss(line);
    EventKeyType key, val;
    std::vector<EventValueType> v;
    SplitStringToVector(iss.str(), " \t\r", true, &stringvector);
    if (stringvector.size() < 3 || stringvector[01] != "?" ||
        !ConvertStringToInteger(stringvector[0], &key)) {
      KALDI_ERR << "Unexpected line : " << line;
    }
    for (int32 i = 2; i < stringvector.size(); i++) {
      if (!ConvertStringToInteger(stringvector[i], &val)) {
        KALDI_ERR << "Unexpected line : " << line;
      }
      v.push_back(val);
    }
    std::sort(v.begin(), v.end());
    qst->AddQuestion(key, v);
  }
}

}  // end namespace kaldi
