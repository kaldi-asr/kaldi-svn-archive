// durmod/kaldi-durmod.cc

// Copyright (c) 2015, Johns Hopkins University (Yenda Trmal<jtrmal@gmail.com>)

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

#include <algorithm>

#include "durmod/kaldi-durmod.h"

namespace kaldi {

PhoneDuratioEgsMaker::PhoneDuratioEgsMaker(
    const PhoneDurationModelOptions &opts,
    const PhoneSets &roots,
    const PhoneSets &questions) {

  left_ctx_ = opts.left_ctx;
  right_ctx_ = opts.right_ctx;

  // and the reverse map for phoneme tree roots
  for (int i = 0; i < roots.size(); i++) {
    for (int j = 0; j < roots[i].size(); j++) {
      KALDI_ASSERT(roots_.count(j) == 0);
      roots_[j] = i;
    }
  }
  // create the reverse map for questions/cluster membership
  for (int i = 0; i < questions.size(); i++) {
    for (int j = 0; j < questions[i].size(); j++) {
      questions_[j].push_back(i);
    }
  }
}

void PhoneDuratioEgsMaker::GenerateExamples(
    const std::vector<std::pair<int32, int32> > &alignment,
    std::vector<PhoneDurationEg> *egs) {

  if (alignment.size() < (left_ctx_ + right_ctx_ +1)) {
    return;
  }

  for (int i = left_ctx_; i < alignment.size() - right_ctx_; i++) {
    PhoneDurationEg eg;
    for (int j = -left_ctx_; j < right_ctx_ + 1; j++) {
      std::pair<int32, int32> curr_phone = alignment[i+j];
      if (j < 0) {
        eg.left_context_phones.push_back(curr_phone.first);
        eg.left_context_durations.push_back(curr_phone.second);
      } else if (j == 0) {
        eg.phone = curr_phone.first;
        eg.duration = curr_phone.second;
      } else {
        eg.right_context_phones.push_back(curr_phone.first);
      }
    }
  }
}


}  // namespace kaldi
