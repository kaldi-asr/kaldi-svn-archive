// durmod/kaldi-durmod.h

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
#ifndef DURMOD_KALDI_DURMOD_H_
#define DURMOD_KALDI_DURMOD_H_

#include <iostream>
#include <vector>
#include <utility>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "util/stl-utils.h"

namespace kaldi {

struct PhoneDurationModelOptions {
  int left_ctx, right_ctx;
  int layer1_dim, layer2_dim;

  void Register(OptionsItf *po);
};

struct PhoneDurationEg {
  std::vector<int32> left_context_phones;
  std::vector<int32> left_context_durations;
  std::vector<int32> right_context_phones;
  std::vector<bool> extra_membership;
  int32 phone;
  int32 duration;
};


class PhoneDurationEgsHolder {
// This is a class to abstract from the actual I/O  format
// As we don't employ splicing, this is a fairly straightforward
// structure to maintain

  std::string utt_id;
  std::vector<PhoneDurationEg> egs;

};

typedef std::vector<std::vector<int32> > PhoneSets;
typedef unordered_map<int32, std::vector<int32> > ClusterQuestionRevMap;
typedef unordered_map<int32, int32> RootsRevMap;

class PhoneDuratioEgsMaker {
 public:
  explicit PhoneDuratioEgsMaker(const PhoneDurationModelOptions &opts,
                                const PhoneSets &roots,
                                const PhoneSets &questions);

  void GenerateExamples(const std::vector<std::pair<int32, int32> > &alignment,
                        std::vector<PhoneDurationEg> *egs);
 private:
  ClusterQuestionRevMap questions_;
  RootsRevMap roots_;

  int32 left_ctx_, right_ctx_;
};

class PhoneDurationModelNnet {
 public:
  void Read(std::iostream &is, bool binary);
  void Write(std::iostream &os, bool binary);
};


}  // namespace kaldi
#endif  // DURMOD_KALDI_DURMOD_H_

