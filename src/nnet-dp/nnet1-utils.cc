// nnet-dp/nnet1-utils.cc

// Copyright 2012  Johns Hopkins University (author:  Daniel Povey)

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

#include "nnet-dp/nnet1-utils.h"

namespace kaldi {

bool ReadAlignmentsAndFeatures(
    std::string feature_rspecifier,
    std::string alignments_rspecifier,
    std::string validation_utt_list_rxfilename,
    std::vector<CompressedMatrix> *train_feats,
    std::vector<CompressedMatrix> *validation_feats,
    std::vector<std::vector<int32> > *train_ali,
    std::vector<std::vector<int32> > *validation_ali) {
  unordered_set<std::string, StringHasher> validation_utts;
  { // read validation_utts.
    kaldi::Input ki(validation_utt_list_rxfilename); // expect no
    // binary-mode header; can only be a text file.
    std::string utt;
    while (std::getline(ki.Stream(), utt)) {
      Trim(&utt); // Remove leading + trailing whitespace.
      if (!IsToken(utt))
        KALDI_ERR << "Bad utterance-id (reading validation utt-list "
                  << validation_utt_list_rxfilename << ")";
      validation_utts.insert(utt);
    }
  }
  int32 num_valid = 0;
  std::vector<CompressedMatrix*> tmp_feats;
  std::vector<std::vector<int32>* > tmp_ali;
  std::vector<bool> is_validation;
  SequentialBaseFloatMatrixReader feats_reader(feature_rspecifier);
  RandomAccessInt32VectorReader ali_reader(alignments_rspecifier);

  int32 num_err = 0;
  for (; !feats_reader.Done(); feats_reader.Next()) {
    std::string utt_id = feats_reader.Key();
    if (!ali_reader.HasKey(utt_id)) {
      KALDI_WARN << "Skipping utterance " << utt_id << " because no alignment ";
      num_err++;
      continue;
    }
    const std::vector<int32> &this_ali = ali_reader.Value(utt_id);
    int count = validation_utts.count(utt_id); // 0 or 1.
    num_valid += count;
    is_validation.push_back(count != 0);
    
    tmp_feats.push_back(new CompressedMatrix(feats_reader.Value()));
    tmp_ali.push_back(new std::vector<int32>(this_ali));
  }
  int32 num_train = static_cast<int32>(tmp_ali.size()) - num_valid;
  if (num_train != 0 && num_valid != 0)
    KALDI_LOG << "Read " << num_train << " training utterances, " << num_valid
              << " validation utterances.";
  else
    KALDI_WARN << "Read " << num_train << " training utterances, " << num_valid
               << " validation utterances.";
  train_feats->resize(num_train);
  train_ali->resize(num_train);
  validation_feats->resize(num_valid);
  validation_ali->resize(num_valid);
  int32 train_count = 0, validation_count = 0;
  for (size_t i = 0; i < tmp_ali.size(); i++) {
    if (is_validation[i]) {
      KALDI_ASSERT(validation_count < num_valid);
      (*validation_feats)[validation_count].Swap(tmp_feats[i]);
      (*validation_ali)[validation_count].swap(*(tmp_ali[i]));
      validation_count++;
    } else {
      KALDI_ASSERT(train_count < num_train);
      (*train_feats)[train_count].Swap(tmp_feats[i]);
      (*train_ali)[train_count].swap(*(tmp_ali[i]));
      train_count++;
    }
    delete tmp_feats[i];
    delete tmp_ali[i];
  }
  KALDI_ASSERT(train_count == num_train && validation_count == num_valid);
  return (num_train != 0 && num_valid != 0);
  
}

void ConvertAlignmentsToPdfs(const TransitionModel &trans_model,
                             std::vector<std::vector<int32> > *ali) {
  for (int32 i = 0; i < ali->size(); i++) {
    std::vector<int32> &this_ali = (*ali)[i];
    for (std::vector<int32>::iterator iter = this_ali.begin(),
             end = this_ali.end();
         iter != end; ++iter)
      *iter = trans_model.TransitionIdToPdf(*iter);
  }
}


} // namespace kaldi
