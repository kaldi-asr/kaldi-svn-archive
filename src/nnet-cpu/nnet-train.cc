// nnet/nnet-train.cc

// Copyright 2012   Johns Hopkins University (author: Daniel Povey)

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

#include "nnet-cpu/nnet-train.h"

namespace kaldi {

void NnetValidationSet::AddUtterance(
    const MatrixBase<BaseFloat> &features,
    const VectorBase<BaseFloat> &spk_info, // may be empty
    std::vector<int32> &pdf_ids,
    BaseFloat utterance_weight) {
  KALDI_ASSERT(pdf_ids.size() == static_cast<size_t>(features.NumRows()));
  KALDI_ASSERT(utterance_weight > 0.0);
  utterances_.push_back(new Utterance(features, spk_info,
                                      pdf_ids, utterance_weight));
  if (utterances_.size() != 0) { // Check they have consistent dimensions.
    KALDI_ASSERT(features.NumCols() == utterances_[0]->features.NumCols());
    KALDI_ASSERT(spk_info.Dim() == utterances_[0]->spk_info.Dim());
  }
}

NnetValidationSet::~NnetValidationSet() {
  for (size_t i = 0; i < utterances_.size(); i++)
    delete utterances_[i];
}


BaseFloat NnetValidationSet::ComputeGradient(const Nnet &nnet,
                                             Nnet *nnet_gradient) const {
  KALDI_ASSERT(!utterances_.empty());
  bool treat_as_gradient = true, pad_input = true;
  BaseFloat tot_objf = 0.0, tot_weight = 0.0;
  nnet_gradient->SetZero(treat_as_gradient);  
  for (size_t i = 0; i < utterances_.size(); i++) {
    const Utterance &utt = *(utterances_[i]);
    tot_objf += NnetGradientComputation(nnet,
                                        utt.features, utt.spk_info,
                                        pad_input, utt.weight,
                                        utt.pdf_ids, nnet_gradient);
    tot_weight += utt.weight * utt.features.NumRows();
  }
  KALDI_VLOG(2) << "Validation set objective function " << (tot_objf / tot_weight)
                << " over " << tot_weight << " frames.";
  return tot_objf / tot_weight;
}

NnetAdaptiveTrainer::NnetAdaptiveTrainer(const NnetAdaptiveTrainerConfig &config,
                                         Nnet *nnet,
                                         NnetValidationSet *validation_set) {
  
}


void NnetAdaptiveTrainer::TrainOnExample(const NnetTrainingExample &value) {
  
}

  
} // namespace
