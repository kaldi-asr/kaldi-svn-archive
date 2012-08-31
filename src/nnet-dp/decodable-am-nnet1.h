// nnet-dp/decodeable-am-nnet1.h

// Copyright 2012  Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_NNET_DP_DECODABLE_AM_NNET1_H_
#define KALDI_NNET_DP_DECODABLE_AM_NNET1_H_

#include <vector>

#include "base/kaldi-common.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "itf/decodable-itf.h"
#include "nnet-dp/am-nnet1.h"

namespace kaldi {

/// DecodableAmNnet1 is a decodable object that decodes
/// with a neural net acoustic model of type AmNnet1.

class DecodableAmNnet1: public DecodableInterface {
 public:
  DecodableAmNnet1(const TransitionModel &trans_model,
                   const AmNnet1 &am_nnet,
                   const Matrix<BaseFloat> &feats,
                   BaseFloat prob_scale = 1.0):
      trans_model_(trans_model), am_nnet_(am_nnet), scale_(prob_scale) {
    ForwardInitialLayers(feats);
  }
  
  // Note, frames are numbered from zero.  But state_index is numbered
  // from one (this routine is called by FSTs).
  virtual BaseFloat LogLikelihood(int32 frame, int32 transition_id);

  int32 NumFrames() { return input_to_softmax_.NumRows(); }
  
  // Indices are one-based!  This is for compatibility with OpenFst.
  virtual int32 NumIndices() { return trans_model_.NumTransitionIds(); }
  
  virtual bool IsLastFrame(int32 frame) {
    KALDI_ASSERT(frame < NumFrames());
    return (frame == NumFrames() - 1);
  }

 protected:
  void ForwardInitialLayers(const Matrix<BaseFloat> &initial_layers);
  void PrepareInput(const Matrix<BaseFloat> &feats,
                    Matrix<BaseFloat> *spliced_input) const;

  const TransitionModel &trans_model_;
  const AmNnet1 &am_nnet_;
  BaseFloat scale_;
  Matrix<BaseFloat> input_to_softmax_;
  Matrix<BaseFloat> category_zero_output_; // output of linear layer for category zero,
  // which is the top level of the tree.  Since this is needed for all
  // frames, we compute it at the start.
  Vector<BaseFloat> neg_log_priors_; // Negated log(prior), obtained from model;
  // we divide the outputs of the neural net by these quantities.

  int32 cur_frame_;
  std::vector<Vector<BaseFloat> > other_category_outputs_; // Outputs
  // of linear layers for the other categories, on the current frame.
  // We compute these on request.

  // We cache the probabilities, even though most of the computation
  // is cached on each frame, simply because people could call LogLikelihood()
  // many times for the same pdf-id.
  struct LikelihoodCacheRecord {
    BaseFloat log_like;  ///< Cache value
    int32 hit_time;     ///< Frame for which this value is relevant
    LikelihoodCacheRecord(): hit_time(-1) { }
  };
  std::vector<LikelihoodCacheRecord> log_like_cache_;
  
  KALDI_DISALLOW_COPY_AND_ASSIGN(DecodableAmNnet1);
};

}  // namespace kaldi

#endif  // KALDI_NNET_DP_DECODABLE_AM_NNET1_H_
