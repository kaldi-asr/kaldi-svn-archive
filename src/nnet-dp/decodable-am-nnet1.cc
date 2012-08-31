// nnet-dp/decodable-am-nnet1-gmm.cc

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

#include "nnet-dp/decodable-am-nnet1.h"

namespace kaldi {

void DecodableAmNnet1::PrepareInput(const Matrix<BaseFloat> &feats,
                                    Matrix<BaseFloat> *padded_feats) const {
  const Nnet1 &nnet = am_nnet_.Nnet();
  int32 raw_chunk_size = feats.NumRows(), feat_dim = feats.NumCols();
  int32 padded_chunk_size = raw_chunk_size +
      nnet.LeftContext() + nnet.RightContext();
  padded_feats->Resize(padded_chunk_size, feat_dim);
  padded_feats->Range(nnet.LeftContext(), raw_chunk_size,
                      0, feat_dim).CopyFromMat(feats);
  // Duplicate first frames to pad.
  for (int32 i = 0; i < nnet.LeftContext(); i++) {
    SubVector<BaseFloat> src(feats, 0), dest(*padded_feats, i);
    dest.CopyFromVec(src);
  }
  // same for last frames.
  for (int32 i = 0; i < nnet.LeftContext(); i++) {
    SubVector<BaseFloat> src(feats, raw_chunk_size - 1),
        dest(*padded_feats, padded_chunk_size - 1 - i);
    dest.CopyFromVec(src);
  }
}

void DecodableAmNnet1::ForwardInitialLayers(
    const Matrix<BaseFloat> &feats) {
  // Note: this code assumes that all layers are spliced,
  // so if some layer is not spliced we'll do an unnecessary
  // copy of the features, but this really doesn't matter.
  // Note: we just have a single chunk that consists of this
  // entire utterance.
  KALDI_ASSERT(feats.NumRows() != 0);
  const Nnet1 &nnet = am_nnet_.Nnet();

  Matrix<BaseFloat> &prev_layer_output = input_to_softmax_;
  PrepareInput(feats, &prev_layer_output);
  
  int32 num_tanh_layers = nnet.initial_layers_.size(),
      num_chunks = 1;
  for (int32 layer = 0; layer < num_tanh_layers; layer++) {
    const TanhLayer *tanh_layer = nnet.initial_layers_[layer].tanh_layer;
    Matrix<BaseFloat> spliced_input(prev_layer_output.NumRows() -
                                    nnet.LeftContextForLayer(0) -
                                    nnet.RightContextForLayer(0),
                                    tanh_layer->InputDim());
    SpliceFrames(prev_layer_output, num_chunks, &spliced_input);
    prev_layer_output.Resize(spliced_input.NumRows(),
                             tanh_layer->OutputDim() +
                             (layer == num_tanh_layers-1 ? 1 : 0));
    if (layer == num_tanh_layers-1) {// append the unit element, since we won't
      // splice [the splicing code does this]..
      for (int32 i = 0; i < prev_layer_output.NumRows(); i++)
        prev_layer_output(i, prev_layer_output.NumCols()-1) = 1.0;
      SubMatrix<BaseFloat> temp_output(prev_layer_output, 0, prev_layer_output.NumRows(),
                                       0, prev_layer_output.NumCols() - 1);
      // temp_output is prev_layer_output without the last column.
      tanh_layer->Forward(spliced_input, &temp_output);
    } else {
      tanh_layer->Forward(spliced_input, &prev_layer_output);
    }
  }
  KALDI_ASSERT(input_to_softmax_.NumRows() == feats.NumRows());
  // Note: prev_layer_output and input_to_softmax_ are the same variable.

  {
    // Now compute category_zero_output_.
    // Other categories are computed on demand.
    const SoftmaxLayer *softmax_layer = nnet.final_layers_[0].softmax_layer;
    const LinearLayer *linear_layer = nnet.final_layers_[0].linear_layer;   
    Matrix<BaseFloat> softmax_output(input_to_softmax_.NumRows(),
                                     softmax_layer->OutputDim());
    softmax_layer->Forward(input_to_softmax_, &softmax_output);
    category_zero_output_.Resize(softmax_output.NumRows(),
                                 linear_layer->OutputDim());
    linear_layer->Forward(softmax_output, &category_zero_output_);
  }
  am_nnet_.GetPriors(&neg_log_priors_);
  neg_log_priors_.ApplyLog();
  neg_log_priors_.Scale(-1.0);
  cur_frame_ = -1;
  log_like_cache_.resize(am_nnet_.NumPdfs());
}

BaseFloat DecodableAmNnet1::LogLikelihood(int32 frame,
                                          int32 transition_id) {
  int32 pdf_id = trans_model_.TransitionIdToPdf(transition_id);  
  KALDI_ASSERT(frame >= 0.0 && frame < NumFrames());
  const Nnet1 &nnet = am_nnet_.Nnet();
  if (frame != cur_frame_) { // invalidate other_category_outputs_.
    cur_frame_ = frame;
    other_category_outputs_.clear();
    other_category_outputs_.resize(nnet.NumCategories());
  } else {
    if (log_like_cache_[pdf_id].hit_time == frame)
      return log_like_cache_[pdf_id].log_like;
  }
  std::vector<std::pair<int32, int32> > pairs;
  am_nnet_.GetCategoryInfo(pdf_id, &pairs);
  // pairs is a set of pairs (category, label-within-category).
  // It will always contain an entry for category zero, and will usually
  // also contain one more entry for another category.
  BaseFloat prob = 1.0; // It's actually a log-prob rather than a log-likelihood,
  // with this model, but it's a standard interface.
  for (int32 i = 0; i < pairs.size(); i++) {
    int32 category = pairs[i].first, label = pairs[i].second;
    if (category == 0) {
      prob *= category_zero_output_(frame, label);
    } else {
      KALDI_ASSERT(category > 0 && category < other_category_outputs_.size());
      Matrix<BaseFloat> &post_vec(other_category_outputs_[category]); // vector
      // of posteriors in this category.  [matrix with one row].
      if (post_vec.NumRows() == 0) { // do the computation.
        const SoftmaxLayer *softmax_layer = nnet.final_layers_[category].softmax_layer;
        const LinearLayer *linear_layer = nnet.final_layers_[category].linear_layer;
        SubMatrix<BaseFloat> softmax_input(input_to_softmax_, frame, 1,
                                           0, input_to_softmax_.NumCols());
        Matrix<BaseFloat> softmax_output(1, softmax_layer->OutputDim());
        softmax_layer->Forward(softmax_input, &softmax_output);
        post_vec.Resize(1, linear_layer->OutputDim());
        linear_layer->Forward(softmax_output, &post_vec);
      }
      prob *= post_vec(0, label);
    }
  }
  BaseFloat floor = 1.0e-20;
  if (prob < floor) {
    KALDI_WARN << "Flooring probability " << prob << " to " << floor;
    prob = floor;
  }
  BaseFloat ans = log(prob) + neg_log_priors_(pdf_id);
  log_like_cache_[pdf_id].hit_time = frame;
  log_like_cache_[pdf_id].log_like = ans;
  return ans;
}

}  // namespace kaldi
