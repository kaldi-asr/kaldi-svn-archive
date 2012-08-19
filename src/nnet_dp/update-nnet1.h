// nnet_dp/update_nnet1.h

// Copyright 2012  Johns Hopkins University (author: Daniel Povey)
//                 Navdeep Jaitly

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

#ifndef KALDI_NNET_DP_UPDATE_NNET1_H_
#define KALDI_NNET_DP_UPDATE_NNET1_H_

#include "nnet_dp/nnet1.h"

namespace kaldi {


// TrainingExample is the labels and input supervision for
// one chunk of the input.  The "chunk size" is a small fixed
// number, e.g. 5 or 10, and is the length of the "labels"
// vector.  The "input" data member has more rows than the
// chunk size-- it has nnet1.LeftContext() extra rows on the
// left and nnet1.RightContext() extra rows on the right.

struct TrainingExample {
  BaseFloat weight; // Allows us to put a weight on each training
  // sample.

  std::vector<std::vector<std::pair<int32, int32> > > labels;
  // each element of "labels" is a list of pairs (category, label).
  // For a typical, monolingual setup with a two-level tree we'll
  // have two pairs: something
  // like ((0, first-level-tree-node),
  // (first-level-tree-node, second-level-tree-node))-- although
  // there may be some integer offsets involved here.
  
  Matrix<BaseFloat> input; // The input data; will typically have more rows
  // than the size of "labels", due to required context.  Context that does
  // not exist due to frame splicing, will be given a zero value but still
  // supplied.  
};


// This class Nnet1Updater basically contains functions for
// updating the neural net, given a set of "chunks" of features
// and corresponding labels.  A "chunk" is a short sequence, of size
// fixed in advance [we do it in these chunks for efficiency, because
// with the left and right context, some of the computation would be
// shared.

class Nnet1Updater {
 public:

  // Note: in the case of training with SGD, "nnet" and "nnet_to_update" will
  // be identical.  They'll be different if we're accumulating the gradient
  // for a held-out set and don't want to update the model.
  Nnet1Updater(const Nnet1 &nnet,
               int32 chunk_size_at_output, // size of chunks (number of output labels).
               int32 num_chunks, // number of chunks we process at the same time.
               Nnet1 *nnet_to_update);

  void TrainOnOneMinibatch(const std::vector<TrainingExample> &data);
 private:
  class ForwardAndBackwardFinalClass;
  
  void FormatInput(const std::vector<TrainingExample> &data); // takes the
  // input and formats as a single matrix, in tanh_forward_data_[0].

  void ForwardTanh(); // Does the forward computation for the initial (tanh)
  // layers.

  void BackwardTanh(); // Does the backward computation for the initial (tanh)
  // layers.

  
  // This function gets lists of which categories are referenced in the
  // training data "data".  It outputs as two lists: one "common_categories"
  // that are referenced on every single frame; and one "other_categories"
  // that are not, and which will be listed in increasing order of
  // frequency.
  static void ListCategories(const std::vector<TrainingExample> &data,
                             std::vector<int32> *common_categories,
                             std::vector<int32> *other_categories);

  double ForwardAndBackwardFinal(const std::vector<TrainingExample> &data);
  // Does the forward and backward computation for the final two layers (softmax
  // and linear).  Note: returns summed, weighted log-prob.
  
  // Does the forward and backward computation for the final two layers (softmax
  // and linear), but just considering one of the categories of output labels.
  double ForwardAndBackwardFinalForCategory(const std::vector<TrainingExample> &data,
                                            int32 category,
                                            bool common_category);

  // Called inside ForwardAndBackwardFinalForCategory, which just handles
  // some mapping issues and then calls this.
  double ForwardAndBackwardFinalInternal(
      const Matrix<BaseFloat> &input, // input to softmax layer
      int32 category,
      const std::vector<BaseFloat> &weights, // one per example.
      const std::vector<int32> &labels, // one per example
      Matrix<BaseFloat> *input_deriv); //derivative w.r.t "input".
  
  // The vector chunk_sizes_ gives the sizes of the chunks at each layer, with
  // chunk_sizes_[0] being the size of the chunk at input, and
  // chunk_sizes_.back() being the size of the chunk at output, which is the
  // same as chunk_size_at_output.  The chunk sizes at earlier layers may be
  // larger, due to frame splicing.  Note: each chunk size is the chunk size at
  // the *input* to that layer, before frame splicing; when we do frame
  // splicing, the chunk sizes will get smaller and be the same as the chunk
  // size at the output to that layer [i.e. the input to the next layer].  
  
  const Nnet1 &nnet_;
  int32 num_chunks_;
  Nnet1 *nnet_to_update_;
  
  std::vector<Matrix<BaseFloat> > tanh_forward_data_; // The forward data
  // for the input layer [with ones appended, if needed], and for the outputs of
  // the tanh layers.  Indexed by [layer][t][dim]; tanh_forward[i] is the input
  // of layer i.
  
  Matrix<BaseFloat> last_tanh_backward_; // This is used to store the backward derivative
  // at the input of the last tanh layer; for other layers we use the
  // "tanh_data" for both forward and backward but in this case we can't do this
  // (relates to the fact that there are multiple linear/softmax layers).
};

void SpliceFrames(const MatrixBase<BaseFloat> &input,
                  int32 num_chunks,
                  MatrixBase<BaseFloat> *spliced_out);


void UnSpliceDerivative(const MatrixBase<BaseFloat> &output_deriv,
                        int32 num_chunks,
                        MatrixBase<BaseFloat> *input_deriv);


} // namespace

#endif // KALDI_NNET_DP_TRAIN_NNET1_H_
