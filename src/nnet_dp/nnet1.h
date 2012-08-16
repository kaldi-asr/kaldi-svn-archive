// nnet_dp/nnet1.h

// Copyright 2012  Daniel Povey


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

#ifndef KALDI_NNET_DP_NNET1_H_
#define KALDI_NNET_DP_NNET1_H_

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/matrix-lib.h"
#include "nnet_dp/layer.h"

namespace kaldi {

// This class stores the parameters of a neural network.
// This is "neural net 1"-- it's just a particular form of
// neural network.
//   Initialized with:
//      (1) a map from leaf to category-of-leaf. [this is
//   a grouping of the leaves of the decision tree that was obtained
//   during tree building].  This also gives us the #leaves.
//      (2) per-leaf occupancies; these are used to help work out
//    multi-Gaussian stuff..
// Overall configurations:
//    total target for #Gauss, and power e.g. 0.2.
// Properties of each transformation matrix:
//  -- input dimension
//  -- is it switched by the UBM? [true / false] : note, should be false for last layer.
//  -- left temporal context, right temporal context.

// Note: for the Gaussian/UBM selection, this can be separate
// gselect stuff.

// program will need mapping from output-index to Gaussian-index or level-1 index.
//  [relates to mixing-up.]
//  ... during a training epoch we should store occupancies for each Gaussian; we'll
//  just do this for the "correct label".

// mix-up program... creates new Gaussians...

// add-new-layer program... interpolates in a layer somewhere.

// Args to training program:
// training set; validation set [labels and raw features]
//  Learning rate... [note: we'll first normalize by the variance
//     of the gradients for that layer... but not for the last layer.]


// This configuration class is only used when initializing a neural network.
// There are some aspects of the neural net's setup that are not covered in this
// config because they're done at a later stage [UBM switching, mixing-up], or
// because they're worked out from other information [sizes of different
// final layers.]

struct Nnet1InitConfig {
  std::string layer_sizes; // e.g. 23:512:512:512.  Only covers the
  // input and hidden layers; for the last layer, we work out the number of
  // leaves etc.  from other information which is supplied to this program.
  // These sizes do not include the frames of context...
  
  std::string context_frames; // For each layer, the number of (left and right)
  // frames of context at the input.  For instance: 1,1:1,1:1,1:0,0.

  std::string learning_rates; // For each layer, the learning rate.  (colon-separated).
  // If just one element, all the learning rates will be set the same.  Note:
  // these are adjusted during training, we're just setting the initial one.
  
  Nnet1InitConfig() { }
  
  void Register (ParseOptions *po) {
    po->Register("layer-sizes", &layer_sizes,
                 "Sizes of input and hidden units (before splicing), e.g. \"23:512:512:512\"");
    po->Register("context-frames", &context_frames,
                 "For each matrix, the (left,right) frames of temporal context at the input "
                 "E.g. \"1,1:1,1:1,1:0,0\"");
    po->Register("learning-rates", &learning_rates,
                 "Colon-separated list of initial learning rates, one for each layer; or just a single "
                 "learning rate, shared between layers.  Note: not too critical, as learning rates "
                 "are automatically adjusted during training. ");
  }
};

struct Nnet1InitInfo {
  // Suppose n is the number of tanh layers....
  
  std::vector<int32> layer_sizes; // #neurons for input and outputs of tanh layers.  Size n+1.
  std::vector<std::pair<int32, int32> > context_frames; // # context frames for tanh layers.  Size n.
  std::vector<float> learning_rates; // just one learning rate, or a vector, one for
  // each layer, including the last two layers.  Size n+2.

  std::vector<int32> category_sizes;  // category_sizes gives number of
  // labels in each category.  For single-language, will be [# first-level nodes in tree],
  // then for each first-level node that has >1 leaf, the # second-level nodes for that
  // first-level node.

  Nnet1InitInfo(const Nnet1InitConfig &config,
                const std::vector<int32> &category_sizes);
};

// Nnet1 is a sequence of TanhLayers, then a SoftmaxLayer, then a LinearLayer
// [which is similar to Gaussian mixture weights].  The TanhLayers may see
// temporal context of the previous layer; also the final sequence
// (SoftmaxLayer, LinearLayer) may exist in many different versions each with
// different parameters and different numbers of output neurons.  These
// different versions are called "categories" and are used for two things:
// two-level tree [where we predict first a coarse then fine version of the
// leaf-- this helps for efficiency in training], and multilingual experiments.

class Nnet1 {

  // Returns number of linear transforms / number of layers of neurons
  int32 NumLayers() { return initial_layers_.size() + 2; }

  int32 LeftContext() const; // Returns #frames of left context needed [just the sum
  // of left_context for each layer.]

  int32 RightContext() const; // Returns #frames of right context needed [just the sum
  // of right_context for each layer.]

  int32 LeftContextForLayer(int32 layer) const {
    KALDI_ASSERT(layer >= 0 && layer < initial_layers_.size());
    return initial_layers_[layer].left_context;
  }
  int32 RightContextForLayer(int32 layer) const {
    KALDI_ASSERT(layer >= 0 && layer < initial_layers_.size());
    return initial_layers_[layer].right_context;
  }
  int32 LayerIsSpliced(int32 layer) const { // return true if input to
    // layer "layer" is spliced over time.
    return (LeftContextForLayer(layer) + RightContextForLayer(layer) > 0);
  }
  
  // Returns number of layers before the softmax and linear layers.
  int32 NumTanhLayers() const { return initial_layers_.size(); }
  
  int32 NumCategories() const { return final_layers_.size(); }
  
  int32 NumLabelsForCategory(int32 category) const {
    return final_layers_[category].linear_layer->OutputDim();
  }
  
  Nnet1(const Nnet1 &other); // Copy constructor
 
  Nnet1() { }

  Nnet1(const Nnet1InitInfo &init_info) { Init(init_info); }

  ~Nnet1() { Destroy(); }

  void Init(const Nnet1InitInfo &init_info);

  void Destroy();
  
  void Write(std::ostream &os, bool binary) const;

  void Read(std::istream &is, bool binary);

  void Check(); // Checks that the parameters make sense.

  friend class Nnet1Trainer;
 private:
  const Nnet1 &operator = (const Nnet1 &other);  // Disallow assignment.
  
  struct InitialLayerInfo {
    int left_context; // >= 0, left temporal context.
    int right_context; // >= 0, right temporal context.
    TanhLayer *tanh_layer; 
  };
  
  std::vector<InitialLayerInfo> initial_layers_; // Info for the initial
  // layers, indexed by the layer index (0 for the first layer, 1 for
  // the second...)
  
  struct FinalLayerInfo { // Info for the two final layers [softmax and linear.]
    SoftmaxLayer *softmax_layer;
    LinearLayer *linear_layer;
  };
  
  std::vector<FinalLayerInfo> final_layers_;  // Info for the last two layers,
  // indexed by category.
};


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


void SpliceFrames(const MatrixBase<BaseFloat> &input,
                  int32 num_chunks,
                  MatrixBase<BaseFloat> *spliced_out);


void UnSpliceDerivative(const MatrixBase<BaseFloat> &output_deriv,
                        int32 num_chunks,
                        MatrixBase<BaseFloat> *input_deriv);


// This class Nnet1Trainer basically contains functions for
// updating the neural net, given a set of "chunks" of features
// and corresponding labels.  A "chunk" is a short sequence, of size
// fixed in advance [we do it in these chunks for efficiency, because
// with the left and right context, some of the computation would be
// shared.

class Nnet1Trainer {

  // Note: in the case of training with SGD, "nnet" and "nnet_to_update" will
  // be identical.  They'll be different if we're accumulating the gradient
  // for a held-out set and don't want to update the model.
  Nnet1Trainer(const Nnet1 &nnet,
               int32 chunk_size_at_output, // size of chunks (number of output labels).
               int32 num_chunks, // number of chunks we process at the same time.
               Nnet1 *nnet_to_update);

  void TrainStep(const std::vector<TrainingExample> &data);

 private:
  class ForwardAndBackwardFinalClass;
  
  void FormatInput(const std::vector<TrainingExample> &data); // takes the
  // input and formats as a single matrix, in tanh_forward_data[0].

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
  std::vector<int32> chunk_sizes_;
  
  int32 num_chunks_;
  const Nnet1 &nnet_;
  Nnet1 *nnet_to_update_;

  std::vector<Matrix<BaseFloat> > tanh_forward_data_; // The forward data
  // for the input layer [with ones appended, if needed], and for the outputs of
  // the tanh layers.  Indexed by [layer][t][dim]; tanh_forward[i] is the input
  // of layer i.
  
  Matrix<BaseFloat> last_tanh_backward_; // This is used to store the backward derivative for the last
  // tanh layer; for other layers we use the "tanh_data" for both forward and backward but in this
  // case we can't do this (relates to the fact that there are multiple linear/softmax layers).
  
  Matrix<BaseFloat> softmax_data_; // indexed by [t][dim], forward and backward data; we do the categories in sequence, reusing the space.
  Matrix<BaseFloat> linear_data_; // indexed by [t][dim], forward and backward data; we do the categories in sequence, reusing the space.
};


  


} // namespace

#endif
