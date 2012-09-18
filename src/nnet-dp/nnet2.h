// nnet-dp/nnet2.h

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

#ifndef KALDI_NNET_DP_NNET2_H_
#define KALDI_NNET_DP_NNET2_H_

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/matrix-lib.h"
#include "nnet-dp/layer.h"

namespace kaldi {

// nnet2 is a neural network that contains a number of generic
// nonlinearities such as tanh, with configurable sizes, possibly
// with the last of these being a bit smaller (such as 200), in
// order to balance the #parameters since the next dimension will
// be very large.
// Then we have a big softmax layer, with typically 10k outputs,
// which can roughly be thought of as the Gaussians of the system
// and which initially correspond with the output indexes.  Then
// we have a large number of LinearLayers, which correspond with
// mixture indices.  This is like a state-clustered tied mixture
// (SCTM) system.
//
// Note: for each of the tanh layers, we actually support a setup
// where the linear layer is split into N parts to reduce the
// number of parameters.  If it's split this way, we'll permute the
// output indices...

struct Nnet2InitConfig {
  std::string layer_sizes; // e.g. 23:512:512:512.  Only covers the
  // input and hidden layers; for the last layer, we work out the number of
  // leaves etc.  from other information which is supplied to this program.
  // These sizes do not include the frames of context...
  
  std::string context_frames; // For each layer, the number of (left and right)
  // frames of context at the input.  For instance: 1,1:1,1:1,1:0,0.

  std::string learning_rates; // For each layer, the learning rate.  (colon-separated).
  // If just one element, all the learning rates will be set the same.  Note:
  // these are adjusted during training, we're just setting the initial one.

  BaseFloat diagonal_element; // Diagonal element we initialize linear_layer with.
  // E.g. 0.9.

  BaseFloat shrinkage_rate;
  
  Nnet1InitConfig(): diagonal_element(0.9), shrinkage_rate(1.0e-07) { }
  
  void Register (ParseOptions *po) {
    po->Register("layer-sizes", &layer_sizes,
                 "Sizes of input and hidden units (before splicing + bias), e.g. \"23:512:512:512\"");
    po->Register("context-frames", &context_frames,
                 "For each matrix, the (left,right) frames of temporal context at the input "
                 "E.g. \"1,1:1,1:1,1:0,0\"");
    po->Register("learning-rates", &learning_rates,
                 "Colon-separated list of initial learning rates, one for each layer; or just a single "
                 "learning rate, shared between layers.  Note: not too critical, as learning rates "
                 "are automatically adjusted during training. ");
    po->Register("diagonal-element", &diagonal_element,
                 "Diagonal element used when initializing linear layer");
    po->Register("shrinkage-rate", &shrinkage_rate,
                 "Initial shrinkage rate for all layers.");
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

  BaseFloat diagonal_element; // diagonal element when initializing linear
                              // layer, > 0.0, <= 1.0.
  BaseFloat shrinkage_rate; // for all layers.
  
  Nnet1InitInfo(const Nnet1InitConfig &config,
                const std::vector<int32> &category_sizes);
};

struct Nnet1ProgressInfo;

// Nnet1 is a sequence of TanhLayers, then a SoftmaxLayer, then a LinearLayer
// [which is similar to Gaussian mixture weights].  The TanhLayers may see
// temporal context of the previous layer; also the final sequence
// (SoftmaxLayer, LinearLayer) may exist in many different versions each with
// different parameters and different numbers of output neurons.  These
// different versions are called "categories" and are used for two things:
// two-level tree [where we predict first a coarse then fine version of the
// leaf-- this helps for efficiency in training], and multilingual experiments.

class Nnet1 {
 public:
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

  int32 InputDim() const { // Input dimension of raw, un-spliced features.
    // Note: there is no OutputDim(), for each category we have
    // NumLabelsForCategory(category).
    // Note: the -1 is account for the unit element (1.0) which we append.
    return (initial_layers_[0].tanh_layer->InputDim() - 1) /
        (1 + LeftContextForLayer(0) + RightContextForLayer(0));
  }
  
  // Returns number of layers before the softmax and linear layers.
  int32 NumTanhLayers() const { return initial_layers_.size(); }
  
  int32 NumCategories() const { return final_layers_.size(); }

  int32 NumLabelsForCategory(int32 category) const {
    return final_layers_[category].linear_layer->OutputDim();
  }
  
  void ZeroOccupancy(); // calls ZeroOccupancy() on the softmax layers.  This
  // resets the occupancy counters; it makes sense to do this once in
  // a while, e.g. at the start of an epoch of training.
  
  Nnet1(const Nnet1 &other); // Copy constructor.
  
  Nnet1() { }

  Nnet1(const Nnet1InitInfo &init_info) { Init(init_info); }

  ~Nnet1() { Destroy(); }

  // Add a new tanh layer (hidden layer).  
  // Use #nodes of top hidden layer.  The new layer will have zero-valued parameters
  void AddTanhLayer(int32 left_context, int32 right_context,
                    BaseFloat learning_rate);

  // Combines with another neural net model, as weighted combination.
  // other_weight = 0.5 means half-and-half.  this_weight will be
  // 1.0 - other_weight.
  // Keeps learning rates of *this.
  void CombineWithWeight(const Nnet1 &other, BaseFloat other_weight);

  std::string Info() const; // some human-readable summary info.

  // some human-readable summary info (summed over final-layers.)
  std::string Info(const std::vector<std::vector<int32> > &final_sets) const;

  std::string LrateInfo() const; // some info on the learning rates,
  // in human-readable form.

  // the same, broken down by sets.
  std::string LrateInfo(const std::vector<std::vector<int32> > &final_sets)
      const;

  // the same, broken down by sets, for shrinkage rates.
  std::string SrateInfo(const std::vector<std::vector<int32> > &final_sets)
      const;
  
  // Mix up by increasing the dimension of the output of softmax layer (and the
  // input of the linear layer).  This is exactly analogous to mixing up
  // Gaussians in a GMM-HMM system, and we use a similar power rule to allocate
  // new ones [so a "category" gets an allocation of indices/Gaussians allocated
  // proportional to a power "power" of its total occupancy.
  void MixUp(int32 target_tot_neurons,
             BaseFloat power, // e.g. 0.2.
             BaseFloat perturb_stddev);
  
  void Init(const Nnet1InitInfo &init_info);

  void Destroy();
  
  void Write(std::ostream &os, bool binary) const;

  void Read(std::istream &is, bool binary);

  void SetZeroAndTreatAsGradient(); // Sets all parameters to zero and the
  // learning rates to 1.0 and shinkage rates to zero.  Mostly useful if this
  // neural net is just being used to store the gradients on a validation set.
  // Will let the contents know that we'll now treat the layers as a store of
  // gradients.  [affects the LinearLayer.]  If so you probably want

  // This is used to separately adjust learning rates of each layer,
  // after each "phase" of training.  We basically ask (using the validation
  // gradient), do we wish we had gone further in this direction?  Yes->
  // increase learning rate, no -> decrease it.
  void AdjustLearningAndShrinkageRates(
      const Nnet1ProgressInfo &start_dotprod, // dot-prod of param@start with valid-grad@end
      const Nnet1ProgressInfo &end_dotprod, // dot-prod of param@end with valid-grad@end
      const std::vector<std::vector<int32> > &final_layer_sets,
      BaseFloat learning_rate_ratio,
      BaseFloat max_learning_rate,
      BaseFloat min_shrinkage_rate,
      BaseFloat max_shrinkage_rate);

  
  // This sets *info to the dot prod of *this . validation_gradient.
  // This is used in updating learning rates and shrinkage rates.
  void ComputeDotProduct(
      const Nnet1 &validation_gradient,
      Nnet1ProgressInfo *info) const;
  
  // Outputs the priors for a particular category of labels, as
  // computed from the "occupancy" statistics stored with the softmax
  // layer.  This should rougly match the priors seen during training.
  void GetPriorsForCategory(int32 category,
                            Vector<BaseFloat> *priors) const;
  
  friend class Nnet1Updater;
  friend class DecodableAmNnet1;
 private:
  const Nnet1 &operator = (const Nnet1 &other);  // Disallow assignment.
  
  struct InitialLayerInfo {
    // should probably use short for left and right context?
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



/*
  This function interprets the input as a number "num_chunks" of
  equally sized sequences of frames (the rows of "input" correspond
  to frames).  We do context-splicing and append the unit element when
  creating the matrix "spliced_out", i.e. each row of "spliced_out" will
  be a sequence of frames of "input", followed by the unit element.
  The splicing is done only within the chunks, and we lose the outer
  frames within each chunk.  The function works out the #frames
  to splice from the sizes of "input" and "output".
 */
void SpliceFrames(const MatrixBase<BaseFloat> &input,
                  int32 num_chunks,
                  MatrixBase<BaseFloat> *spliced_out);


/* This does essentially the opposite of SpliceFrames, but when
   backpropagating the derivatives.  There is summation involved here,
   in the same place where SpliceFrames would do duplication.
*/
void UnSpliceDerivative(const MatrixBase<BaseFloat> &output_deriv,
                        int32 num_chunks,
                        MatrixBase<BaseFloat> *input_deriv);

struct Nnet1ProgressInfo {
  // We use this structure to store certain information on the performance and
  // dot-products of the validation-set gradient with a change in parameters.
  std::vector<BaseFloat> tanh_dot_prod; // dot prod of validation objf with
  // parameter change for tanh layers.
  std::vector<BaseFloat> softmax_dot_prod; // same for softmax layers.
  std::vector<BaseFloat> linear_dot_prod; // same for softmax layers.
  
  std::string Info() const; // the info in human-readable form.

  // some human-readable summary info (summed over final-layers.)
  std::string Info(const std::vector<std::vector<int32> > &final_sets) const;

  void Add(const Nnet1ProgressInfo &other, BaseFloat scale);
  void Scale(BaseFloat scale);
};


} // namespace

#endif
