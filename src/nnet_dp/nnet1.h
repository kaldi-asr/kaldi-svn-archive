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
#include "matrix/matrix-lib.h"

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
  std::string layer_size; // e.g. 23:512:512:512.  Only covers the
  // input and hidden layers; we work out the number of leaves etc.
  // from other information which is supplied to this program.  does
  // not include the frames of context...
  
  std::string context_frames; // For each layer, the number of (left and right)
  // frames of context at the input.  For instance: 1,1:1,1:1,1:0,0.

  Nnet1InitConfig(): num_ubm_gauss(0) { }
  
  void Register (ParseOptions *po) {
    po->Register("layer-size", &layer_size,
                 "Sizes of input and hidden units (before splicing), e.g. \"23:512:512:512\"");
    po->Register("context-frames", &context_frames,
                 "For each matrix, the (left,right) frames of temporal context at the input "
                 "E.g. \"1,1:1,1:1,1:0,0\"");
  }
};


// Nnet1 is a sequence of TanhLayers, then a SoftmaxLayer, then a LinearLayer [which
// is similar to Gaussian mixture weights].  But it's a bit more complicated than
// this, because the TanhLayers may be "switched" by UBM Gaussians so the parameters
// we use are dependent on the place in acoustic space; also the TanhLayers and the
// SoftmaxLayer may see temporal context of the previous layer; also the final
// sequence (SoftmaxLayer, LinearLayer) may exist in many different versions each with
// different parameters and different numbers of output neurons.  These different
// versions are called "categories" and are used for two things: two-level tree
// [where we predict first a coarse then fine version of the leaf-- this helps for
//  efficiency in training], and multilingual experiments.

class Nnet1 {

  // Returns number of linear transforms / number of layers of neurons
  int32 NumLayers() { return initial_layers_.size() + 2; }

  // Returns number of layers before the softmax and linear layers.
  int32 NumTanhLayers() { return initial_layers_.size(); }

  int32 NumCategoriesLastLayer();



  struct InitialLayerInfo {
    int left_context; // >= 0, left temporal context.
    int right_contet; // >= 0, right temporal context.
    std::vector<TanhLayer*> layers; // one for each UBM Gaussian
    // or just a vector of size one, if not using switching.
  };

  std::vector<InitialLayerInfo> initial_layers_;

  struct FinalLayerInfo {
    int left_context; // >= 0, left temporal context for softmax layer.
    int right_context; // >= 0, right temporal context for softmax layer.

    // Note: the sigmoid_layers and linear_layers are indexed by the
    // same index; this corresponds to the "category" of output
    // label.
    std::vector<SoftmaxLayer*> softmax_layers;
    std::vector<LinearLayer*> linear_layers;
  };

  std::vector<FinalLayerInfo> final_layers_;    
  


  Nnet1() { }

  void Init(const Nnet1Config &config,
            std::vector<int32> category_sizes); // category_sizes gives number of
  // labels in each category.  For single-language, will be [# first-level nodes in tree],
  // then for each first-level node that has >1 leaf, the # second-level nodes for that
  // first-level node.
  
  int32 num_ubm_gauss_; // Relates to switching by the UBM: the number of Gaussians
  // in the UBM.
  
  vector<int32> layer_size_no_last_; // For the input layer and the hidden layers,
  // this gives the sizes (raw, before splicing.)  Note: for
  // the input, if layer_size_ is the actual input size + 1, we append a 1.
  
  vector<int32> last_layer_sizes_; // Sizes of the different "categories"
  // of output at the last layer.
    
  vector<bool> switched_; // For each layer, true if switched by the UBM.
  vector<int> left_context_frames_; // Number of left frames of context at the
  // input to each layer.
  vector<int> right_context_frames_; // Number of left frames of context at the
  // input to each layer.

  // For all but final frames,
  // indexed [layer][UBM-index][output-index][input-index].
  // Use UBM-index = 0 for non-switched layers.

  // For final frame, indexed [category][output-index][input-index].
  // The "category" is the category of things we want to predict; this
  // mostly relates to multi-level trees but also to multi-lingual.
  // Basically we have soft-max within each category.

  std::vector<std::vector<Matrix<BaseFloat> > > data_; // the parameters of the
  // neural net.

  // Indexed by layer, then (UBM-index or, for last layer, the category);
  // then by the neuron index; each "gradient_variance" is the
  // average variance of the derivatives w.r.t. the input 
  // weights to that neuron.  The average is over the input weights,
  // and then a weighted average over time.
  // This is only used if --use-fisher-preconditioning is true (yes by default).
  // We keep this as part of the neural net class because we generally
  // want to read and write it with the neural net itself.
  std::vector<std::vector<Vector<double> > > gradient_variance_;

  // Derived from gradient_variance [inversed and copied, each time
  // we update gradient_variance].
  std::vector<std::vector<Vector<float> > > inverse_gradient_variance_;
  void ComputeInverseGradientVariance(); // derive inverse_gradient_variance_
  // from gradient_variance_.

  void Write(std::ostream &os, bool binary) const;

  void Read(std::istream &is, bool binary);  
};


struct Nnet1LearningRateConfig {
  std::string learning_rates; // Learning rate for each layer: floats, separated by colon.
  // Note: if --use-fisher-preconditioning is true, these get multiplied by the inverse
  // of the scatter of gradients.
  
  bool use_fisher_preconditioning;
  
  double fisher_average_frames; // Number of observations
  // over which to average the Fisher information.  Needs to be
  // comfortably more than the number of classes, in order to
  // get a smooth average.
  
  Nnet1LearningRateConfig(): use_fisher_preconditioning(true),
                             fisher_average_frames(20000) { }
  
  void Register (ParseOptions *po) {
    po->Register("learning-rates", &learning_rates,
                 "Learning rates for layers (from bottom).");
    po->Register("use-fisher-preconditioning",
                 &use_fisher_preconditioning, "If true, use a preconditioning for the "
                 "learning rate that's based on the diagonal of the Fisher matrix.");
  }
  
  vector<BaseFloat> learning_rates_float; // Vector that's set up from "learning_rates".
  
}
  
struct Nnet1LearningRatePolicy {
  Nnet1LearningRatePolicy(const Nnet1 &nnet1,
                          const Nnet1LearningRateConfig &config);
  

  vector<BaseFloat> layer_learning_rates; // A global factor on
  // the learning rates for each layer.  These might all be the same.
  
  bool use_fisher_preconditioning; // If true, use our diagonal-Fisher preconditioning:
  // note, this is applied at the level of each neuron (we average over all
  // the parameters that weight the inputs for that neuron), so it's really
  // a diagonal matrix with certain sets of the diagonal elements constrained
  // to be the same.
  
  double fisher_average_frames; // Number of frames to average the
  // variance of derivatives over for the Fisher preconditioning.  
};
  




// This class stores temporary variables for
// the forward and backward passes of calculation
// for the neural net Nnet1.
class Nnet1Calculation {
  // This initializer sets up the sizes and the storage.
  //
  Nnet1Calculation(Nnet1 &nnet1,
                   const Matrix<BaseFloat> &feats,
                   const std::vector<int> &gselect, // one-best Gaussian from Gaussian selection.
                   int start_frame_at_output,
                   int end_frame_at_output,
                   bool do_backward):
      nnet1_(nnet1) {
    // TODO: set up sizes and data_ with storage.
  }

  void ForwardCalculationNoLastLayer(); // Forward calculation, but not the last layer.
  
  const Vector<BaseFloat>& LastLayerOutput(int32 t, int32 category); // t must be
  // >= start_frame_at_output [as supplied to the initializer] and < end_frame_at_output;
  // this function will compute on-demand
  
  // This function AddSupervision says we know the label for this category.
  // [typically "posterior" will be 1.0; in discriminative training it may take
  // other values.]
  void AddSupervision(int32 t, int32 category, BaseFloat posterior = 1.0);
  
  // Does the back-propagation: it's plain SGD, with the given learning rate
  // for each layer dictated by policy.layer_learning_rate.
  void BackpropSgd(const Nnet1LearningRatePolicy &policy,
                   Nnet1 *nnet1);
  
  // Does the back-propagation: it's plain SGD, with the given learning rate,
  // but with additional preconditioning by the inverse of the Fisher information
  // matrix; this routine also has to keep the fisher information updated.
  void BackpropSgdFisher(const Nnet1LearningRatePolicy &policy,
                         Nnet1FisherInfo *fisher,
                         Nnet1 *nnet1);
  

  // This function treats the "nnet1" in the pointer as a place to put
  // the sum of the gradients; it would be initialized with zero parameters.
  // This may be used to get the gradient on the validation set.
  void BackpropGradient(Nnet1 *nnet1);
  
  
 private:

  // The vector start_frame contains, indexed by layer, the first
  // frame for which we have the input of that layer.  The last
  // element is the 1st frame for which we have the output of the network.
  vector<int32> start_frame_;
  // The vector start_frame contains, indexed by layer, the (last+1)
  // frame for which we have the input of that layer.
  vector<int32> end_frame_;

  vector<Matrix<BaseFloat> > data_; // for layer l, the input of that
  // layer is data_[l][f], where f is a frame index (f=0 is
  // start_frame_[l].  This goes up to nnet1_.NumLayers()-1.
  // The very last layer of output is stored differently,

  // output_data_ stores the very last layer of output.  It's in
  // a slightly different format because we index by the
  // "output class".  Note: we don't actually size these vectors
  // until they're needed, in order to avoid wasting time and
  // memory.  
  vector<vector<Vector<BaseFloat> > > output_data_;
  
  const Nnet1 &nnet1_;

  // The next data structure is a vector (indexed by t -
  // start_frame_at_output), of a vector [indexed by output class]
  // of a list of (label in class, 
  std::vector<std::vector<std::pair<int32, BaseFloat> > > supervision_;
  
};




} // namespace

#endif

