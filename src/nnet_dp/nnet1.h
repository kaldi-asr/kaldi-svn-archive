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
    po->Register("layer-size", &layer_size,
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

  int32 LeftContext(); // Returns #frames of left context needed [just the sum
  // of left_context for each layer.]

  int32 RightContext(); // Returns #frames of right context needed [just the sum
  // of right_context for each layer.]
  
  // Returns number of layers before the softmax and linear layers.
  int32 NumTanhLayers() { return initial_layers_.size(); }
  
  int32 NumCategories();

  int32 NumLabelsForCategory(int32 category);
  
  Nnet1(const Nnet1 &other); // Copy constructor
  
  const Nnet1 &operator = (const Nnet1 &other) = 0; // Disallow assignment.
  
  Nnet1() { }

  Nnet1(const Nnet1Config &config,
        const std::vector<int32> &category_sizes) { Init(config, category_sizes); }
  
  ~Nnet1() { Destroy(); }

  void Init(const Nnet1Config &config,
            const std::vector<int32> &category_sizes); // category_sizes gives number of
  // labels in each category.  For single-language, will be [# first-level nodes in tree],
  // then for each first-level node that has >1 leaf, the # second-level nodes for that
  // first-level node.

  void Destroy();
  
  void Write(std::ostream &os, bool binary) const;

  void Read(std::istream &is, bool binary);

  void Check(); // Checks that the parameters make sense.
  
 private:
  struct InitialLayerInfo {
    int32 num_ubm_gauss; // Relates to switching by the UBM: the number of Gaussians
    // in the UBM.  It is set to one if we don't use this feature.  Note:
    // for now, these would be the same for all layers that don't use UBM switching.
    
    int left_context; // >= 0, left temporal context.
    int right_contet; // >= 0, right temporal context.
    std::vector<TanhLayer*> layers; // one for each UBM Gaussian
    // or just a vector of size one, if not using switching.
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
  std::vector<Matrix<BaseFloat> > tanh_data; // The forward and also backward data; indexed by [layer][t][dim];
  // tanh_forward[i] is the input of layer i.
  
  Matrix<BaseFloat> last_tanh_backward; // This is used to store the backward derivative for the last
  // tanh layer; for other layers we use the "tanh_data" for both forward and backward but in this
  // case we can't do this (relates to the fact that there are multiple linear/softmax layers).
  
  Matrix<BaseFloat> softmax_data; // indexed by [t][dim], forward and backward data; we do the categories in sequence, reusing the space.
  Matrix<BaseFloat> linear_data; // indexed by [t][dim], forward and backward data; we do the categories in sequence, reusing the space.
};


  
  

  


  Nnet1Stats(Nnet1 *nnet_to_update,
             int32 output_chunk_size);

  ~Nnet1Stats();

  friend class Nnet1Foo; // Not sure who.
 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(Nnet1Stats);
  Nnet1 *nnet_to_update;
  
  std::vector<std::vector<TanhLayerStats*> > tanh_stats_; // indexed by [layer index]
  // [UBM-Gaussian], stats for initial layer.
  std::vector<SoftmaxLayerStats*> softmax_stats_; // indexed by category; for the
  // last-but-one layer.
  std::vector<LinearLayerStats*> linear_stats_; // indexed by category; for the
  // final layer.
};


class Nnet1ForwardComputation {
  // This class handles the "forward" computation for the neural net.  Note:
  // it's for a number of consecutive frames, not just one frame.

  // Initialize the arrays, etc. 
  Nnet1ForwardComputation(const Nnet1 &nnet1,
                          int32 num_frames_output);

  // Do the forward computation for a particular input.  Note regarding left and
  // right context: the input is expected to be either of size (num_frames_output
  //  + nnet1.LeftContext() + nnet1.RightContext()), in which case it's the actual
  // input, or of size (num_frames_output), in which case we pad with zero input.
  void Forward(const Matrix<BaseFloat> &input);
  
  const Matrix<BaseFloat> &GetOutput();

 private:
  
  Matrix<BaseFloat> input_; // The input data.
  std::vector<Matrix<BaseFloat> > initial_data_; // The first element
  
};

// This class stores temporary variables for
// the forward and backward passes of calculation
// for the neural net Nnet1.
class Nnet1alculation {
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

