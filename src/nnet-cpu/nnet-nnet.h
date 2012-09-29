// nnet-cpu/nnet-nnet.h

// Copyright 2011-2012  Karel Vesely
//                      Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_NNET_CPU_NNET_NNET_H_
#define KALDI_NNET_CPU_NNET_NNET_H_

#include "base/kaldi-common.h"
#include "util/kaldi-io.h"
#include "matrix/matrix-lib.h"
#include "nnet-cpu/nnet-component.h"

#include <iostream>
#include <sstream>
#include <vector>


namespace kaldi {


/*
  This neural net is basically a series of Components, and is a fairly
  passive object that mainly acts as a store for the Components.  Training
  is handled by a separate class NnetTrainer(), and extracting likelihoods
  for decoding is handled by DecodableNnetCpu(). 
  
  There are a couple of things that make this class more than just a vector of
  Components.

   (1) It handles frame splicing (temporal context.)
   We'd like to encompass the approach described in
   http://www.fit.vutbr.cz/research/groups/speech/publi/2011/vesely_asru2011_00042.pdf
   where at a certain point they splice together frames -10, -5, 0, +5 and +10.  It
   seems that it's not necessarily best to splice together a contiguous sequence
   of frames.

   (2) It handles situations where the input features have two parts--
   a "frame-specific" part (the normal features), and a "speaker-specific", or at
   least utterance-specific part that does not vary with the frame index.
   These features are provided separately from the frame-specific ones, to avoid
   redundancy.
*/


class Nnet {
 public:
  /// Returns number of components-- think of this as similar to # of layers, but
  /// e.g. the nonlinearity and the linear part count as separate components.
  int32 NumComponents() const { return components_.size(); }

  /// For 0 <= component < NumComponents(), gives a vector of the
  /// relative frame indices that are spliced together at the input of
  /// this component.  For typical hidden layers will just be the
  /// vector [ 0 ].  For component == NumComponents(), returns the
  /// vector [ 0 ]; this happens to be convenient.
  const std::vector<int32> &RawSplicingForComponent(int32 component) const;

  /// Returns the relative frame indices at the input of this component that
  /// we'll need to get a single frame of the output.  This is a function of
  /// RawSplicingForComponent() for this component and later ones.  For
  /// component == NumComponents(), returns the vector [ 0 ]; this happens to be
  /// convenient.
  const std::vector<int32> &FullSplicingForComponent(int32 component) const;

  /// This returns a matrix that lets us know in a convenient form how to splice
  /// together the outputs of the previous layer (or the frame-level input to
  /// the network) to become the input of this layer.  To be more precise: For a
  /// particular component index 0 <= c < NumComponents(), this returns an array
  /// of size M by N, where M equals FullSplicingForComponent(c+1).size(), 
  /// and N equals RawSplicingForComponent(c).size().  This array specifies how
  /// we splice together the list of frames at the output of the previous layer
  /// (or the raw input), into the list of frames at the input of this layer.
  /// All the values in this
  /// array are positive, unlike the results of RawSplicingForComponent() and
  /// FullSplicingForComponent(), because they are not frame indexes but indexes
  /// into the result that FullSplicingForComponent() returns.
  /// See the comment above the code for Nnet::InitializeArrays() for an example.
  const std::vector<std::vector<int32> > &RelativeSplicingForComponent(
      int32 component) const;

  /// The output dimension of the network -- typically
  /// the number of pdfs.
  int32 OutputDim() const; 

  /// Dimension of the per-frame input features, e.g. 13.
  /// Does not take account of splicing-- see FullSplicingForComponent(0) to
  /// see what relative frame indices you'll need (typically a contiguous
  /// stretch but we don't guarantee this.  
  int32 FeatureDim() const; 

  /// Dimension of any speaker-specific or utterance-specific
  /// input features-- the dimension of a vector we provide that says something about
  /// the speaker.  This will be zero if we don't give any speaker-specific information
  /// to the network.
  int32 SpeakerDim() const { return speaker_info_dim_; }
  
  const Component &GetComponent(int32 component) const;

  /// Number of frames of left context the network needs.
  int32 LeftContext() { return -FullSplicingForComponent(0).front(); }


  /// Number of frames of right context the network needs.
  int32 RightContext() { return FullSplicingForComponent(0).back(); }

  Component &GetComponent(int32 component);
  
  void ZeroOccupancy(); // calls ZeroOccupancy() on the softmax layers.  This
  // resets the occupancy counters; it makes sense to do this once in
  // a while, e.g. at the start of an epoch of training.
  
  Nnet(const Nnet &other); // Copy constructor.
  
  Nnet() { }

  /// Initialize from config file.
  /// Each line of the config is either a comment line starting
  /// with whitespace then #, or it is a vector specifying splicing to the
  /// input of this layer, e.g. [ -1 0 1 ] or [ 0 ] followed by
  /// a string that would initialize a Component: for example,
  /// AffineComponent 0.01 0.001 1000 1000.
  void InitFromConfig(std::istream &is); 
  
  ~Nnet() { Destroy(); }

  /*  
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
  */
  void Destroy();
  
  void Write(std::ostream &os, bool binary) const;

  void Read(std::istream &is, bool binary);

  void SetZero(bool treat_as_gradient); // Sets all parameters to zero and if
  // treat_as_gradient == true, also sets the learning rates to 1.0 and shinkage
  // rates to zero and instructs the components to think of themselves as
  // storing the gradient (this part only affects components of type
  // LinearComponent).


  /*
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
  */
  
  // This sets *info to the dot prod of *this . validation_gradient,
  // separately for each component.
  // This is used in updating learning rates and shrinkage rates.
  void ComputeDotProduct(
      const Nnet &validation_gradient,
      VectorBase<BaseFloat> *dot_prod) const;
  
  friend class NnetUpdater;
  friend class DecodableNnet;
 private:
  /// Sets up full_splicing_ and relative_splicing_, which
  /// are functions of raw_splicing_.
  void InitializeArrays();
  
  const Nnet &operator = (const Nnet &other);  // Disallow assignment.

  int32 speaker_info_dim_; /// Dimension of speaker information the
  /// networks accepts; will be zero for traditional networks.
  
  /// raw_splicing_ contains a sorted list of the frames that are
  /// spliced together for each layer-- the elements are relative
  /// frame indices.  Note: for the input layer, this is the splicing
  /// that's applied to the raw features, and
  std::vector<std::vector<int32> > raw_splicing_;

  /// full_splicing_ is a function of raw_splicing_; it's
  /// a list of the frame offsets that we'd need at the
  /// beginning of this component, in order to compute
  /// a single frame of output.
  std::vector<std::vector<int32> > full_splicing_;

  /// See comment for function RelativeSplicingForComponent().
  std::vector<std::vector<std::vector<int32> > >  relative_splicing_;
  
  std::vector<Component*> components_;
};




} // namespace kaldi

#endif


