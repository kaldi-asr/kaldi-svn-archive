// nnet/nnet-nnet.h

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



#ifndef KALDI_NNET_NNET_H_
#define KALDI_NNET_NNET_H_

#include "base/kaldi-common.h"
#include "util/kaldi-io.h"
#include "matrix/matrix-lib.h"
#include "nnet-cpu/nnet-component.h"

#include <iostream>
#include <sstream>
#include <vector>


namespace kaldi {



class Nnet1 {
 public:
  // Returns number of components-- think of this as similar to # of layers, but
  // e.g. the nonlinearity and the linear part count as separate components.
  int32 NumComponents() { return components_.size(); }
  
  int32 LeftContext() const; // Returns #frames of left context needed [just the sum
  // of left_context for each component.]

  int32 RightContext() const; // Returns #frames of right context needed [just the sum
  // of right_context for each component.]

  int32 LeftContextForComponent(int32 component) const;
  int32 RightContextForComponent(int32 component) const;

  const Component &GetComponent(int32 component) const;

  Component &GetComponent(int32 component);
  
  int32 OutputDim() const {
    return (components_.empty() ? 0 : components_.back()->OutputDim());
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

  
  // This sets *info to the dot prod of *this . validation_gradient,
  // separately for each component.
  // This is used in updating learning rates and shrinkage rates.
  void ComputeDotProduct(
      const Nnet1 &validation_gradient,
      VectorBase<BaseFloat> *dot_prod) const;
  
  friend class NnetUpdater;
  friend class DecodableNnet;
 private:
  const Nnet &operator = (const Nnet &other);  // Disallow assignment.
  
  // the left and right context (splicing) for layers.
  std::vector<int32> left_context_;
  std::vector<int32> right_context_;
  std::vector<Component*> components_;
};



class Nnet {
 public:
  Nnet() { } // Called prior to Read() or Init()
  void Init(std::istream &config_file); // initialize given text configuration file,
  // one line for each layer.
  void Read(std::istream &is, bool binary);
  void Write(std::ostream &os, bool binary) const;
  
  ~Nnet();

  
 public:
  /// Perform forward pass through the network
  void Propagate(const Matrix<BaseFloat> &in, Matrix<BaseFloat> *out); 
  /// Perform backward pass through the network
  void Backpropagate(const Matrix<BaseFloat> &in_err, Matrix<BaseFloat> *out_err);
  /// Perform forward pass through the network, don't keep buffers (use it when not training)
  void Feedforward(const Matrix<BaseFloat> &in, Matrix<BaseFloat> *out); 

  MatrixIndexT InputDim() const; ///< Dimensionality of the input features
  MatrixIndexT OutputDim() const; ///< Dimensionality of the desired vectors

  MatrixIndexT LayerCount() const { ///< Get number of layers
    return nnet_.size(); 
  }
  Component* Layer(MatrixIndexT index) { ///< Access to individual layer
    return nnet_[index]; 
  }
  int IndexOfLayer(const Component& comp) const; ///< Get the position of layer in network

  /// Access to forward pass buffers
  const std::vector<Matrix<BaseFloat> >& PropagateBuffer() const { 
    return propagate_buf_; 
  }

  /// Access to backward pass buffers
  const std::vector<Matrix<BaseFloat> >& BackpropagateBuffer() const { 
    return backpropagate_buf_; 
  }
  
  /// Read the MLP from file (can add layers to exisiting instance of Nnet)
  void Read(const std::string &file);  
  /// Read the MLP from stream (can add layers to exisiting instance of Nnet)
  void Read(std::istream &in, bool binary);  
  /// Write MLP to file
  void Write(const std::string &file, bool binary); 
  /// Write MLP to stream 
  void Write(std::ostream &out, bool binary);    
  
  /// Set the learning rate values to trainable layers, 
  /// factors can disable training of individual layers
  void LearnRate(BaseFloat lrate, const char *lrate_factors); 
  /// Get the global learning rate value
  BaseFloat LearnRate() { 
    return learn_rate_; 
  }
  /// Get the string with real learning rate values
  std::string LearnRateString();  

  void Momentum(BaseFloat mmt);
  void L2Penalty(BaseFloat l2);
  void L1Penalty(BaseFloat l1);

 private:
  /// Creates a component by reading from stream, return NULL if no more components
  static Component* ComponentFactory(std::istream &in, bool binary, Nnet *nnet);
  /// Dumps individual component to stream
  static void ComponentDumper(std::ostream &out, bool binary, const Component &comp);

  typedef std::vector<Component*> NnetType;
  
  NnetType nnet_;     ///< vector of all Component*, represents layers

  std::vector<Matrix<BaseFloat> > propagate_buf_; ///< buffers for forward pass
  std::vector<Matrix<BaseFloat> > backpropagate_buf_; ///< buffers for backward pass

  BaseFloat learn_rate_; ///< global learning rate

  KALDI_DISALLOW_COPY_AND_ASSIGN(Nnet);
};
  

inline Nnet::~Nnet() {
  // delete all the components
  NnetType::iterator it;
  for(it=nnet_.begin(); it!=nnet_.end(); ++it) {
    delete *it;
  }
}

   
inline MatrixIndexT Nnet::InputDim() const { 
  if (LayerCount() > 0) {
   return nnet_.front()->InputDim(); 
  } else {
   KALDI_ERR << "No layers in MLP"; 
  }
}


inline MatrixIndexT Nnet::OutputDim() const { 
  if (LayerCount() > 0) {
    return nnet_.back()->OutputDim(); 
  } else {
    KALDI_ERR << "No layers in MLP"; 
  }
}


inline int32 Nnet::IndexOfLayer(const Component &comp) const {
  for(int32 i=0; i<LayerCount(); i++) {
    if (&comp == nnet_[i]) return i;
  }
  KALDI_ERR << "Component:" << &comp 
            << " type:" << comp.GetType() 
            << " not found in the MLP";
  return -1;
}
 
  
inline void Nnet::Read(const std::string &file) {
  bool binary;
  Input in(file, &binary);
  Read(in.Stream(), binary);
  in.Close();
}


inline void Nnet::Write(const std::string &file, bool binary) {
  Output out(file, binary, true);
  Write(out.Stream(), binary);
  out.Close();
}


inline void Nnet::Write(std::ostream &out, bool binary) {
  for(int32 i=0; i<LayerCount(); i++) {
    nnet_[i]->Write(out, binary);
  }
}

    
inline void Nnet::Momentum(BaseFloat mmt) {
  for(int32 i=0; i<LayerCount(); i++) {
    if (nnet_[i]->IsUpdatable()) {
      dynamic_cast<UpdatableComponent*>(nnet_[i])->Momentum(mmt);
    }
  }
}


inline void Nnet::L2Penalty(BaseFloat l2) {
  for(int32 i=0; i<LayerCount(); i++) {
    if (nnet_[i]->IsUpdatable()) {
      dynamic_cast<UpdatableComponent*>(nnet_[i])->L2Penalty(l2);
    }
  }
}


inline void Nnet::L1Penalty(BaseFloat l1) {
  for(int32 i=0; i<LayerCount(); i++) {
    if (nnet_[i]->IsUpdatable()) {
      dynamic_cast<UpdatableComponent*>(nnet_[i])->L1Penalty(l1);
    }
  }
}




} // namespace kaldi

#endif


