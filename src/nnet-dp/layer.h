// nnet-dp/layer.h

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

#ifndef KALDI_NNET_DP_LAYER_H_
#define KALDI_NNET_DP_LAYER_H_

#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"
#include "thread/kaldi-mutex.h"

namespace kaldi {


// This is used as a base-class for the individual layers;
// they override the functions as needed.
class GenericLayer {
 public:
  int32 InputDim() const { return params_.NumCols(); }
  int32 OutputDim() const { return params_.NumRows(); }

  virtual void Write(std::ostream &out, bool binary) const = 0;
  virtual void Read(std::istream &in, bool binary) = 0; // This Read function will only work
  // if the actual type of this object is the same as what's on disk.

  static GenericLayer *ReadNew(std::istream &in, bool binary); // Returns a pointer
  // to a newly created Layer object of the correct type.
  
  virtual void ApplyNonlinearity(MatrixBase<BaseFloat> *output) const;

  virtual void ApplyNonlinearity(VectorBase<BaseFloat> *output) const = 0;

  virtual void CombineWithWeight(const GenericLayer &other,
                                 BaseFloat other_weight);
  
  // each row of the args to this function is one frame.
  virtual void Forward(const MatrixBase<BaseFloat> &input,
                       MatrixBase<BaseFloat> *output) const;

  // forward function that works on one frame.
  virtual void Forward(const VectorBase<BaseFloat> &input,
                       VectorBase<BaseFloat> *output) const;
  
  // each row of the args to this function is one frame.
  virtual void Backward(const MatrixBase<BaseFloat> &input,
                        const MatrixBase<BaseFloat> &output,
                        const MatrixBase<BaseFloat> &output_deriv,
                        MatrixBase<BaseFloat> *input_deriv,
                        GenericLayer *layer_to_update) const;
  
  BaseFloat GetLearningRate() const { return learning_rate_; }
  void SetLearningRate(BaseFloat lrate) { learning_rate_ = lrate; }

  BaseFloat GetShrinkageRate() const { return shrinkage_rate_; }
  void SetShrinkageRate(BaseFloat srate) { shrinkage_rate_ = srate; }
  
  // Use default copy constructor and assignment operator.
  
  void SetZero() { params_.Set(0.0); }

  GenericLayer(): learning_rate_(0.0) { } // avoid undefined
  // behavior.  This value should always be overwritten though.
  const Matrix<BaseFloat> &Params() const { return params_; }

  void SetParams(const MatrixBase<BaseFloat> &params);

  // some human-readable information about the object.
  std::string Info() const;

 protected:

  // Propagate the derivative back through the nonlinearity.
  virtual void ComputeSumDeriv(const MatrixBase<BaseFloat> &output,
                               const MatrixBase<BaseFloat> &output_deriv,
                               MatrixBase<BaseFloat> *sum_deriv) const = 0;
  
  // Default implementation is supplied, but may be overridden (e.g. for
  // LinearLayer, which does the update differently for efficiency).
  virtual void Update(const MatrixBase<BaseFloat> &input,
                      const MatrixBase<BaseFloat> &output_deriv,
                      const MatrixBase<BaseFloat> &output);

  BaseFloat learning_rate_;
  BaseFloat shrinkage_rate_;
  Matrix<BaseFloat> params_; // parameters, indexed [output-index][input-index].
};


// This class is for a special type of neural-network layer that we have
// at the very end, after the soft-max.  We constrain each column of the
// matrix to sum to one, which will ensure that the transformed probabilities
// sum to one.
// Note: when we put it together, we have a bunch of these LinearLayers
// in parallel, one for each of the SoftmaxLayers [which we have for the
// different categories of labels... note, categories of labels relate
// to the 2-level tree.].
class LinearLayer: public GenericLayer {
 public:
  LinearLayer() { } // called prior to Read().
  LinearLayer(int32 size, BaseFloat diagonal_element,
              BaseFloat learning_rate, BaseFloat shrinkage_rate);
  
  virtual void Write(std::ostream &out, bool binary) const;
  virtual void Read(std::istream &in, bool binary);
  
  void SetZero() { params_.Set(0.0); }
  void MakeGradient() { is_gradient_ = true; }

  void CombineWithWeight(const LinearLayer &other,
                         BaseFloat other_weight);
  
  // Override the base-class Backward for efficiency.
  void Backward(const MatrixBase<BaseFloat> &input,
                const MatrixBase<BaseFloat> &output,
                const MatrixBase<BaseFloat> &output_deriv,
                MatrixBase<BaseFloat> *input_deriv, // derivative w.r.t. input.
                GenericLayer *layer_to_update) const;
  
  void ApplyNonlinearity(MatrixBase<BaseFloat> *output) const;

  void ApplyNonlinearity(VectorBase<BaseFloat> *output) const;
 private:
  // override the base-class Update; we do it differently.
  virtual void Update(const MatrixBase<BaseFloat> &input,
                      const MatrixBase<BaseFloat> &output_deriv,
                      const MatrixBase<BaseFloat> &output);

  // Note: this function shouldn't ever be called; we define it to throw an
  // error, to avoid having undefined pure-virtual functions.
  void ComputeSumDeriv(const MatrixBase<BaseFloat> &output,
                       const MatrixBase<BaseFloat> &output_deriv,
                       MatrixBase<BaseFloat> *sum_deriv) const;
  
  bool is_gradient_; // If true, this class is just a holder for the gradient,
  // e.g. on a held out set.  This affects how we do the update [since we
  // don't use simple gradient descent for this type of layer.]
};

class SparseLayer: public GenericLayer {
  int32 InputDim() const { return input_dim_; }
  int32 OutputDim() const { return sparse_params_.size(); }
  
  // This layer contains a sparse random projection, typically from a higher to a lower
  // dimension; currently this layer is not trained but is randomly initialized
  // and fixed.  It helps us to decrease the dimension of the features, so we
  // can have a lower number of parameters (for the trainable part) and a
  // higher number of nonlinearities.
  SparseLayer(int32 input_dim, int32 output_dim);

  virtual void Write(std::ostream &out, bool binary) const; // 
  virtual void Read(std::istream &in, bool binary);
  
  // each row of the args to this function is one frame.
  virtual void Forward(const MatrixBase<BaseFloat> &input,
                       MatrixBase<BaseFloat> *output) const;

  // forward function that works on one frame.
  virtual void Forward(const VectorBase<BaseFloat> &input,
                       VectorBase<BaseFloat> *output) const;
  
  // each row of the args to this function is one frame.
  virtual void Backward(const MatrixBase<BaseFloat> &input,
                        const MatrixBase<BaseFloat> &output,
                        const MatrixBase<BaseFloat> &output_deriv,
                        MatrixBase<BaseFloat> *input_deriv,
                        GenericLayer *layer_to_update) const;
 private:
  // note: don't use params_ variable.
  int32 input_dim_;
  std::vector<std::vector<std::pair<int32, BaseFloat> > > sparse_params_;
};



class SoftmaxLayer: public GenericLayer {
 public:
  SoftmaxLayer() { } // called prior to Read().
  SoftmaxLayer(int32 input_size, int32 output_size,
               BaseFloat learning_rate, BaseFloat shrinkage_rate); // Note:
  // this layer is initialized to zero.

  virtual void Write(std::ostream &out, bool binary) const;
  virtual void Read(std::istream &in, bool binary);

  void SetZero() { params_.Set(0.0); occupancy_.Set(0.0); }

  void SetOccupancy(const VectorBase<BaseFloat> &occupancy);

  void ZeroOccupancy() { occupancy_.Set(0.0); } // Set occupancy to zero so we
  // can start counting afresh...
  BaseFloat TotOccupancy() { return occupancy_.Sum(); }
  const Vector<BaseFloat> &Occupancy() const { return occupancy_; }

  void CombineWithWeight(const SoftmaxLayer &other,
                         BaseFloat other_weight);
 private:
  // override the base-class Update; we do it differently.
  virtual void Update(const MatrixBase<BaseFloat> &input,
                      const MatrixBase<BaseFloat> &sum_deriv,
                      const MatrixBase<BaseFloat> &output);
  
  void ApplyNonlinearity(MatrixBase<BaseFloat> *output) const;

  void ApplyNonlinearity(VectorBase<BaseFloat> *output) const;  

  // Propagate the derivative back through the nonlinearity.
  void ComputeSumDeriv(const MatrixBase<BaseFloat> &output,
                       const MatrixBase<BaseFloat> &output_deriv,
                       MatrixBase<BaseFloat> *sum_deriv) const;

  // A quasi-occupancy count, accumulated from the data and used for splitting.
  Vector<BaseFloat> occupancy_;
};
  

class TanhLayer: public GenericLayer {
  // Note: the tanh function is the same as the sigmoid, except modified to go
  // from -1 to +1 (instead of 0 to 1), and with the input (x) multiplied by
  // two-- so it's stretched vertically and squashed horizontally, relative to
  // the sigmoid.
 public:
  TanhLayer() { } // called prior to Read().
  TanhLayer(int32 input_size,
            int32 output_size,
            BaseFloat learning_rate,
            BaseFloat shrinkage_rate,
            BaseFloat parameter_stddev);
  
  virtual void Write(std::ostream &out, bool binary) const;
  virtual void Read(std::istream &in, bool binary);
  
  
  // Use default copy constructor and assignment operator.
 private:
  // Propagate the derivative back through the nonlinearity.
  void ComputeSumDeriv(const MatrixBase<BaseFloat> &output,
                       const MatrixBase<BaseFloat> &output_deriv,
                       MatrixBase<BaseFloat> *sum_deriv) const;
      
  void ApplyNonlinearity(MatrixBase<BaseFloat> *output) const;

  void ApplyNonlinearity(VectorBase<BaseFloat> *output) const;
};


/* MixUpFinalLayers() increases the output dimension of the
   SoftmaxLayer and the corresponding input dimension of the
   LinearLayer that comes after it, in a process analogous
   to mixing-up of Gaussians.  Similar to the approach when
   we mix up Gaussians, we always choose the Gaussian/node
   with the highest occupancy to split.
*/
void MixUpFinalLayers(int32 new_num_neurons,
                      BaseFloat perturb_stddev,
                      SoftmaxLayer *softmax_layer,
                      LinearLayer *linear_layer);
                      
  

} // namespace

#endif

