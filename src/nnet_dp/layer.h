// nnet_dp/layer.h

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


// This class is for a special type of neural-network layer that we have
// at the very end, after the soft-max.  We constrain each column of the
// matrix to sum to one, which will ensure that the transformed probabilities
// sum to one.
// Note: when we put it together, we have a bunch of these LinearLayers
// in parallel, one for each of the SoftmaxLayers [which we have for the
// different categories of labels... note, categories of labels relate
// to the 2-level tree.].
class LinearLayer {
 public:
  int32 InputDim() const { return params_.NumCols(); }
  int32 OutputDim() const { return params_.NumRows(); }
  
  LinearLayer(int size, BaseFloat diagonal_element, BaseFloat learning_rate);
  
  void Write(std::ostream &out, bool binary) const;
  void Read(std::istream &in, bool binary);
  
  // each row of the args to this function is one frame.
  void Forward(const MatrixBase<BaseFloat> &input,
               MatrixBase<BaseFloat> *output) const;
  
  // each row of the args to this function is one frame.
  void Backward(const MatrixBase<BaseFloat> &input,
                const MatrixBase<BaseFloat> &output_deriv,
                MatrixBase<BaseFloat> *input_deriv, // derivative w.r.t. input.
                LinearLayer *layer_to_update) const;

  void SetZero(); // used if we're using this to store gradients on a held out
  // set.  zeroes params_ and sets is_gradient_ = true.
  BaseFloat GetLearningRate() const { return learning_rate_; }
  void SetLearningRate(BaseFloat learning_rate) { learning_rate_ = learning_rate; }
 private:
  void Update(const MatrixBase<BaseFloat> &input,
              const MatrixBase<BaseFloat> &output_deriv);
      
  BaseFloat learning_rate_;
  Matrix<BaseFloat> params_; // parameters, indexed [output-index][input-index].
  bool is_gradient_; // If true, this class is just a holder for the gradient,
  // e.g. on a held out set.  This affects how we do the update [since we
  // don't use simple gradient descent for this type of layer.]
};

class SoftmaxLayer {
 public:
  int32 InputDim() const { return params_.NumCols(); }
  int32 OutputDim() const { return params_.NumRows(); }
  
  SoftmaxLayer(int input_size, int output_size, BaseFloat learning_rate); // Note:
  // this layer is initialized to zero.
  
  void Write(std::ostream &out, bool binary) const;
  void Read(std::istream &in, bool binary);

  // each row of the args to this function is one frame.
  // Note: support frame splicing, so if input.NumCols() is < input_size,
  // splice input, and shift by one each time.
  void Forward(const MatrixBase<BaseFloat> &input,
               MatrixBase<BaseFloat> *output) const;
  
  void Backward(const MatrixBase<BaseFloat> &input, 
                const MatrixBase<BaseFloat> &output,
                const MatrixBase<BaseFloat> &output_deriv,
                MatrixBase<BaseFloat> *input_deriv, // derivative w.r.t. input
                SoftmaxLayer *layer_to_update) const;
  
  BaseFloat GetLearningRate() const { return learning_rate_; }
  void SetLearningRate(BaseFloat learning_rate) { learning_rate_ = learning_rate; }

 private:
  void Update(const MatrixBase<BaseFloat> &input,
              const MatrixBase<BaseFloat> &sum_deriv,
              const MatrixBase<BaseFloat> &output);
  
  void ApplySoftmax(MatrixBase<BaseFloat> *output) const;

  // Propagate the derivative back through the nonlinearity.
  void ComputeSumDeriv(const MatrixBase<BaseFloat> &output,
                       const MatrixBase<BaseFloat> &output_deriv,
                       MatrixBase<BaseFloat> *sum_deriv) const;

  // Propagate derivative back from sum at nodes, to input.
  void ComputeInputDeriv(const MatrixBase<BaseFloat> &sum_deriv,
                         MatrixBase<BaseFloat> *input_deriv) const;

  BaseFloat learning_rate_;
  Matrix<BaseFloat> params_; // parameters, indexed [output-index][input-index].
  
  // A quasi-occupancy count, accumulated from the data and used for splitting.
  Vector<BaseFloat> occupancy_;
};
  

class TanhLayer {
  // Note: the tanh function is the same as the sigmoid, except modified to go
  // from -1 to +1 (instead of 0 to 1), and with the input (x) multiplied by
  // two-- so it's stretched vertically and squashed horizontally, relative to
  // the sigmoid.
 public:
  // We initialize the weights to be uniformly distributed on
  // [-1/sqrt(n), +1/sqrt(n)], where n is the input dimension.
  // Apparently this is widely used: see  glorot10a.pdf (search term), 
  // Glorot and Bengio, "Understanding the difficulty of training deep feedforward networks".
  int32 InputDim() const { return params_.NumCols(); }
  int32 OutputDim() const { return params_.NumRows(); }

  TanhLayer(int input_size,
            int output_size,
            BaseFloat learning_rate);
  
  void Write(std::ostream &out, bool binary) const;
  void Read(std::istream &in, bool binary);
  
  // The forward pass.  Note: in this function we support doing the operation
  // on multiple frames at a time.  If the dimension of each row of the input
  // is not the same as the input dimension of the layer [but divides it]
  // then the first row of the "real" input will be spliced first n rows
  // of the input, then shift by one each time.
  void Forward(const MatrixBase<BaseFloat> &input,
               MatrixBase<BaseFloat> *output) const;
  
  // The backward pass.  Similar note about sizes and frame-splicing
  // applies as in "Forward".
  void Backward(const MatrixBase<BaseFloat> &input,
                const MatrixBase<BaseFloat> &output,
                const MatrixBase<BaseFloat> &output_deriv,                
                MatrixBase<BaseFloat> *input_deriv, // derivative w.r.t. input.
                TanhLayer *layer_to_update) const;
  
  BaseFloat GetLearningRate() const { return learning_rate_; }
  void SetLearningRate(BaseFloat learning_rate) { learning_rate_ = learning_rate; }

 private:
  void Update(const MatrixBase<BaseFloat> &input,
              const MatrixBase<BaseFloat> &sum_deriv);

  // Propagate the derivative back through the nonlinearity.
  void ComputeSumDeriv(const MatrixBase<BaseFloat> &output,
                       const MatrixBase<BaseFloat> &output_deriv,
                       MatrixBase<BaseFloat> *sum_deriv) const;
      
  // Called from Backward().
  void ComputeInputDeriv(const MatrixBase<BaseFloat> &sum_deriv,
                         MatrixBase<BaseFloat> *input_deriv) const;

  
  void ApplyTanh(MatrixBase<BaseFloat> *output) const;

  BaseFloat learning_rate_;
  Matrix<BaseFloat> params_; // parameters, indexed [output-index][input-index].
};
  

} // namespace

#endif

