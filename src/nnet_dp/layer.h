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

#ifndef KALDI_NNET_DP_NNET1_H_
#define KALDI_NNET_DP_NNET1_H_

#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"

namespace kaldi {


// This class is for a special type of neural-network layer that we have
// at the very end, after the soft-max.  We constrain each column of the
// matrix to sum to one, which will ensure that the transformed probabilities
// sum to one.
// Note: when we put it together, we have a bunch of these LinearLayers
// in parallel, one for each of the SoftmaxLayers [which we have for the
// different classes].
class LinearLayer {
 public:
  LinearLayer(int size, BaseFloat diagonal_element);

  void Write(std::ostream &out, bool binary) const;
  void Read(std::istream &in, bool binary) const;

  // each row of the args to this function is one frame.
  void Forward(const MatrixBase<BaseFloat> &input,
               MatrixBase<BaseFloat> *output);

  // each row of the args to this function is one frame.  
  void Backward(const MatrixBase<BaseFloat> &output_deriv,
                MatrixBase<BaseFloat> *input_deriv, // derivative w.r.t. input.
                LinearLayerStats *stats);
  
  friend class LinearLayerStats;
 private:
  Matrix<BaseFloat> params_; // parameters, indexed [output-index][input-index].
  
  // This is in effect the diagonal of the Fisher matrix, computed using
  // a weighted average over time..
  Matrix<BaseFloat> average_derivative_variance_;
};
  
class LinearLayerStats {
 public:
  void SawFrame() { num_frames_++; } // must be incremented each time
  // we saw a frame-- this is necessary for getting average_derivative_variance_
  // in LinearLayer correctly updated.
  LinearLayerStats (const LinearLayer &layer);

  void Update(LinearLayer *layer,
              double learning_rate,
              bool use_fisher); // Updates the layer; also clears the stats.

 private:  
  Matrix<BaseFloat> deriv_;
  Matrix<BaseFloat> deriv_variance_; // accumulates the variance of the derivative
  // [just summed up.]
  int num_frames_;
};


class SoftmaxLayer {
 public:
  SoftmaxLayer(int input_size, int output_size); // Note:
  // this layer is initialized to zero.
  
  void Write(std::ostream &out, bool binary) const;
  void Read(std::istream &in, bool binary) const;

  // each row of the args to this function is one frame.
  void Forward(const MatrixBase<BaseFloat> &input,
               MatrixBase<BaseFloat> *output);

  // each row of the args to this function is one frame.
  void Backward(const MatrixBase<BaseFloat> &output_deriv,
                MatrixBase<BaseFloat> *input_deriv, // derivative w.r.t. input
                SoftmaxLayerStats *stats);
  
  friend class SoftmaxLayerStats;

 private:
  Matrix<BaseFloat> params_; // parameters, indexed [output-index][input-index].
  
  // This is in effect the diagonal of the Fisher matrix, computed using a
  // weighted average over time... but we store the average of this over all
  // input-indexes, for each output-index [the assumption is that the input
  // variables are all distributed in about the same way.]
  Vector<BaseFloat> average_derivative_variance_;
};
  
class SoftmaxLayerStats {
 public:
  void SawFrame() { num_frames_++; } // must be incremented each time
  // we saw a frame-- this is necessary for getting average_derivative_variance_
  // in LinearLayer correctly updated.
  LinearLayerStats (const LinearLayer &layer);

  void Update(LinearLayer *layer,
              double learning_rate,
              bool use_fisher); // Updates the layer; also clears the stats.

 private:  
  Matrix<BaseFloat> deriv_;
  Matrix<BaseFloat> deriv_variance_; // accumulates the variance of the derivative
  // [just summed up.]
  int num_frames_;
};


class SigmoidLayer { // "symmetric sigmoid" that goes from -1 to +1.
 public:
  // We initialize the weights to be uniformly distributed on
  // [-1/sqrt(n), +1/sqrt(n)], where n is the input dimension.
  // Apparently this is widely used: see  glorot10a.pdf (search term), 
  // Glorot and Bengio, "Understanding the difficulty of training deep feedforward networks".
  SigmoidLayer(int input_size,
               int output_size);
  
  void Write(std::ostream &out, bool binary) const;
  void Read(std::istream &in, bool binary) const;
  
  // The forward pass.  Note: in this function we support doing the operation
  // on multiple frames at a time.  If the dimension of each row of the input
  // is not the same as the input dimension of the layer [but divides it]
  // then the first row of the "real" input will be spliced first n rows
  // of the input, then shift by one each time.
  void Forward(const MatrixBase<BaseFloat> &input,
               MatrixBase<BaseFloat> *output);

  // The backward pass.  Similar note about sizes and frame-splicing
  // applies as in "Forward".
  void Backward(const MatrixBase<BaseFloat> &output_deriv,
                MatrixBase<BaseFloat> *input_deriv, // derivative w.r.t. input
                int32 input_stride,
                SigmoidLayerStats *stats);
  
  friend class SigmoidLayerStats;
  
 private:
  Matrix<BaseFloat> params_; // parameters, indexed [output-index][input-index].
  
  // This is in effect the diagonal of the Fisher matrix, computed using a
  // weighted average over time... but we store the average of this over all
  // input-indexes, for each output-index.  The assumption is that the input
  // variables are all distributed in about the same way.
  Vector<BaseFloat> average_derivative_variance_;
};
  
class SoftmaxLayerStats {
 public:
  void SawFrame() { num_frames_++; } // must be incremented each time
  // we saw a frame-- this is necessary for getting average_derivative_variance_
  // in LinearLayer correctly updated.
  LinearLayerStats (const LinearLayer &layer);

  void Update(LinearLayer *layer,
              double learning_rate,
              bool use_fisher); // Updates the layer; also clears the stats.

 private:  
  Matrix<BaseFloat> deriv_;
  Matrix<BaseFloat> deriv_variance_; // accumulates the variance of the derivative
  // [just summed up.]
  int num_frames_;
};




} // namespace

#endif

