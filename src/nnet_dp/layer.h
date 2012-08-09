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
#include "thread/kaldi-mutex.h"

namespace kaldi {

// This is a base-class for statistics for neural-network
// training.  This is for multi-threaded training.  The general idea is
// as follows.  Each update in SGD is of the form
// [parameter-matrix] += \alpha v w^T
// where \alpha is the current learning rate.  If we were to do that
// each time we saw a sample, it would involve a lot of writes to memory
// and a lot of work for the system bus.  So we batch them up so that
// we can do something like:
// [parameter-matrix] += \alpha W V^T
// where V and W are wide, short matrices, each row of which comes
// from a separate training example.  Note: the rows of v would
// be derivatives w.r.t. the input to those neurons, and the rows of
// w would be the outputs of the previous layer.  We use rows not
// columns because BLAS stores matrices with row-major indexing, so this
// gives us more memory locality.
// Every so often we'll call some kind of BLAS matrix-multiply routine
// to make the change.
// [note: it's slightly more complicated than this because we use
//  somewhat neuron-specific learning rates, using an idea based on
//  preconditioning with the Fisher matrix.]
//
// We should first mention something else: since each thread processes
// a small window of frames, of a fixed size (e.g. 5 to 10) each time,
// it's easier to not think of rows, but of chunks of rows.  So let
// this "chunk" be the small fixed quantity, e.g. 5 or 10.
//
// Anyway, in a multi-threaded situation, the main issue is allocating
// which chunks of rows of the matrix different threads should write to.
// [once a thread knows which rows to write to, it doesn't need the
//  lock any more.]
// 
// A thread will acquire a lock in order to get access to a chunk of
// rows.  The chunks will have 4 different states:
// "empty", "full", "filling" and "emptying".


class ChunkManager {
  enum {
    kEmpty,
    kFull,
    kFilling,
    kEmptying
  };
  
  ChunkManager(int32 num_chunks, // note: num_chunks must be >= num_threads.
               int32 num_threads); // initializer.  Sets
  // all chunks from 0 ... num_chunks-1 to status kEmpty.
  // num_chunks_update (<=num_chunks) is how full
  
  void SetToFull(int32 chunk); // Sets chunk [which should be in status kFilling]
  // to kFull.
  
  void SetToEmpty(int32 begin, int32 end); // Sets chunks in range begin ... end-1
  // [which should be in status kEmptying] to status kEmpty.

  
  // GetTask() will either:
  // (a) find a chunk in state kEmpty, change it to
  // kFilling and return it in *chunk1, setting *chunk2 to -1
  // or 
  // (b) Find the largest contiguous range of chunks in state kFull, set
  // them to state kEmptying, and return this range as *chunk1, *chunk2 [so the
  // range is from *chunk1 to (*chunk2)-1].
  //
  // It's up to the discretion of GetTask() which of these to do. 
  void GetTask(int32 *chunk1, int32 *chunk2);
  
 private:
  Mutex mutex_; // generic lock used while changing chunk_status_.
  Mutex get_task_mutex_; // lock held while calling GetTask(), or when
  // no task is available because all statuses are kFilling or kEmptying.
  
  std::vector<int32> chunk_status_;
};

struct GenericLayerUpdateConfig {
  // A configuration class for the update of the layers-- this one
  // is generic to all the layer types; it can be used as a base class
  // if needed.  Note: although the class itself is generic, some
  // members (e.g. chunk_size) will typically be different for
  // different layers.
  double learning_rate;
  int num_chunks; // Number of chunks that the stats class should store.
  int chunk_size; // This is layer-specific-- the number of frames in each chunk of
  // features.  [typically in the range 5 to 10.]
};


class LinearLayerStats;

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
  LinearLayer(int size, BaseFloat diagonal_element);
  
  void Write(std::ostream &out, bool binary) const;
  void Read(std::istream &in, bool binary) const;
  
  // each row of the args to this function is one frame.
  void Forward(const MatrixBase<BaseFloat> &input,
               MatrixBase<BaseFloat> *output);

  // each row of the args to this function is one frame.
  void Backward(const MatrixBase<BaseFloat> &input, 
                const MatrixBase<BaseFloat> &output,
                const MatrixBase<BaseFloat> &output_deriv,
                MatrixBase<BaseFloat> *input_deriv, // derivative w.r.t. input.
                LinearLayerStats *stats);
  
  friend class LinearLayerStats;
 private:
  Matrix<BaseFloat> params_; // parameters, indexed [output-index][input-index].
};

// Note: this class is not just a passive container of stats; it
// also updates the model when it gets full [or when it's deinitialized.]
// So we need to initialize it with the learning rate.
class LinearLayerStats {
 public:
  LinearLayerStats (GenericLayerUpdateConfig &config,
                    LinearLayer *layer_to_update); // this is the layer we'll update;
  // also gives size info to initialize the stats.  Does not have to be the same
  // as the layer we got the stats from [this is useful in getting derivative w.r.t.
  // parameters, on held-out set.]
  
  // Accumulate stats [may also update the model.]
  void AccStats(const MatrixBase<BaseFloat> &input,
                const MatrixBase<BaseFloat> &output_deriv);
                
 private:

  void Update(int32 chunk_start, int32 chunk_end); // This is
  // called from AccStats.

  LinearLayer *layer_to_update_;
  GenericLayerUpdateConfig config_;
  ChunkManager chunk_manager_;
  Matrix<BaseFloat> stats_;
  SubMatrix<BaseFloat> input_stats_; // sub-matrix of stats_; each row is input vector.
  SubMatrix<BaseFloat> output_stats_; // sub-matrix of stats_; each row is deriv at output.
};

class SoftmaxLayerStats;

class SoftmaxLayer {
 public:
  SoftmaxLayer(int input_size, int output_size); // Note:
  // this layer is initialized to zero.
  
  void Write(std::ostream &out, bool binary) const;
  void Read(std::istream &in, bool binary) const;

  // each row of the args to this function is one frame.
  // Note: support frame splicing, so if input.NumCols() is < input_size,
  // splice input, and shift by one each time.
  void Forward(const MatrixBase<BaseFloat> &input,
               MatrixBase<BaseFloat> *output);
  
  void Backward(const MatrixBase<BaseFloat> &input, 
                const MatrixBase<BaseFloat> &output,
                const MatrixBase<BaseFloat> &output_deriv,
                MatrixBase<BaseFloat> *input_deriv, // derivative w.r.t. input
                SoftmaxLayerStats *stats);
  
  friend class SoftmaxLayerStats;

 private:
  void ApplySoftmax(MatrixBase<BaseFloat> *output);

  // Propagate the derivative back through the nonlinearity.
  void ComputeSumDeriv(const MatrixBase<BaseFloat> &output,
                       const MatrixBase<BaseFloat> &output_deriv,
                       MatrixBase<BaseFloat> *sum_deriv);
  
  Matrix<BaseFloat> params_; // parameters, indexed [output-index][input-index].

  // A quasi-occupancy count, accumulated from the data and used for splitting.
  Vector<BaseFloat> occupancy_;
};
  
class SoftmaxLayerStats {
 public:
  SoftmaxLayerStats (GenericLayerUpdateConfig &config,
                     SoftmaxLayer *layer_to_update); // this is the layer we'll update;
  // also gives size info to initialize the stats.  Does not have to be the same
  // as the layer we got the stats from [this is useful in getting derivative w.r.t.
  // parameters, on held-out set.]
  
  // Accumulate stats [may also update the model.]
  // Note: this type of layer supports frame splicing, so the "input" vector
  // may have to be spliced together.
  // Note: the "output" is only needed for storing occupancies, which we'll use
  // for splitting output indices (like splitting Gaussians).
  void AccStats(const MatrixBase<BaseFloat> &input,
                const MatrixBase<BaseFloat> &output, // special to this layer type: for getting occupancies.
                const MatrixBase<BaseFloat> &sum_deriv); // sum_deriv is deriv w.r.t. sum before sigmoid.
  
 private:
  void Update(int32 chunk_start, int32 chunk_end); // This is
  // called from AccStats.

  SoftmaxLayer *layer_to_update_;
  GenericLayerUpdateConfig config_;
  ChunkManager chunk_manager_;
  Matrix<BaseFloat> stats_;
  SubMatrix<BaseFloat> input_stats_; // sub-matrix of stats_; each row is input vector.
  SubMatrix<BaseFloat> output_stats_; // sub-matrix of stats_; each row is objf derivative w.r.t output.
  Matrix<BaseFloat> output_sums_;  // sums of output, one row per chunk.
};


class TanhLayer {
  // This sigmoid is a symmetric sigmoid that goes from -1 to +1, i.e. the tanh function.
 public:
  // We initialize the weights to be uniformly distributed on
  // [-1/sqrt(n), +1/sqrt(n)], where n is the input dimension.
  // Apparently this is widely used: see  glorot10a.pdf (search term), 
  // Glorot and Bengio, "Understanding the difficulty of training deep feedforward networks".
  TanhLayer(int input_size,
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
  void Backward(const MatrixBase<BaseFloat> &input,
                const MatrixBase<BaseFloat> &output,
                const MatrixBase<BaseFloat> &output_deriv,                
                MatrixBase<BaseFloat> *input_deriv, // derivative w.r.t. input.
                TanhLayerStats *stats);
  
  void ClearOccupancy() { occupancy_.SetZero(); }
  friend class TanhLayerStats;
  
 private:
  // Propagate the derivative back through the nonlinearity.
  void ComputeSumDeriv(const MatrixBase<BaseFloat> &output,
                       const MatrixBase<BaseFloat> &output_deriv,
                       MatrixBase<BaseFloat> *sum_deriv);
      
  // Called from Backward().
  void ComputeInputDeriv(const MatrixBase<BaseFloat> &output,
                         const MatrixBase<BaseFloat> &sum_deriv,
                         MatrixBase<BaseFloat> *input_deriv);

  
  void ApplyTanh(MatrixBase<BaseFloat> *output);
  
  Matrix<BaseFloat> params_; // parameters, indexed [output-index][input-index].
  Vector<BaseFloat> occupancy_; // Accumulate the occupation count during training.
};
  
class TanhLayerStats {
 public:
  TanhLayerStats (GenericLayerUpdateConfig &config,
                     TanhLayer *layer_to_update); // this is the layer we'll update;
  // also gives size info to initialize the stats.  Does not have to be the same
  // as the layer we got the stats from [this is useful in getting derivative w.r.t.
  // parameters, on held-out set.]
  
  // Accumulate stats [may also update the model.]
  // Note: this type of layer supports frame splicing, so the "input" vector
  // may have to be spliced together by this function.
  void AccStats(const MatrixBase<BaseFloat> &input,
                const MatrixBase<BaseFloat> &sum_deriv); // sum_deriv is deriv w.r.t. sum before sigmoid.
  
 private:
  void Update(int32 chunk_start, int32 chunk_end); // This is
  // called from AccStats.

  TanhLayer *layer_to_update_;
  GenericLayerUpdateConfig config_;
  ChunkManager chunk_manager_;
  Matrix<BaseFloat> stats_;
  SubMatrix<BaseFloat> input_stats_; // sub-matrix of stats_; each row is input vector.
  SubMatrix<BaseFloat> output_stats_; // sub-matrix of stats_; each row is objf derivative w.r.t output.
};



} // namespace

#endif

