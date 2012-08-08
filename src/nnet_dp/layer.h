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
#include "thread/mutex.h"

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
// rows.  We'll store the state of the rows in 3 different states:
// "free", "full", and "working".  In the "free" case, a row is empty;
// in the "full" case it has data in it; in the "working" case, it means
// some process is doing something with it [e.g. writing data to it,
// or committing data to the parameter matrix.]
//



  

class ChunkManager {
  enum {
    kEmpty;
    kFull;
    kFilling;
    kEmptying;
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
  // It's up to the discretion of GetTask() which of these to do.  In order to
  // avoid nasty synchronization issues that are beyond my (DP)'s parallel
  // programming skills (and also to ensure that processes never block for very
  // long), we have the following policy: only one range at a time may be in the
  // kEmptying state, and the range must be no larger than (num_chunks -
  // num_threads + 1).  This ensures that even if all the other threads start
  // "filling" a chunk while that one thread is emptying the range, they won't
  // need to block.
  // This also means that we don't have to have a separate mutex to stop
  // more than one thread at a time from attempting to update that part of
  // the model.
  void GetTask(int32 *chunk1, int32 *chunk2);
  
  // This function returns the largest contiguous sequence of stats that are in
  // state kFull.  Sets them to status kEmptying.  Blocks if the result would
  // be
  std::pair<int32,int32> GetLargestFullRange();
  
 private:
  Mutex mutex_; // generic lock used while changing chunk_status_.
  Mutex get_task_mutex_; // lock held while calling GetTask(), or when
  // no task is available because all statuses are kFilling or kEmptying.
  
  vector<int32> chunk_status_;
};

struct GenericLayerUpdateConfig {
  // A configuration class for the update of the layers-- this one
  // is generic to all the layer types. [can be used as a base class
  // if needed.  Note: although the class itself is generic, some
  // members (e.g. chunk_size) will typically be different for
  // different layers.
  double learning_rate;
  bool use_fisher;
  int num_chunks; // Number of chunks that the stats class should store.
  int num_threads; // The chunk manager needs to know how many threads we have.
  int chunk_size; // This is layer-specific-- the number of frames in each chunk of
  // features.  [typically in the range 5 to 10.]
  double fisher_average_frames; // Number of frames that the (diagonal) Fisher matrix is
  // averaged over.
};


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
  void Backward(const MatrixBase<BaseFloat> &output_deriv,
                const MatrixBase<BaseFloat> &input,
                MatrixBase<BaseFloat> *input_deriv, // derivative w.r.t. input.
                LinearLayerStats *stats);
  
  friend class LinearLayerStats;
 private:
  Matrix<BaseFloat> params_; // parameters, indexed [output-index][input-index].
  
  // This is in effect the diagonal of the Fisher matrix, computed using
  // a weighted average over time..
  Matrix<BaseFloat> average_derivative_variance_;
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
  // parameters, on held-out set.
  
  // Will accumulate stats [may also update the model.]
  void AccStats(const MatrixBase<BaseFloat> &output_deriv,
                const MatrixBase<BaseFloat> &input);
  
 private:

  void Update(int32 row_start, int32 row_end); // This is
  // called from AccStats and from the destructor.
  
              double learning_rate,
              bool use_fisher); // Updates the layer; also clears the stats.

  LinearLayer *layer_to_update_;
  GenericLayerUpdateConfig config_;
  ChunkManager chunk_manager_;
  Matrix<BaseFloat> stats_;
  SubMatrix<BaseFloat> input_stats_; // sub-matrix of stats_; each row is input vector.
  SubMatrix<BaseFloat> output_stats_; // sub-matrix of stats_; each row is deriv at output.
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
  SoftmaxLayerStats (const SoftmaxLayer &layer,
                     int num_rows);
  
  void Update(LinearLayer *layer,
              double learning_rate,
              bool use_fisher); // Updates the layer; also clears the stats.

 private:
  Mutex mutex_;
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
  
class SigmoidLayerStats {
 public:
  void SawFrame() { num_frames_++; } // must be incremented each time
  // we saw a frame-- this is necessary for getting average_derivative_variance_
  // in SigmoidLayer correctly updated.
  SigmoidLayerStats (const SigmoidLayer &layer);

  void Update(SigmoidLayer *layer,
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

