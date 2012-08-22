// nnet-dp/train_nnet1.h

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

#ifndef KALDI_NNET_DP_TRAIN_NNET1_H_
#define KALDI_NNET_DP_TRAIN_NNET1_H_

#include "nnet-dp/update-nnet1.h"
#include "nnet-dp/am-nnet1.h" // Note: the reason we include
// am-nnet1 here, and use that class, is because it's responsible
// for converting pdf-ids to a vector of pairs of integers, and
// we prefer (for compactness) to have this code take vectors of
// pdf-ids.

namespace kaldi {


struct Nnet1BasicTrainerConfig {
  int32 chunk_size; // chunk size (measured at output).
  int32 num_chunks; // number of chunks in a minibatch.

  Nnet1BasicTrainerConfig():
      chunk_size(5), num_chunks(200) { }
  
  void Register (ParseOptions *po) {
    po->Register("chunk-size", &chunk_size,
                 "Size of chunks of features to train on.  Should typically be about "
                 "the same order as the total #frames of (left and right) context.");
    po->Register("num-chunks", &num_chunks,
                 "Number of chunks per minibatch, that we train on.");
  }
};

// Class Nnet1BasicTrainer is responsible for randomly selecting a minibatch of training
// data, and training on the minibatch using SGD.
class Nnet1BasicTrainer {
 public:
  /*
    The initializer takes: the features [one per training file]; and
    the labels, which are indexed [training file][t][list of pair (category, label)].
    For construction of labels from pdf-ids etc., see am-nnet1.h.    
  */
  Nnet1BasicTrainer(
      const Nnet1BasicTrainerConfig &config,
      const std::vector<CompressedMatrix> &features,
      const std::vector<std::vector<int32> > &pdf_ids,
      AmNnet1 *am_nnet);

  BaseFloat TrainOnOneMinibatch(); // returns average objective function over this minibatch.
  
  BaseFloat NumEpochs(); // returns approximate number of epochs we've seen already.

  const Nnet1 &Nnet() const { return updater_.Nnet(); }
  Nnet1 &Nnet() { return *updater_.NnetToUpdate(); } // actually same object as updater.Nnet().
 private:
  void GetTrainingExamples(std::vector<TrainingExample> *egs); // gets num_chunks_ training
  // examples.

  static void ExtractToMatrix(const CompressedMatrix &input,
                              int32 time_offset,
                              Matrix<BaseFloat> *output);

  AmNnet1 *am_nnet_; // not owned here.
  Nnet1Updater updater_;
  const std::vector<CompressedMatrix> &features_;
  const std::vector<std::vector<int32> > &pdf_ids_;

  int32 chunk_size_;
  int32 num_chunks_;
  int32 left_context_; // left context required by the network.
  int32 right_context_; // right context required by the network.
  
  void FillQueue();
  void GetChunk(int32 *file_id, int32 *chunk_offset);
                
  std::vector<std::pair<int32, int32> > chunk_queue_; // queue of chunks, in randomized
  // order, in the form of pairs (file-index, start-time).

  int32 num_chunks_trained_;
  int32 chunks_per_epoch_;
};

// Class Nnet1ValidationSet computes the objective function and gradient on the
// validation set.  Note: in this case we don't use the chunk mechanism in quite
// the same way: since we're summing the gradient, there's no point in breaking
// things up into small chunks [and we'd lose something at the edges if we did this.]
// Instead we just treat each file as a single chunk, and in fact use minibatches
// of just one (typically large) chunk.  We trust ATLAS to just do the right thing.
class Nnet1ValidationSet {
 public:
  Nnet1ValidationSet(
      const std::vector<CompressedMatrix> &features,
      const std::vector<std::vector<int32> > &labels,
      const AmNnet1 &am_nnet,
      Nnet1 *gradient); // store the gradient as class Nnet1.
  
  BaseFloat ComputeGradient(); // Computes the gradient (stored in *gradient)
  // and returns average objective function over batch.
  const Nnet1 &Gradient() const { return *gradient_; }
 private:
  const std::vector<CompressedMatrix> &features_;
  const std::vector<std::vector<int32> > &pdf_ids_;
  const AmNnet1 &am_nnet_;
  Nnet1 *gradient_; // store the gradient as class Nnet1.   Not owned here.
};
  

struct Nnet1AdaptiveTrainerConfig {
  int32 num_minibatches;
  int32 learning_rate_ratio;
  int32 num_phases;

  Nnet1AdaptiveTrainerConfig():
      num_minibatches(50), learning_rate_ratio(1.1),
      num_phases(50) { }
  
  void Register (ParseOptions *po) {
    po->Register("num-minibatches", &num_minibatches,
                 "Number of minibatches accessed in each phase of training "
                 "(after each phase we adjust learning rates");
    po->Register("learning-rate-ratio", &learning_rate_ratio,
                 "Ratio by which we change the learning rate in each phase of "
                 "training (can get larger or smaller by this factor).");
    po->Register("num-phases", &num_phases,
                 "Number of \"phases\" of training (a phase is a sequence of "
                 "num-minibatches minibatches; after each phase we modify "
                 "learning rates.");
  }  
};

// Class Nnet1AdaptiveTrainer takes a reference to class Nnet1BasicTrainer, and
// a reference to class Nnet1ValidationSet (for computation of gradient and
// objective function on the validation set).  This class is responsible for
// changing the learning rate using the gradients on the validation set.
class Nnet1AdaptiveTrainer {
  Nnet1AdaptiveTrainer(const Nnet1AdaptiveTrainerConfig &config,
                       Nnet1BasicTrainer *basic_trainer,
                       Nnet1ValidationSet *validation_set);
  void Train();
 private:
  void TrainOnePhase();
  
  Nnet1BasicTrainer *basic_trainer_; // Not owned here.
  Nnet1ValidationSet *validation_set_; // Not owned here.
  Nnet1AdaptiveTrainerConfig config_;
};



} // namespace

#endif // KALDI_NNET_DP_TRAIN_NNET1_H_
