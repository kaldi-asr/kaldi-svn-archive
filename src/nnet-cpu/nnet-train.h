// nnet-cpu/nnet-train.h

// Copyright 2012  Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_NNET_CPU_NNET_TRAIN_H_
#define KALDI_NNET_CPU_NNET_TRAIN_H_

#include "nnet-cpu/nnet-update.h"
#include "nnet-cpu/nnet-compute.h"
#include "util/parse-options.h"

namespace kaldi {



/*


/// Class NnetBackpropComputation is responsible for storing the temporary
/// variables necessary to do propagation and backprop, and
/// either updating the network or computing gradients.  This class
/// is defined in nnet-train.cc because it isn't needed at the use
/// level; its functionality is exported by the function DoBackprop().
///You give it a neural
/// net to do the computation on, and a ponter to a neural net to
/// update; these will be the same for typical SGD training, and
/// different if what you're doing is computing the parameter gradient
/// on a validation set.  The logic for doing things like keeping
/// the learning rates updated lies outside this class.

class NnetBasicTrainer {
 public:
//    The initializer takes: the features [one per training file]; and
//    the labels, which are indexed [training file][t][list of pair (category, label)].
//    For construction of labels from pdf-ids etc., see am-nnet1.h.    
  Nnet1BasicTrainer(
      const NnetBasicTrainerConfig &config,
      Nnet &nnet,
      Nnet *nnet_to_update); // may equal &nnet or may be different.

  void void UseOneMinibatch

  
  BaseFloat TrainOnOneMinibatch(); // returns average objective function over this minibatch.
  
  BaseFloat NumEpochs(); // returns approximate number of epochs we've seen already.

  const Nnet1 &Nnet() const { return updater_.Nnet(); }
  Nnet1 &Nnet() { return *updater_.NnetToUpdate(); } // actually same object as updater.Nnet().
  int32 NumChunks() { return num_chunks_; }
  int32 ChunkSize() { return chunk_size_; }
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
  int32 num_chunks_; // Num chunks per minibatch.
  int32 left_context_; // left context required by the network.
  int32 right_context_; // right context required by the network.
  
  void FillQueue();
  void GetChunk(int32 *file_id, int32 *chunk_offset);
                
  std::vector<std::pair<int32, int32> > chunk_queue_; // queue of chunks, in randomized
  // order, in the form of pairs (file-index, start-time).

  int32 num_chunks_trained_;
  int32 chunks_per_epoch_;
};

*/


/// Class Nnet1ValidationSet stores the validation set feature data and labels,
/// and is responsible for calling code that computes the objective function and
/// gradient on the validation set.
class NnetValidationSet {
 public:
  NnetValidationSet() { }

  void AddUtterance(const MatrixBase<BaseFloat> &features,
                    const VectorBase<BaseFloat> &spk_info, // may be empty
                    std::vector<int32> &pdf_ids,
                    BaseFloat utterance_weight = 1.0);
  // Here, "nnet" will be a neural net and "gradient" will be a copy of it that
  // this function will overwrite with the gradient.  This function will compute
  // the gradient and return the average per-frame objective function.
  BaseFloat ComputeGradient(const Nnet &nnet,
                            Nnet *gradient) const;
                    
  ~NnetValidationSet();
 private:
  struct Utterance {
    Matrix<BaseFloat> features;
    Vector<BaseFloat> spk_info;
    std::vector<int32> pdf_ids;
    BaseFloat weight;
    Utterance(const MatrixBase<BaseFloat> &features_in,
              const VectorBase<BaseFloat> &spk_info_in,
              const std::vector<int32> &pdf_ids_in,
              BaseFloat weight_in): features(features_in),
                                    spk_info(spk_info_in),
                                    pdf_ids(pdf_ids_in),
                                    weight(weight_in) { }
  };
  std::vector<Utterance*> utterances_;
};
  

struct NnetAdaptiveTrainerConfig {
  int32 minibatch_size;
  int32 minibatches_per_phase;
  BaseFloat learning_rate_ratio;
  BaseFloat shrinkage_rate_ratio;
  BaseFloat max_learning_rate;
  BaseFloat min_shrinkage_rate;
  BaseFloat max_shrinkage_rate;  
  int32 num_phases;
  
  NnetAdaptiveTrainerConfig():
      minibatch_size(500), minibatches_per_phase(50), learning_rate_ratio(1.1),
      shrinkage_rate_ratio(1.1), max_learning_rate(0.1),
      min_shrinkage_rate(1.0e-20), max_shrinkage_rate(0.001),
      num_phases(50) { }
  
  void Register (ParseOptions *po) {
    po->Register("minibatch-size", &minibatch_size,
                 "Number of samples per minibatch of training data.");
    po->Register("minibatches-per-phase", &minibatches_per_phase,
                 "Number of minibatches accessed in each phase of training "
                 "(after each phase we adjust learning rates");
    po->Register("learning-rate-ratio", &learning_rate_ratio,
                 "Ratio by which we change the learning rate in each phase of "
                 "training (can get larger or smaller by this factor).");
    po->Register("max-learning-rate", &max_learning_rate,
                 "Maximum learning rate we allow when dynamically updating "
                 "learning and shrinkage rates");
    po->Register("min-shrinkage-rate", &min_shrinkage_rate,
                 "Minimum allowed shrinkage rate.");
    po->Register("max-shrinkage-rate", &max_shrinkage_rate,
                 "Maximum allowed shrinkage rate.");
    po->Register("num-phases", &num_phases,
                 "Number of \"phases\" of training (a phase is a sequence of "
                 "num-minibatches minibatches; after each phase we modify "
                 "learning rates).  If <= 0, continue till input stream stops.");
  }  
};

// Class NnetAdaptiveTrainer is responsible for changing the learning rate using
// the gradients on the validation set, and calling the SGD training code in
// nnet-update.h.  It takes in the training examples through the call
// "TrainOnExample()", which means that the I/O code that reads in the training
// examples can be in the .cc file (we prefer to segregate that out).
class NnetAdaptiveTrainer {
 public:
  NnetAdaptiveTrainer(const NnetAdaptiveTrainerConfig &config,
                      Nnet *nnet,
                      NnetValidationSet *validation_set);

  /// TrainOnExample will take the example and add it to a buffer;
  /// if we've reached the minibatch size it will do the training.
  void TrainOnExample(const NnetTrainingExample &value);
 private:


  void TrainOneMinibatch();
  
  // The following two functions are called by TrainOneMinibatch()
  // when we enter a new phase.
  void BeginNewPhase();
  void EndNewPhase();
  
  void PrintProgress();  
  void TrainOnePhase();

  // Things we were given in the initializer:
  NnetAdaptiveTrainerConfig config_;
  Nnet *nnet_; // the nnet we're training.
  NnetValidationSet *validation_set_; // Stores validation data, used
  // to compute gradient on validation set.

  // State information:  
  int32 minibatches_seen_this_phase_;
  std::vector<NnetTrainingExample> buffer_;
  BaseFloat validation_objf_; // stores validation objective function at
  // start/end of phase.
  Nnet nnet_at_phase_start_; // Snapshot of the neural net at the start
  // of this phase of training.

  
  //
  BaseFloat initial_validation_objf_; // validation objf at start.
  Vector<BaseFloat> progress_stats_; // Per-layer stats on progress so far.

  std::vector<std::vector<int32> > final_layer_sets_; // relates to updating
  // learning rates.
};



} // namespace

#endif

