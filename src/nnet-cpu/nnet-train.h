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
#include "util/parse-options.h"

namespace kaldi {


/// Configuration variables that will be given to the program that
/// randomizes and weights the data for us.
struct NnetDataRandomizerConfig {
  /// If a particular class appears with a certain frequency f, we'll make it
  /// appear at frequency f^{frequency_power}, and reweight the samples with a
  /// higher weight to compensate.  Note: we'll give these weights an overall
  /// scale such that the expected weight of any given sample is 1; this helps
  /// keep this independent from the learning rates.  frequency_power=1.0
  /// means "normal" training.  We probably want between 0.5 and 1.0.
  BaseFloat frequency_power;
  
  int32 num_samples; // Total number of samples we want to train on (if >0).  The program
  // will select this many samples before it stops.

  BaseFloat num_epochs; // Total number of epochs we want (if >0).  The program will run
  // for this many epochs before it stops.

  NnetDataRandomizerConfig(): frequency_power(0.5), num_samples(1000000),
                              num_epochs(-1) { }

  void Register(ParseOptions *po) {
    po->Register("frequency-power", &frequency_power, "Power by which we rescale "
                 "the frequencies of samples.");
    po->Register("num-epochs", &num_epochs, "If >0, this will define how many "
                 "times to train on the whole data.  Note, we will see some "
                 "samples more than once if frequency-power < 1.0.");
    po->Register("num-samples", &num_samples, "The number of samples of training "
                 "data to train on.");
  }

};

/// This class does the job of randomizing and reweighting the data,
/// before training on it (the weights on samples are a mechanism
/// to make common classes less common, to avoid wasting time,
/// but then upweighting the samples so all the expectations are the
/// the same.
class NnetDataRandomizer {
 public:
  NnetDataRandomizer(const Nnet &nnet,
                     const NnetDataRandomizerConfig &config);
      
  void AddTrainingFile(const Matrix<BaseFloat> &feats,
                       const Vector<BaseFloat> &spk_info,
                       const std::vector<int32> &labels);
  
  bool Done();
  void Next();
  const NnetTrainingExample &Value();
  ~NnetDataRandomizer();
 private:
  /// Called from RandomizeSamples().
  void GetPdfCounts(std::vector<int32> *pdf_counts);
  /// Called from RandomizeSamples().  Get samples indexed first
  /// by pdf-id, without any randomization or reweighting.
  void GetRawSamples(
      std::vector<std::vector<std::pair<int32, int32> > > *pdf_counts);

  /// Called from RandomizeSamples().  Takes the samples indexed first by pdf,
  /// which are assumed to be in random order for each pdf, and writes them in
  /// pseudo-random order to *samples as one long sequence.  Uses a recursive
  /// algorithm (based on splitting in two) that is designed to ensure a kind
  /// of balance, e.g. each time we split in two we try to distribute examples
  /// of a pdf equally between the two splits.  This will tend to reduce
  /// the variance of the parameter estimates.  Note: the samples_by_pdf_input
  /// is the input but is destroyed by the algorithm to save memory.
  static void RandomizeSamplesRecurse(
      std::vector<std::vector<std::pair<int32, int32> > > *samples_by_pdf_input,
      std::vector<std::pair<int32, int32> > *samples);
  
  /// Called when samples_ is empty: sets
  /// up samples_ and pdf_weights_.  
  void RandomizeSamples(); 

  struct TrainingFile {
    CompressedMatrix feats;
    Vector<BaseFloat> spk_info;
    std::vector<int32> labels; // Vector of pdf-ids (targets for training).
  };

  const Nnet &nnet_;
  NnetDataRandomizerConfig config_;    
  std::vector<TrainingFile*> data_;

  int32 num_samples_tgt_;
  int32 num_samples_given_;
  std::vector<std::pair<int32, int32> > samples_; // each time we randomize
  // the whole data, we store it here.
  std::vector<BaseFloat> pdf_weights_; // each time we randomize the data,
  // we compute a new weighting for each pdf, which is to cancel out the
  // difference in frequency between the original frequency and the sampled
  // frequency.
  NnetTrainingExample cur_example_; // Returned from Value().  NnetDataRandomizerConfig_ config_;
};




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
  /*
    The initializer takes: the features [one per training file]; and
    the labels, which are indexed [training file][t][list of pair (category, label)].
    For construction of labels from pdf-ids etc., see am-nnet1.h.    
  */
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
  // and returns the objective function.  These are both divided by the #frames
  // in the validation set, which is more useful for diagnostic purposes.
  const Nnet1 &Gradient() const { return *gradient_; }
 private:
  const std::vector<CompressedMatrix> &features_;
  const std::vector<std::vector<int32> > &pdf_ids_;
  int32 tot_num_frames_; // total #frames in validation set.
  BaseFloat objf();
  const AmNnet1 &am_nnet_;
  Nnet1 *gradient_; // store the gradient as class Nnet1.   Not owned here.
};
  

struct Nnet1AdaptiveTrainerConfig {
  int32 num_minibatches;
  BaseFloat learning_rate_ratio;
  BaseFloat shrinkage_rate_ratio;
  BaseFloat max_learning_rate;
  BaseFloat min_shrinkage_rate;
  BaseFloat max_shrinkage_rate;  
  int32 num_phases;

  Nnet1AdaptiveTrainerConfig():
      num_minibatches(50), learning_rate_ratio(1.1), shrinkage_rate_ratio(1.1),
      max_learning_rate(0.1), min_shrinkage_rate(1.0e-20), max_shrinkage_rate(0.001),
      num_phases(50) { }
  
  void Register (ParseOptions *po) {
    po->Register("num-minibatches", &num_minibatches,
                 "Number of minibatches accessed in each phase of training "
                 "(after each phase we adjust learning rates");
    po->Register("learning-rate-ratio", &learning_rate_ratio,
                 "Ratio by which we change the learning rate in each phase of "
                 "training (can get larger or smaller by this factor).");
    po->Register("max-learning-rate", &max_learning_rate,
                 "Maximum learning rate we allow when dynamically updating learning and shrinkage rates");
    po->Register("min-shrinkage-rate", &min_shrinkage_rate,
                 "Minimum allowed shrinkage rate.");
    po->Register("max-shrinkage-rate", &max_shrinkage_rate,
                 "Maximum allowed shrinkage rate.");
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
 public:
  Nnet1AdaptiveTrainer(const Nnet1AdaptiveTrainerConfig &config,
                       const std::vector<std::vector<int32> > &final_layer_sets,
                       Nnet1BasicTrainer *basic_trainer,
                       Nnet1ValidationSet *validation_set);
  void Train();
 private:
  void PrintProgress();
  void TrainOnePhase();
  Nnet1BasicTrainer *basic_trainer_; // Not owned here.
  Nnet1ValidationSet *validation_set_; // Stores validation gradient.  Not owned here.
  BaseFloat validation_objf_; // stores validation objective function.
  BaseFloat initial_validation_objf_; // validation objf at start.
  Nnet1ProgressInfo progress_stats_;
  Nnet1AdaptiveTrainerConfig config_;
  std::vector<std::vector<int32> > final_layer_sets_; // relates to updating
  // learning rates.
};



} // namespace

#endif
