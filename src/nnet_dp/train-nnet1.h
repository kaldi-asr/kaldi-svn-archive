// nnet_dp/train_nnet1.h

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

#include "nnet_dp/update-nnet1.h"

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


class Nnet1BasicTrainer {
  /*
    The initializer takes: the features [one per training file]; and
    the labels, which are indexed [training file][t][list of pair (category, label)].
    For construction of labels from pdf-ids etc., see am-nnet1.h.    
  */
  Nnet1BasicTrainer(
      const Nnet1BasicTrainerConfig &config,
      const std::vector<CompressedMatrix*> &features,
      const std::vector<std::vector<std::pair<int32, int32> > > &labels,
      const Nnet1 &nnet,
      Nnet1 *nnet_to_update);


  void TrainOnOneMinibatch();
  
  BaseFloat NumEpochs(); // returns approximate number of epochs we've seen already.
  
 private:
  void GetTrainingExamples(std::vector<TrainingExample> *egs); // gets num_chunks_ training
  // examples.

  static void ExtractToMatrix(const CompressedMatrix &input,
                              int32 time_offset,
                              Matrix<BaseFloat> *output);
  
  Nnet1Updater updater_;
  const std::vector<CompressedMatrix*> &features_;
  const std::vector<std::vector<std::pair<int32, int32> > > &labels_;

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
  



} // namespace

#endif // KALDI_NNET_DP_TRAIN_NNET1_H_
