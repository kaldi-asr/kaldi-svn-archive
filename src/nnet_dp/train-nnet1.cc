// nnet_dp/train-nnet1.cc

// Copyright 2012  Johns Hopkins University (author:  Daniel Povey)

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

#include "nnet_dp/train-nnet1.h"

namespace kaldi {

Nnet1BasicTrainer::Nnet1BasicTrainer(
    const Nnet1BasicTrainerConfig &config,    
    const std::vector<CompressedMatrix*> &features,
    const std::vector<std::vector<std::pair<int32, int32> > > &labels,
    const Nnet1 &nnet,
    Nnet1 *nnet_to_update):
    updater_(nnet, config.chunk_size, config.num_chunks, nnet_to_update),
    features_(features),
    labels_(labels),
    chunk_size_(config.chunk_size),
    num_chunks_(config.num_chunks),
    left_context_(nnet.LeftContext()),
    right_context_(nnet.RightContext()),    
    num_chunks_trained_(0) {
  size_t tot_size = 0;
  for (size_t i = 0; i < features.size(); i++)
    tot_size += features[i]->NumRows();
  chunk_queue_.reserve(tot_size / chunk_size_ + 1);
  chunks_per_epoch_ = tot_size / chunk_size_; // This is slightly
  // approximate.
}

//static
void Nnet1BasicTrainer::ExtractToMatrix(const CompressedMatrix &input,
                                        int32 time_offset,
                                        Matrix<BaseFloat> *output) {
  for (int32 row = 0; row < output->NumCols(); row++) {
    SubVector<BaseFloat> dest(*output, row);
    int32 input_row = row + time_offset;
    if (input_row < 0)
      input_row = 0;
    else if (input_row >= input.NumRows())
      input_row = input.NumRows() - 1;
    input.CopyRowToVec(input_row, &dest);
  }
}

void Nnet1BasicTrainer::TrainOnOneMinibatch() {
  std::vector<TrainingExample> egs;
  GetTrainingExamples(&egs);
  updater_.TrainOnOneMinibatch(egs);
}
void Nnet1BasicTrainer::GetTrainingExamples(std::vector<TrainingExample> *egs) {
  // gets num_chunks_ training examples.
  egs->clear();
  egs->resize(num_chunks_);
  for (int32 chunk = 0; chunk < num_chunks_; chunk++) {
    TrainingExample &eg ((*egs)[chunk]);
    eg.weight = 1.0; // We don't use the "weight" feature for now.
    int32 file_id, chunk_offset;
    GetChunk(&file_id, &chunk_offset);
    eg.labels.resize(chunk_size_);
    for (int32 t = 0; t < chunk_size_; t++) {
      KALDI_ASSERT(chunk_offset + t < labels_[file_id].size());
      eg.labels[t] = labels_[file_id + chunk_offset + t];
    }
    const CompressedMatrix &feature_file(*(features_[file_id]));
    // note: the times in the chunk corresponding to the labels must
    // be inside the file, but the context may be outside (we duplicate the
    // frames of context.)
    KALDI_ASSERT(chunk_offset + chunk_size_ <= feature_file.NumRows());
    int32 dim = feature_file.NumCols();
    eg.input.Resize(chunk_size_ + left_context_ + right_context_, dim);
    ExtractToMatrix(feature_file, chunk_offset - left_context_,
                    &eg.input);        
  }  
}


BaseFloat Nnet1BasicTrainer::NumEpochs() {
  return num_chunks_ / static_cast<BaseFloat>(chunks_per_epoch_);
}

void Nnet1BasicTrainer::GetChunk(int32 *file_id, int32 *chunk_offset) {
  if (chunk_queue_.empty()) FillQueue();
  KALDI_ASSERT(!chunk_queue_.empty());
  std::pair<int32,int32> pr = chunk_queue_.back();
  chunk_queue_.pop_back();
  *file_id = pr.first;
  *chunk_offset = pr.second;
  num_chunks_trained_++;
}

void Nnet1BasicTrainer::FillQueue() {
  KALDI_ASSERT(chunk_queue_.empty());

  // Note: with this way of allocating chunks of features we lose a little bit
  // of the features close to the edges of the file, i.e. it's not statistically
  // quite the same as a uniform draw of features; but this is a small effect
  // for typical chunk sizes and we're not too worried about it.
  for (int32 i = 0; i < features_.size(); i++) {
    int32 offset = rand() % chunk_size_;
    for (; offset + chunk_size_ < features_[i]->NumRows(); offset++) {
      chunk_queue_.push_back(std::make_pair(i, offset));
    }
  }
  std::random_shuffle(chunk_queue_.begin(), chunk_queue_.end());
}

  

} // namespace kaldi
