// nnet-dp/train-nnet1.cc

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

#include "nnet-dp/train-nnet1.h"

namespace kaldi {
using std::vector;
using std::pair;

Nnet1BasicTrainer::Nnet1BasicTrainer(
    const Nnet1BasicTrainerConfig &config,    
    const std::vector<CompressedMatrix> &features,
    const vector<vector<int32> > &pdf_ids,
    AmNnet1 *am_nnet):
    am_nnet_(am_nnet),
    updater_(am_nnet->Nnet(),
             config.chunk_size, config.num_chunks,
             &(am_nnet->Nnet())),
    features_(features),
    pdf_ids_(pdf_ids),
    chunk_size_(config.chunk_size),
    num_chunks_(config.num_chunks),
    left_context_(am_nnet_->Nnet().LeftContext()),
    right_context_(am_nnet_->Nnet().RightContext()),    
    num_chunks_trained_(0) {
  size_t tot_size = 0;
  for (size_t i = 0; i < features.size(); i++)
    tot_size += features[i].NumRows();
  chunk_queue_.reserve(tot_size / chunk_size_ + 1);
  chunks_per_epoch_ = tot_size / chunk_size_; // This is slightly
  // approximate.
}

//static
void Nnet1BasicTrainer::ExtractToMatrix(const CompressedMatrix &input,
                                        int32 time_offset,
                                        Matrix<BaseFloat> *output) {
  for (int32 row = 0; row < output->NumRows(); row++) {
    SubVector<BaseFloat> dest(*output, row);
    int32 input_row = row + time_offset;
    if (input_row < 0)
      input_row = 0;
    else if (input_row >= input.NumRows())
      input_row = input.NumRows() - 1;
    input.CopyRowToVec(input_row, &dest);
  }
}

BaseFloat Nnet1BasicTrainer::TrainOnOneMinibatch() {
  vector<TrainingExample> egs;
  GetTrainingExamples(&egs);
  return updater_.TrainOnOneMinibatch(egs);
}
void Nnet1BasicTrainer::GetTrainingExamples(vector<TrainingExample> *egs) {
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
      KALDI_ASSERT(chunk_offset + t < pdf_ids_[file_id].size());
      // The next call turns the pdf_id (type int32) into a vector of
      // (typically two) pairs of (int32, int32), which is the labels in
      // the two-level tree structure.
      am_nnet_->GetCategoryInfo(pdf_ids_[file_id][chunk_offset + t], &eg.labels[t]);
    }
    const CompressedMatrix &feature_file(features_[file_id]);
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
  return num_chunks_trained_ / static_cast<BaseFloat>(chunks_per_epoch_);
}

void Nnet1BasicTrainer::GetChunk(int32 *file_id, int32 *chunk_offset) {
  if (chunk_queue_.empty()) FillQueue();
  KALDI_ASSERT(!chunk_queue_.empty());
  pair<int32,int32> pr = chunk_queue_.back();
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
    for (; offset + chunk_size_ < features_[i].NumRows(); offset++) {
      chunk_queue_.push_back(std::make_pair(i, offset));
    }
  }
  std::random_shuffle(chunk_queue_.begin(), chunk_queue_.end());
}

Nnet1ValidationSet::Nnet1ValidationSet(
    const vector<CompressedMatrix> &features,
    const vector<vector<int32> > &pdf_ids,
    const AmNnet1 &am_nnet,
    Nnet1 *gradient):
    features_(features), pdf_ids_(pdf_ids), am_nnet_(am_nnet), gradient_(gradient) {
  tot_num_frames_ = 0;
  for (int32 i = 0; i < pdf_ids_.size(); i++)
    tot_num_frames_ += pdf_ids_[i].size();
}

// Computes the gradient (stored in *gradient)
BaseFloat Nnet1ValidationSet::ComputeGradient() {
  double tot_objf = 0.0;
      
  gradient_->SetZero();
  for (int32 f = 0; f < features_.size(); f++) { // for each file..
    const CompressedMatrix &compressed_feats(features_[f]);
    int32 left_context = am_nnet_.Nnet().LeftContext(),
        right_context = am_nnet_.Nnet().RightContext(),
        num_frames = compressed_feats.NumRows(),
        dim = compressed_feats.NumCols(),
        padded_num_frames = left_context + num_frames + right_context;

    vector<TrainingExample> egs(1);
    TrainingExample &eg = egs[0];

    KALDI_ASSERT(pdf_ids_[f].size() == num_frames);
    
    eg.weight = 1.0 / tot_num_frames_;
    eg.input.Resize(padded_num_frames, dim); // include needed context...
    
    SubMatrix<BaseFloat> features_part(eg.input, left_context, num_frames,
                                       0, dim);
    features_[f].CopyToMat(&features_part);
    for (int32 t = 0; t < left_context; t++) {
      SubVector<BaseFloat> src(eg.input, left_context), dst(eg.input, t);
      dst.CopyFromVec(src);
    }
    for (int32 t = 0; t < right_context; t++) {
      SubVector<BaseFloat> src(eg.input, padded_num_frames - right_context - 1),
          dst(eg.input, padded_num_frames - t - 1);
      dst.CopyFromVec(src);
    }

    eg.labels.resize(num_frames); // note: labels are not padded.
    for (int32 t = 0; t < num_frames; t++) {
      // the next call converts from an int32 to vector<pair<int32, int32> >
      am_nnet_.GetCategoryInfo(pdf_ids_[f][t], &eg.labels[t]);
    }

    int32 num_chunks = 1;
    Nnet1Updater updater(am_nnet_.Nnet(), num_frames, num_chunks, gradient_);
    BaseFloat avg_objf = updater.TrainOnOneMinibatch(egs);
    tot_objf += num_frames * avg_objf;
  }
  KALDI_VLOG(2) << "Objective function on validation set "
                << (tot_objf / tot_num_frames_) << " per frame, over "
                << tot_num_frames_ << " frames.";
  return tot_objf / tot_num_frames_;
}

Nnet1AdaptiveTrainer::Nnet1AdaptiveTrainer(
    const Nnet1AdaptiveTrainerConfig &config,
    Nnet1BasicTrainer *basic_trainer,
    Nnet1ValidationSet *validation_set):
    basic_trainer_(basic_trainer),
    validation_set_(validation_set),
    config_(config) {
}

void Nnet1AdaptiveTrainer::TrainOnePhase() {
  Nnet1 nnet_at_start(basic_trainer_->Nnet()); // deep copy.
  double train_objf = 0.0;
  for (int32 m = 0; m < config_.num_minibatches; m++)
    train_objf += basic_trainer_->TrainOnOneMinibatch();
  train_objf /= config_.num_minibatches;
  
  Nnet1 &nnet_at_end(basic_trainer_->Nnet()); // this one
  // is a reference, not a copy.

  Nnet1ProgressInfo old_progress_info, new_progress_info; // certain dot products
  // old_progress_info is dot product of delta parameters with
  // (validation-set gradient before change in parameters).
  // new_progress_info is the same after the change in parameters.
  basic_trainer_->Nnet().ComputeProgressInfo(
      nnet_at_start,
      validation_set_->Gradient(),
      &old_progress_info);

  // Now recompute validation set objf and gradient.
  validation_objf_ = validation_set_->ComputeGradient();
  KALDI_VLOG(1) << "Average objf is " << train_objf << " (train) and "
                << validation_objf_ << " (test).";
  
  basic_trainer_->Nnet().ComputeProgressInfo(
      nnet_at_start,
      validation_set_->Gradient(),
      &new_progress_info);
  
  nnet_at_end.AdjustLearningRates(new_progress_info,
                                  config_.learning_rate_ratio);

  UpdateProgressStats(old_progress_info, new_progress_info, &progress_stats_);

  KALDI_VLOG(2) << "Progress stats: " << progress_stats_.Info();
  KALDI_VLOG(2) << "Learning rates: " << basic_trainer_->Nnet().LrateInfo();
  KALDI_VLOG(3) << "Neural net info: " << basic_trainer_->Nnet().Info();
}

void Nnet1AdaptiveTrainer::Train() {
  // We need the validation set gradient to be computed
  // prior to each iteration, for purposes of computing
  // progress, so we do this the first time.
  // Later we use the gradient that was stored at the
  // end of the previous iteration.
  validation_objf_ = validation_set_->ComputeGradient();
  initial_validation_objf_ = validation_objf_;
  
  for (int32 i = 0; i < config_.num_phases; i++) {
    KALDI_LOG << "Phase " << i << " of " << config_.num_phases
              << " ( " << basic_trainer_->NumEpochs() << " epochs.)";
    TrainOnePhase();
    KALDI_VLOG(1) << "Validation set progress, based on gradients, by layer: "
                  << progress_stats_.Info();
    KALDI_VLOG(1) << "Actual validation-set improvement is "
                  << (validation_objf_ - initial_validation_objf_)
                  << " ( " << initial_validation_objf_ << " -> "
                  << validation_objf_ << " ) ";
  }
  KALDI_LOG << "Validation set progress, based on gradients, by layer: "
            << progress_stats_.Info();
  KALDI_LOG << "Actual validation-set improvement is "
            << (validation_objf_ - initial_validation_objf_)
            << " ( " << initial_validation_objf_ << " -> "
            << validation_objf_;
}



} // namespace kaldi
