// nnet/nnet-nnet.cc

// Copyright 2011-2012  Karel Vesely
//                      Johns Hopkins University (author: Daniel Povey)

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

#include "nnet-cpu/nnet-nnet.h"

namespace kaldi {

const std::vector<int32> &Nnet::RawSplicingForComponent(int32 component) {
  KALDI_ASSERT(static_cast<size_t>(component) < raw_splicing_.size());
  return  raw_splicing_[component];
}

const std::vector<int32> &Nnet::FullSplicingForComponent(int32 component) {
  KALDI_ASSERT(static_cast<size_t>(component) < full_splicing_.size());
  return  full_splicing_[component];
}

const std::vector<std::vector<int32> > &Nnet::RelativeSplicingForComponent(
    int32 component) {
  KALDI_ASSERT(static_cast<size_t>(component) < relative_splicing_.size());
  return  relative_splicing_[component];
}


int32 Nnet::OutputDim() const {
  KALDI_ASSERT(!components_.empty());
  return components_.back()->OutputDim();
}

int32 Nnet::FeatureDim() const {
  KALDI_ASSERT(!components_empty());
  int32 spliced_feature_dim = components_.front()->InputDim() - speaker_info_dim_,
      num_spliced = raw_splicing_[0].size();
  
  KALDI_ASSERT(spliced_feature_dim > 0 && num_spliced > 0);
  KALDI_ASSERT(spliced_feature_dim % num_spliced == 0);
  return spliced_feature_dim / num_spliced;
}

const Component& Nnet::GetComponent(int32 component) const {
  KALDI_ASSERT(static_cast<size_t>(component) < components_.size());
  return *(components_[component]);
}

Component& Nnet::GetComponent(int32 component) {
  KALDI_ASSERT(static_cast<size_t>(component) < components_.size());
  return *(components_[component]);
}

int32 Nnet::RightContext() const {
  return std::accumulate(right_context_.begin(), right_context_.end(), 0);
}
int32 Nnet::LeftContext() const {
  return std::accumulate(left_context_.begin(), left_context_.end(), 0);
}

int32 Nnet::LeftContextForComponent(int32 component) const {
  KALDI_ASSERT(component >= 0 && component < components_.size());
  return left_context_[component];
}
int32 Nnet::RightContextForComponent(int32 component) const {
  KALDI_ASSERT(component >= 0 && component < components_.size());
  return right_context_[component];
}
const Component &Nnet::GetComponent(int32 component) const {
  KALDI_ASSERT(component >= 0 && component < components_.size());
  return *(components_[component]);
}

Component &Nnet::GetComponent(int32 component) {
  KALDI_ASSERT(component >= 0 && component < components_.size());
  return *(components_[component]);
}

/**
   Let's imagine we have three components.
   Imagine raw_splicing is the following array:
   [ [ 0 5 ]
     [ 0 ]
     [ -1 1 1 ]
     [ 0 ] ]
   Note: the last element doesn't correspond to any component; it just
   happens to be convenient.  This means that the input component splices
   frames 0 and +5, the middle component doesn't splice, and the last component
   splices 3 adjacent frames.

   full_splicing will be:
   [ [ -1 1 1 4 5 6 ]
     [ -1 1 1 ]
     [ -1 1 1 ]
     [ 0 ] ]

   relative_splicing will be:
   [ [ 0 3 ], [ 1 4 ], [ 2 5 ],
     [ 0 ], [ 1 ], [2],
     [ 0 1 2 ] ]
 */
void Nnet::InitializeArrays() {
  // Check that raw_splicing_ has an entry for each component,
  // plus one [ 0 ] at the end.
  KALDI_ASSERT(raw_splicing_.size() == components_.size() + 1 &&
               raw_splicing_.back().size() == 1 &&
               raw_splicing_.back()[0] == 0);
  // Set up full_splicing_.  This is the frame indices at the input of the
  // indexed component that we need to get a single frame of output.  And
  // full_splicing_[NumComponents()]  == [ 0 ].
  full_splicing_.clear();
  full_splicing_.resize(raw_splicing_.size());
  // Set the last one (index == components.size()) to [ 0 ].
  full_splicing_.back().resize(1); 
  full_splicing_.back()[0] = 1;
  // full_splicing[c] is a function of full_splicing[c+1] and
  // raw_splicing[c].
  for (int32 c = full_splicing_.size() - 2; c >= 0; c--) {
    for (size_t i = 0; i < full_splicing_[c+1].size(); i++) {
      for (size_t j = 0; j < raw_splicing_[c].size(); j++) {
        int32 fi = full_splicing_[c+1][i], rj = raw_splicing_[c][j];
        full_splicing_[c].push_back(fi + rj);
      }
    }
    SortAndUniq(&full_splicing_[c]);
  }

  relative_splicing_.clear();
  relative_splicing_.resize(components_.size()); // one smaller than
  // the other two arrays.
  for (int32 c = 0; c < relative_splicing_.size(); c++) {
    relative_splicing_[c].resize(full_splicing_[c+1].size());
    std::vector<int32> check;
    for (int32 i = 0; i < full_splicing_[c+1].size(); i++) { // i is
      // the index into the list of spliced frames at the output
      // of this layer.
      relative_splicing_[c][i].resize(raw_splicing_[c].size());
      // for each index i, we'll have an entry for each thing we
      // have to splice together from the input.
      for (int32 j = 0; j < raw_splicing_[c].size(); j++) {
        int32 frame_offset = full_splicing_[c+1][i] + raw_splicing_[c][j];
        // frame_offset is the frame offset we're looking for.. we want
        // to turn it into an index into the list full_splicing_[c].
        int32 index = -1;
        for (k = 0; k < full_splicing_[c].size(); k++)
          if (full_splicing_[c][k] == frame_offset)
            index = k;
        KALDI_ASSERT(index != -1);
        relative_splicing_[c][i][j] = index;
        check.push_back(index);
      }
    }
    SortAndUniq(&check);
    KALDI_ASSSERT(check.size() == full_splicing_[c].size()
                  && check.front() == 0 && check.back() + 1 == check.size());
    // Make sure we used everything, otherwise something's wrong.
  }
}


const std::vector<int32>& Nnet::RawSplicingForComponent(int32 component) const {
  KALDI_ASSERT(static_cast<size_t>(component) < raw_splicing_.size());
  return raw_splicing_[component];
}

const std::vector<int32>& Nnet::FullSplicingForComponent(int32 component) const {
  KALDI_ASSERT(static_cast<size_t>(component) < full_splicing_.size());
  return full_splicing_[component];
}

const std::vector<int32>& Nnet::RelativeSplicingForComponent(
    int32 component) const {
  KALDI_ASSERT(static_cast<size_t>(component) < relative_splicing_.size());
  return relative_splicing_[component];
}

int32 Nnet::OutputDim() const {
  return components_.Back()->OutputDim();
}

void Nnet::Propagate(const Matrix<BaseFloat> &in, Matrix<BaseFloat> *out) {
  KALDI_ASSERT(NULL != out);

  if (LayerCount() == 0) { 
    out->Resize(in.NumRows(), in.NumCols(), kUndefined);
    out->CopyFromMat(in); 
    return; 
  }

  // we need at least L+1 input buffers
  KALDI_ASSERT((int32)propagate_buf_.size() >= LayerCount()+1);

  
  propagate_buf_[0].Resize(in.NumRows(), in.NumCols(), kUndefined);
  propagate_buf_[0].CopyFromMat(in);

  for(int32 i=0; i<(int32)nnet_.size(); i++) {
    nnet_[i]->Propagate(propagate_buf_[i], &propagate_buf_[i+1]);
  }

  Matrix<BaseFloat> &mat = propagate_buf_[nnet_.size()];
  out->Resize(mat.NumRows(), mat.NumCols(), kUndefined);
  out->CopyFromMat(mat);
}


void Nnet::Backpropagate(const Matrix<BaseFloat> &in_err, Matrix<BaseFloat> *out_err) {
  if(LayerCount() == 0) { KALDI_ERR << "Cannot backpropagate on empty network"; }

  // we need at least L+1 input bufers
  KALDI_ASSERT((int32)propagate_buf_.size() >= LayerCount()+1);
  // we need at least L-1 error bufers
  KALDI_ASSERT((int32)backpropagate_buf_.size() >= LayerCount()-1);

  // find out when we can stop backprop
  int32 backprop_stop = -1;
  if (NULL == out_err) {
    backprop_stop++;
    while (1) {
      if (nnet_[backprop_stop]->IsUpdatable()) {
        if (0.0 != dynamic_cast<UpdatableComponent*>(nnet_[backprop_stop])->LearnRate()) {
          break;
        }
      }
      backprop_stop++;
      if (backprop_stop == (int32)nnet_.size()) {
        KALDI_ERR << "All layers have zero learning rate!";
        break;
      }
    }
  }
  // disable!
  backprop_stop=-1;

  //////////////////////////////////////
  // Backpropagation
  //

  // don't copy the in_err to buffers, use it as is...
  int32 i = nnet_.size()-1;
  if (nnet_[i]->IsUpdatable()) {
    UpdatableComponent *uc = dynamic_cast<UpdatableComponent*>(nnet_[i]);
    if (uc->LearnRate() > 0.0) {
      uc->Update(propagate_buf_[i], in_err);
    }
  }
  nnet_.back()->Backpropagate(in_err, &backpropagate_buf_[i-1]);

  // backpropagate by using buffers
  for(i--; i >= 1; i--) {
    if (nnet_[i]->IsUpdatable()) {
      UpdatableComponent *uc = dynamic_cast<UpdatableComponent*>(nnet_[i]);
      if (uc->LearnRate() > 0.0) {
        uc->Update(propagate_buf_[i], backpropagate_buf_[i]);
      }
    }
    if (backprop_stop == i) break;
    nnet_[i]->Backpropagate(backpropagate_buf_[i], &backpropagate_buf_[i-1]);
  }

  // update first layer 
  if (nnet_[0]->IsUpdatable()  &&  0 >= backprop_stop) {
    UpdatableComponent *uc = dynamic_cast<UpdatableComponent*>(nnet_[0]);
    if (uc->LearnRate() > 0.0) {
      uc->Update(propagate_buf_[0], backpropagate_buf_[0]);
    }
  }
  // now backpropagate through first layer, but only if asked to (by out_err pointer)
  if (NULL != out_err) {
    nnet_[0]->Backpropagate(backpropagate_buf_[0], out_err);
  }

  //
  // End of Backpropagation
  //////////////////////////////////////
}


void Nnet::Feedforward(const Matrix<BaseFloat> &in, Matrix<BaseFloat> *out) {
  KALDI_ASSERT(NULL != out);

  if (LayerCount() == 0) { 
    out->Resize(in.NumRows(), in.NumCols(), kUndefined);
    out->CopyFromMat(in); 
    return; 
  }

  // we need at least 2 input buffers
  KALDI_ASSERT(propagate_buf_.size() >= 2);

  // propagate by using exactly 2 auxiliary buffers
  int32 L = 0;
  nnet_[L]->Propagate(in, &propagate_buf_[L%2]);
  for(L++; L<=LayerCount()-2; L++) {
    nnet_[L]->Propagate(propagate_buf_[(L-1)%2], &propagate_buf_[L%2]);
  }
  nnet_[L]->Propagate(propagate_buf_[(L-1)%2], out);
}


void Nnet::Read(std::istream &in, bool binary) {
  // get the network layers from a factory
  Component *comp;
  while (NULL != (comp = Component::Read(in, binary, this))) {
    if (LayerCount() > 0 && nnet_.back()->OutputDim() != comp->InputDim()) {
      KALDI_ERR << "Dimensionality mismatch!"
                << " Previous layer output:" << nnet_.back()->OutputDim()
                << " Current layer input:" << comp->InputDim();
    }
    nnet_.push_back(comp);
  }
  // create empty buffers
  propagate_buf_.resize(LayerCount()+1);
  backpropagate_buf_.resize(LayerCount()-1);
  // reset learn rate
  learn_rate_ = 0.0;
}


void Nnet::LearnRate(BaseFloat lrate, const char *lrate_factors) {
  // split lrate_factors to a vector
  std::vector<BaseFloat> lrate_factor_vec;
  if (NULL != lrate_factors) {
    char *copy = new char[strlen(lrate_factors)+1];
    strcpy(copy, lrate_factors);
    char *tok = NULL;
    while(NULL != (tok = strtok((tok==NULL?copy:NULL),",:; "))) {
      lrate_factor_vec.push_back(atof(tok));
    }
    delete copy;
  }

  // count trainable layers
  int32 updatable = 0;
  for(int i=0; i<LayerCount(); i++) {
    if (nnet_[i]->IsUpdatable()) updatable++;
  }
  // check number of factors
  if (lrate_factor_vec.size() > 0 && updatable != (int32)lrate_factor_vec.size()) {
    KALDI_ERR << "Mismatch between number of trainable layers " << updatable
              << " and learn rate factors " << lrate_factor_vec.size();
  }

  // set learn rates
  updatable=0;
  for(int32 i=0; i<LayerCount(); i++) {
    if (nnet_[i]->IsUpdatable()) {
      BaseFloat lrate_scaled = lrate;
      if (lrate_factor_vec.size() > 0) lrate_scaled *= lrate_factor_vec[updatable++];
      dynamic_cast<UpdatableComponent*>(nnet_[i])->LearnRate(lrate_scaled);
    }
  }
  // set global learn rate
  learn_rate_ = lrate;
}


std::string Nnet::LearnRateString() {
  std::ostringstream oss;
  oss << "LEARN_RATE global: " << learn_rate_ << " individual: ";
  for(int32 i=0; i<LayerCount(); i++) {
    if (nnet_[i]->IsUpdatable()) {
      oss << dynamic_cast<UpdatableComponent*>(nnet_[i])->LearnRate() << " ";
    }
  }
  return oss.str();
}




  
} // namespace
