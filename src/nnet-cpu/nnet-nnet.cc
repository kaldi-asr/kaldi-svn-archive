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
#include "util/stl-utils.h"

namespace kaldi {

const std::vector<int32> &Nnet::RawSplicingForComponent(int32 component) const {
  KALDI_ASSERT(static_cast<size_t>(component) < raw_splicing_.size());
  return  raw_splicing_[component];
}

const std::vector<int32> &Nnet::FullSplicingForComponent(int32 component) const {
  KALDI_ASSERT(static_cast<size_t>(component) < full_splicing_.size());
  return  full_splicing_[component];
}

const std::vector<std::vector<int32> > &Nnet::RelativeSplicingForComponent(
    int32 component) const {
  KALDI_ASSERT(static_cast<size_t>(component) < relative_splicing_.size());
  return  relative_splicing_[component];
}


int32 Nnet::OutputDim() const {
  KALDI_ASSERT(!components_.empty());
  return components_.back()->OutputDim();
}

int32 Nnet::FeatureDim() const {
  KALDI_ASSERT(!components_.empty());
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
        for (int32 k = 0; k < full_splicing_[c].size(); k++)
          if (full_splicing_[c][k] == frame_offset)
            index = k;
        KALDI_ASSERT(index != -1);
        relative_splicing_[c][i][j] = index;
        check.push_back(index);
      }
    }
    SortAndUniq(&check);
    KALDI_ASSERT(check.size() == full_splicing_[c].size()
                  && check.front() == 0 && check.back() + 1 == check.size());
    // Make sure we used everything, otherwise something's wrong.
  }
}


void Nnet::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<Nnet>");
  int32 num_components = components_.size();
  WriteToken(os, binary, "<NumComponents>");
  WriteBasicType(os, binary, num_components);
  WriteToken(os, binary, "<SpeakerInfoDim>");
  WriteBasicType(os, binary, speaker_info_dim_);
  WriteToken(os, binary, "<Components>");
  for (int32 c = 0; c < num_components; c++) 
    components_[c]->Write(os, binary);
  WriteToken(os, binary, "</Components>");
  WriteToken(os, binary, "</Nnet>");  
}

void Nnet::Read(std::istream &is, bool binary) {
  Destroy();
  ExpectToken(is, binary, "<Nnet>");
  int32 num_components;
  ExpectToken(is, binary, "<NumComponents>");
  ReadBasicType(is, binary, &num_components);
  ExpectToken(is, binary, "<SpeakerInfoDim>");
  ReadBasicType(is, binary, &speaker_info_dim_);
  ExpectToken(is, binary, "<Components>");
  components_.resize(num_components);
  for (int32 c = 0; c < num_components; c++) 
    components_[c]->Read(is, binary);
  ExpectToken(is, binary, "</Components>");
  ExpectToken(is, binary, "</Nnet>");  
}

void Nnet::Destroy() {
  while (!components_.empty()) {
    delete components_.back();
    components_.pop_back();
  }
}
  
} // namespace
