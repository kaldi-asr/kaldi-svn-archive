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


int32 Nnet::OutputDim() const {
  KALDI_ASSERT(!components_.empty());
  return components_.back()->OutputDim();
}

int32 Nnet::LeftContext() const {
  KALDI_ASSERT(!components_.empty());
  int32 ans = 0;
  for (size_t i = 0; i < components_.size(); i++)
    ans += components_[i]->LeftContext();
  return ans;
}

int32 Nnet::RightContext() const {
  KALDI_ASSERT(!components_.empty());
  int32 ans = 0;
  for (size_t i = 0; i < components_.size(); i++)
    ans += components_[i]->RightContext();
  return ans;
}

const Component& Nnet::GetComponent(int32 component) const {
  KALDI_ASSERT(static_cast<size_t>(component) < components_.size());
  return *(components_[component]);
}

Component& Nnet::GetComponent(int32 component) {
  KALDI_ASSERT(static_cast<size_t>(component) < components_.size());
  return *(components_[component]);
}


void Nnet::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<Nnet>");
  int32 num_components = components_.size();
  WriteToken(os, binary, "<NumComponents>");
  WriteBasicType(os, binary, num_components);
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

Nnet::Nnet(const Nnet &other): components_(other.components_.size()) {
  for (size_t i = 0; i < other.components_.size(); i++)
    components_[i] = other.components_[i]->Copy();
}

void Nnet::Check() const {
  // TODO.  Make sure the OutputDim() of each layer is the same as
  // the InputDim() of the next.
}

void Nnet::InitFromConfig(std::istream &is) {
  Destroy();
  std::string line;
  /* example config file as follows.  The things in brackets specify the context
     splicing for each layer, and after that is the info about the actual layer.
     Imagine the input dim is 13, and the speaker dim is 40, so (13 x 9) + 40 =  527.
     The config file might be as follows; the lines beginning with # are comments.
     
     # layer-type layer-options
     AffineLayer 0.01 0.001 527 1000 0.04356
  */
  components_.clear();
  while (getline(is, line)) {
    std::istringstream line_is(line);
    line_is >> std::ws; // Eat up whitespace.
    if (line_is.peek() == '#' || line_is.eof()) continue; // Comment or empty.
    Component *c = Component::NewFromString(line);
    KALDI_ASSERT(c != NULL);
    components_.push_back(c);    
  }
  Check();
}
  
} // namespace
