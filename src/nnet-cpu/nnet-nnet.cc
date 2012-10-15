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

void Nnet::SetZero(bool treat_as_gradient) {
  for (size_t i = 0; i < components_.size(); i++) {
    UpdatableComponent *uc = dynamic_cast<UpdatableComponent*>(components_[i]);
    if (uc != NULL) uc->SetZero(treat_as_gradient);
  }
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


void Nnet::ZeroOccupancy() {
  for (size_t i = 0; i < components_.size(); i++) {
    SoftmaxComponent *softmax_component =
        dynamic_cast<SoftmaxComponent*>(components_[i]);
    if (softmax_component != NULL) { // If it was this type
      softmax_component->ZeroOccupancy();
    }
  }
}
void Nnet::Destroy() {
  while (!components_.empty()) {
    delete components_.back();
    components_.pop_back();
  }
}

void Nnet::ComponentDotProducts(
    const Nnet &other,
    VectorBase<BaseFloat> *dot_prod) const {
  KALDI_ASSERT(dot_prod->Dim() == NumComponents());
  for (size_t i = 0; i < components_.size(); i++) {
    UpdatableComponent *uc1 = dynamic_cast<UpdatableComponent*>(components_[i]);
    const UpdatableComponent *uc2 = dynamic_cast<const UpdatableComponent*>(
        &(other.GetComponent(i)));
    KALDI_ASSERT((uc1 != NULL) == (uc2 != NULL));
    if (uc1 != NULL)
      (*dot_prod)(i) = uc1->DotProduct(*uc2);
    else
      (*dot_prod)(i) = 0.0;
  }    
}


Nnet::Nnet(const Nnet &other): components_(other.components_.size()) {
  for (size_t i = 0; i < other.components_.size(); i++)
    components_[i] = other.components_[i]->Copy();
}


Nnet &Nnet::operator = (const Nnet &other) {
  Destroy();
  components_.resize(other.components_.size());
  for (size_t i = 0; i < other.components_.size(); i++)
    components_[i] = other.components_[i]->Copy();
  return *this;
}

void Nnet::Check() const {
  KALDI_ASSERT(!components_.empty());
  for (size_t i = 0; i + 1 < components_.size(); i++) {
    int32 output_dim = components_[i]->OutputDim(),
      next_input_dim = components_[i+1]->InputDim();
    KALDI_ASSERT(output_dim == next_input_dim);
  }
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
