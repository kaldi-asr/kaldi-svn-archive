// nnet_dp/nnet1.cc

// Copyright 2012  Johns Hopkins University (author:  Daniel Povey)
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

#include "nnet_dp/nnet1.h"
#include "thread/kaldi-thread.h"

namespace kaldi {


Nnet1InitInfo::Nnet1InitInfo(const Nnet1InitConfig &config,
                             const std::vector<int32> &category_sizes_in) {
  diagonal_element = config.diagonal_element;
  SplitStringToIntegers(config.layer_sizes, ":", false, &layer_sizes);
  if (layer_sizes.size() < 2 ||
      *std::min_element(layer_sizes.begin(), layer_sizes.end()) < 1)
    KALDI_ERR << "Invalid option --layer-sizes="
              << config.layer_sizes;
  std::vector<std::string> context_frames_vec;
  SplitStringToVector(config.context_frames, ":", false, &context_frames_vec);
  context_frames.clear();
  for (size_t i = 0; i < context_frames_vec.size(); i++) {
    std::vector<int32> this_context_frames; // vector of size 2.
    SplitStringToIntegers(context_frames_vec[i], ",", false,
                          &this_context_frames);
    if (this_context_frames.size() != 2 ||
        this_context_frames[0] < 0 || this_context_frames[1] > 0)
      KALDI_ERR << "Invalid option --context-frames=" << config.context_frames;
    context_frames.push_back(std::make_pair(this_context_frames[0],
                                            this_context_frames[1]));
  }
  SplitStringToFloats(config.learning_rates, ":", false,
                      &learning_rates);
  int32 n = context_frames.size(); // Number of tanh layers.
  if (layer_sizes.size() != n+1)
    KALDI_ERR << "Inconsistent layer sizes: from --context-frames, number of tanh "
        "layers seems to be " << n << ", expecting " << (n+1) << " entries in "
        "--layer-sizes option but got --layer-sizes=" << config.layer_sizes;
  if (learning_rates.size() == 1)
    learning_rates.resize(n+2, learning_rates[0]); // all the same.
  if (learning_rates.size() != n+2)
    KALDI_ERR << "Inconsistent layer sizes: from --context-frames, number of tanh "
        "layers seems to be " << n << ", expecting 1 or " << (n+2) << " entries "
        "in --learning-rates option but got --learning-rates="
              << config.learning_rates;
  category_sizes = category_sizes_in;
  KALDI_ASSERT(category_sizes.size() > 0 &&
               *std::min_element(category_sizes.begin(),
                                 category_sizes.end()) > 0);
}


void Nnet1::Init(const Nnet1InitInfo &info) {
  Destroy(); // In case there was anything there.
  int32 n = info.context_frames.size(); // Number of tanh layers.
  initial_layers_.resize(n);
  for (int32 i = 0; i < n; i++) {
    InitialLayerInfo &layer_info = initial_layers_[i];
    int32 input_dim = info.layer_sizes[i],
        output_dim = info.layer_sizes[i+1];
    layer_info.left_context = info.context_frames[i].first;
    layer_info.right_context = info.context_frames[i].second;
    int32 tot_context = 1 + layer_info.left_context + layer_info.right_context;
    layer_info.tanh_layer = new TanhLayer(input_dim * tot_context,
                                          output_dim,
                                          info.learning_rates[i]);
  }
  // Now the final layers.
  final_layers_.resize(info.category_sizes.size());
  int32 final_layer_size = info.layer_sizes.back();
  for (int32 category = 0; category < info.category_sizes.size();
       category++) {
    int32 category_size = info.category_sizes[category];
    final_layers_[category].softmax_layer = new SoftmaxLayer(
        final_layer_size, category_size, info.learning_rates[n]);
    final_layers_[category].linear_layer = new LinearLayer(
        category_size, info.diagonal_element, info.learning_rates[n]);
  }
}


void SpliceFrames(const MatrixBase<BaseFloat> &input,
                  int32 num_chunks,
                  MatrixBase<BaseFloat> *output) { // splice together
  // the input rows into the output.  The #frames we splice is worked
  // out from the input+output dims and the num_chunks.  
  
  // Check input is of size [chunk_size_input * num_chunks] by [dim]
  int32 input_dim = input.NumCols();
  KALDI_ASSERT(input.NumRows() % num_chunks == 0 &&
               output->NumRows() % num_chunks == 0);
  int32 chunk_size_input = input.NumRows() / num_chunks,
      chunk_size_output = output->NumRows() / num_chunks;
  int32 num_splice = 1 + chunk_size_input - chunk_size_output; // # of frames
  // we splice together each time
  KALDI_ASSERT(num_splice > 0);
  KALDI_ASSERT(output->NumCols() == num_splice * input_dim);
  for (int32 c = 0; c < num_chunks; c++) {
    for (int32 s = 0; s < num_splice; s++) {
      SubMatrix<BaseFloat> input_chunk(input,
                                       c*chunk_size_input + s, chunk_size_output,
                                       0, input_dim),
          output_chunk(*output,
                       c*chunk_size_output, chunk_size_output,
                       s * input_dim, input_dim);
      output_chunk.CopyFromMat(input_chunk);
    }
  }
}

// This does almost the opposite of SpliceFrames; it's used in the backward
// pass.  Where SpliceFrames would copy a chunk A to destinations B, C and D,
// this function set A <-- B + C + D.  [this is appropriate for the derivatives
// we're computing in the backward pass].

void UnSpliceDerivative(const MatrixBase<BaseFloat> &spliced_deriv,
                        int32 num_chunks,
                        MatrixBase<BaseFloat> *deriv) { // splice together
  // the input rows into the output.  The #frames we splice is worked
  // out from the input+output dims and the num_chunks.

  int32 dim = deriv->NumCols(); // basic dimension before splicing
  KALDI_ASSERT(spliced_deriv.NumRows() % num_chunks == 0 &&
               deriv->NumRows() % num_chunks == 0);
  int32 chunk_size_unspliced = deriv->NumRows() / num_chunks,
      chunk_size_spliced = spliced_deriv.NumRows() / num_chunks;
  int32 num_splice = 1 + chunk_size_unspliced - chunk_size_spliced; // # of frames
  // we splice together each time
  KALDI_ASSERT(num_splice > 0);
  KALDI_ASSERT(spliced_deriv.NumCols() == num_splice * dim);
  deriv->Set(0.0);
  for (int32 c = 0; c < num_chunks; c++) {
    for (int32 s = 0; s < num_splice; s++) {
      SubMatrix<BaseFloat> spliced_chunk(spliced_deriv,
                                         c * chunk_size_spliced, chunk_size_spliced,
                                         s * dim, dim),
          unspliced_chunk(*deriv,
                          c*chunk_size_unspliced + s, chunk_size_spliced,
                          0, dim);
      if (s == 0)
        unspliced_chunk.CopyFromMat(spliced_chunk); // slightly faster.
      else
        unspliced_chunk.AddMat(1.0, spliced_chunk);
    }
  }
}

void Nnet1::Destroy() {
  for (std::vector<InitialLayerInfo>::iterator iter = initial_layers_.begin();
       iter != initial_layers_.end(); ++iter) {
    if (iter->tanh_layer != NULL) {
      delete iter->tanh_layer ; 
    }
  }
  initial_layers_.clear() ; 
  for (std::vector<FinalLayerInfo>::iterator iter = final_layers_.begin();
       iter != final_layers_.end(); ++iter) {
    if (iter->softmax_layer != NULL) {
      delete iter->softmax_layer ; 
    }
    if (iter->linear_layer != NULL) {
      delete iter->linear_layer ; 
    }
  }
  final_layers_.clear() ; 
}

int32 Nnet1::LeftContext() const {
  int32 left_context = 0;
  for (int32 i = 0, end = NumTanhLayers(); i < end; i++) 
    left_context += LeftContextForLayer(i);
  return left_context ; 
}

int32 Nnet1::RightContext() const {
  int32 right_context = 0;
  for (int32 i = 0, end = NumTanhLayers(); i < end; i++) 
    right_context += RightContextForLayer(i);
  return right_context ; 
}

void Nnet1::Write(std::ostream &os, bool binary) const {
  int32 num_tanh_layers = NumTanhLayers() ; 
  if (num_tanh_layers == 0) {
    KALDI_WARN << "Trying to write empty Nnet1 object.";
  }
  WriteToken(os, binary, "<NumTanhLayers>");
  WriteBasicType(os, binary, num_tanh_layers);

  // Category information may end up being derivative
  // of other information. If so, remove the following
  // parts about categories and their labels, and
  // derive from the rest of the information.
  int32 num_categories = NumCategories() ; 
  if (num_categories == 0) {
    KALDI_WARN << "NumCategories is 0." ;
  }
  WriteToken(os, binary, "<NumCategories>");
  WriteBasicType(os, binary, num_categories);

  WriteToken(os, binary, "<NumLabelsForCategories>");
  for (int i = 0; i < num_categories; i++)
    WriteBasicType(os, binary, NumLabelsForCategory(i)) ;

  WriteToken(os, binary, "<InitialLayers>");
  // InitialLayerInfo is a small enough struct that we 
  // are not adding a Read/Write functionality to it.
  for (std::vector<InitialLayerInfo>::const_iterator 
        it = initial_layers_.begin(), end = initial_layers_.end(); 
        it != end; ++it) {
    WriteToken(os, binary, "<left_context>");
    WriteBasicType(os, binary, it->left_context);
    WriteToken(os, binary, "<right_context>");
    WriteBasicType(os, binary, it->right_context);
    it->tanh_layer->Write(os, binary);
  }

  WriteToken(os, binary, "<FinalLayers>");
  // FinalLayerInfo is a small enough struct that we 
  // are not adding a Read/Write functionality to it.
  for (std::vector<FinalLayerInfo>::const_iterator 
        it = final_layers_.begin(), end = final_layers_.end(); 
        it != end; ++it) {
    it->softmax_layer->Write(os, binary);
    it->linear_layer->Write(os, binary);
  }
}

void Nnet1::Read(std::istream &is, bool binary) {
  if (NumTanhLayers() != 0)
    KALDI_WARN << "Adding to a neural network that is already initialized" ;

  int32 num_tanh_layers = 0;
  ExpectToken(is, binary, "<NumTanhLayers>");
  ReadBasicType(is, binary, &num_tanh_layers);
  if (num_tanh_layers == 0) {
    KALDI_WARN << "Read NumLayers = 0";
  }

  // Category information may end up being derivative
  // of other information. If so, remove the following
  // parts about categories and their labels, and
  // derive from the rest of the information.
  int32 num_categories = 0;
  ExpectToken(is, binary, "<NumCategories>");
  ReadBasicType(is, binary, &num_categories);
  KALDI_WARN << "NumCategories read but not yet used!";
  if (num_categories == 0) {
    KALDI_WARN << "Read NumCategories = 0." ;
  }

  ExpectToken(is, binary, "<NumLabelsForCategories>");
  for (int32 i = 0; i < num_categories; i++) {
    int32 num_categ_lay = 0;
    ReadBasicType(is, binary, &num_categ_lay);
    KALDI_WARN << "NumLabels for category read by not yet used!";
    if (num_categ_lay == 0) {
      KALDI_WARN << "NumLabels for category " << i << " is zero.";
    }
  }

  ExpectToken(is, binary, "<InitialLayers>");
  // InitialLayerInfo is a small enough struct that we 
  // are not adding a Read/Write functionality to it.
  for (int32 i = 0; i < num_tanh_layers; i++) {
    InitialLayerInfo layer_info;
    ExpectToken(is, binary, "<left_context>");
    ReadBasicType(is, binary, &layer_info.left_context);
    ExpectToken(is, binary, "<right_context>");
    ReadBasicType(is, binary, &layer_info.right_context);
    // Dan, it is possible that for consistency's sake,
    // you want to use a call to TanhLayer->Read here. 
    // I did it this way, to avoid using the default 
    // constructor.
    layer_info.tanh_layer = new TanhLayer(is, binary);
    initial_layers_.push_back(layer_info);
  }

  ExpectToken(is, binary, "<FinalLayers>");
  for (int32 i = 0; i < num_categories; i++){
    FinalLayerInfo layer_info;
    // see comment above about the way TanhLayer is constructed
    // using a constructor taking the input stream as argument.
    layer_info.softmax_layer = new SoftmaxLayer(is, binary);
    layer_info.linear_layer = new LinearLayer(is, binary);
    final_layers_.push_back(layer_info);
  }
}

void Nnet1::SetZero() {
  for (int32 i = 0; i < initial_layers_.size(); i++)
    initial_layers_[i].tanh_layer->SetZero();
  for (int32 i = 0; i < final_layers_.size(); i++) {
    final_layers_[i].softmax_layer->SetZero();
    final_layers_[i].linear_layer->SetZero();
  }
}

Nnet1::Nnet1(const Nnet1 &other):
    initial_layers_(other.initial_layers_),
    final_layers_(other.final_layers_) {
  // This initialization just copied the pointers; now we want a deep copy,
  // so call New().
  for (int32 i = 0; i < initial_layers_.size(); i++)
    initial_layers_[i].tanh_layer =
        new TanhLayer(*initial_layers_[i].tanh_layer);
  for (int32 i = 0; i < final_layers_.size(); i++) {
    final_layers_[i].softmax_layer =
        new SoftmaxLayer(*final_layers_[i].softmax_layer);
    final_layers_[i].linear_layer =
        new LinearLayer(*final_layers_[i].linear_layer);
  }
}

void Nnet1::AdjustLearningRates(
    const Nnet1 &previous_value,
    const Nnet1 &validation_gradient,
    BaseFloat learning_rate_ratio) {
  for (int32 i = 0; i < initial_layers_.size(); i++) {
    initial_layers_[i].tanh_layer->AdjustLearningRate(
        *previous_value.initial_layers_[i].tanh_layer,
        *validation_gradient.initial_layers_[i].tanh_layer,
        learning_rate_ratio);
  }
  for (int32 i = 0; i < final_layers_.size(); i++) {
    final_layers_[i].softmax_layer->AdjustLearningRate(
        *previous_value.final_layers_[i].softmax_layer,
        *validation_gradient.final_layers_[i].softmax_layer,
        learning_rate_ratio);
    final_layers_[i].linear_layer->AdjustLearningRate(
        *previous_value.final_layers_[i].linear_layer,
        *validation_gradient.final_layers_[i].linear_layer,
        learning_rate_ratio);
  }
}
  


} // namespace kaldi
