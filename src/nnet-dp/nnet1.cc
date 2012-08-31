// nnet-dp/nnet1.cc

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

#include "nnet-dp/nnet1.h"
#include "thread/kaldi-thread.h"
#include "gmm/model-common.h" // for GetSplitTargets

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
        this_context_frames[0] < 0 || this_context_frames[1] < 0)
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
    int32 real_input_dim =
        input_dim * (1 + layer_info.left_context + layer_info.right_context) + 1;
    layer_info.tanh_layer = new TanhLayer(real_input_dim,
                                          output_dim,
                                          info.learning_rates[i],
                                          1.0 / sqrt(real_input_dim));
  }
  // Now the final layers.
  final_layers_.resize(info.category_sizes.size());
  int32 final_layer_input = info.layer_sizes.back() + 1; // input dim of
  // softmax layer.
  for (int32 category = 0; category < info.category_sizes.size();
       category++) {
    int32 category_size = info.category_sizes[category];
    final_layers_[category].softmax_layer = new SoftmaxLayer(
        final_layer_input, category_size, info.learning_rates[n]);
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
  KALDI_ASSERT(output->NumCols()-1 == num_splice * input_dim);
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
  // Set the last column of "output" to 1.0.
  {
    int32 nr = output->NumRows(), stride = output->Stride();
    BaseFloat *d = output->Data() + output->NumCols() - 1;
    // currently d points to last element of 1st row of "output".
    for (int32 r = 0; r < nr; r++, d += stride) *d = 1.0;
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
  // the + 1 in the line below is for the deriv w.r.t. the appended 1.0.
  // we ignore the value of this derivative.
  KALDI_ASSERT(spliced_deriv.NumCols() == num_splice * dim + 1);
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
      delete iter->tanh_layer;
    }
  }
  initial_layers_.clear();
  for (std::vector<FinalLayerInfo>::iterator iter = final_layers_.begin();
       iter != final_layers_.end(); ++iter) {
    if (iter->softmax_layer != NULL) {
      delete iter->softmax_layer;
    }
    if (iter->linear_layer != NULL) {
      delete iter->linear_layer;
    }
  }
  final_layers_.clear();
}

int32 Nnet1::LeftContext() const {
  int32 left_context = 0;
  for (int32 i = 0, end = NumTanhLayers(); i < end; i++) 
    left_context += LeftContextForLayer(i);
  return left_context;
}

int32 Nnet1::RightContext() const {
  int32 right_context = 0;
  for (int32 i = 0, end = NumTanhLayers(); i < end; i++) 
    right_context += RightContextForLayer(i);
  return right_context;
}

void Nnet1::Write(std::ostream &os, bool binary) const {
  int32 num_tanh_layers = NumTanhLayers();
  WriteToken(os, binary, "<NumTanhLayers>");
  WriteBasicType(os, binary, num_tanh_layers);

  // Category information may end up being derivative
  // of other information. If so, remove the following
  // parts about categories and their labels, and
  // derive from the rest of the information.
  int32 num_categories = NumCategories();
  WriteToken(os, binary, "<NumCategories>");
  WriteBasicType(os, binary, num_categories);

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
    KALDI_WARN << "Adding to a neural network that is already initialized";

  int32 num_tanh_layers = 0;
  ExpectToken(is, binary, "<NumTanhLayers>");
  ReadBasicType(is, binary, &num_tanh_layers);

  int32 num_categories = 0;
  ExpectToken(is, binary, "<NumCategories>");
  ReadBasicType(is, binary, &num_categories);

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
    layer_info.tanh_layer = new TanhLayer();
    layer_info.tanh_layer->Read(is, binary);
    initial_layers_.push_back(layer_info);
  }

  ExpectToken(is, binary, "<FinalLayers>");
  for (int32 i = 0; i < num_categories; i++){
    FinalLayerInfo layer_info;
    // see comment above about the way TanhLayer is constructed
    // using a constructor taking the input stream as argument.
    layer_info.softmax_layer = new SoftmaxLayer();
    layer_info.softmax_layer->Read(is, binary);
    layer_info.linear_layer = new LinearLayer();
    layer_info.linear_layer->Read(is, binary);
    final_layers_.push_back(layer_info);
  }
}

void Nnet1::SetZeroAndTreatAsGradient() {
  for (int32 i = 0; i < initial_layers_.size(); i++) {
    initial_layers_[i].tanh_layer->SetZero();
    initial_layers_[i].tanh_layer->SetLearningRate(1.0);
  }
  for (int32 i = 0; i < final_layers_.size(); i++) {
    final_layers_[i].softmax_layer->SetZero();
    final_layers_[i].softmax_layer->SetLearningRate(1.0);
    final_layers_[i].linear_layer->SetZero();
    final_layers_[i].linear_layer->SetLearningRate(1.0);
  }
}

Nnet1::Nnet1(const Nnet1 &other):
    initial_layers_(other.initial_layers_),
    final_layers_(other.final_layers_) {
  // This initialization just copied the pointers; now we want a deep copy,
  // so call new for each of the layers.
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

void Nnet1::ZeroOccupancy() {
  for (int32 i = 0; i < final_layers_.size(); i++)
    final_layers_[i].softmax_layer->ZeroOccupancy();
}

void Nnet1::MixUp(int32 target_tot_neurons,
                  BaseFloat power, // e.g. 0.2.
                  BaseFloat perturb_stddev) {
  int32 num_categories = final_layers_.size();
  Vector<BaseFloat> counts_by_category(num_categories);
  for (int32 category = 0; category < num_categories; category++) {
    counts_by_category(category) =
        final_layers_[category].softmax_layer->TotOccupancy();
  }
  KALDI_ASSERT(counts_by_category.Sum() > 0 &&
               "You cannot mix up before you have trained (because "
               "there is no occupation-count data.)");
  std::vector<int32> target_neurons(num_categories);
  BaseFloat min_count = 100.0; // This is kind of arbitrary; anyway
  // we expect this will never be active because there will
  // probably be much more data than this.  That's why we're not
  // making it configurable.
  GetSplitTargets(counts_by_category, target_tot_neurons,
                  power, min_count, &target_neurons);

  int32 old_tot_neurons = 0, new_tot_neurons = 0;
  for (int32 category = 0; category < num_categories; category++) {
    int32 old_num_neurons = final_layers_[category].softmax_layer->OutputDim(),
        new_num_neurons = std::max(old_num_neurons,
                                   target_neurons[category]);

    // Note: MixUpFinalLayers is a non-class-member function.
    if (new_num_neurons > old_num_neurons) 
      MixUpFinalLayers(new_num_neurons,
                       perturb_stddev,
                       final_layers_[category].softmax_layer,
                       final_layers_[category].linear_layer);
    
    old_tot_neurons += old_num_neurons;
    new_tot_neurons += new_num_neurons;
  }
  KALDI_LOG << "Mixed up from " << old_tot_neurons << " to "
            << new_tot_neurons;
}

void Nnet1::GetPriorsForCategory(int32 category,
                                 Vector<BaseFloat> *priors) const {
  KALDI_ASSERT(category >= 0 && category < final_layers_.size());
  const SoftmaxLayer &softmax_layer = *(final_layers_[category].softmax_layer);
  const LinearLayer &linear_layer = *(final_layers_[category].linear_layer);
  const Vector<BaseFloat> &occupancy = softmax_layer.Occupancy();
  const Matrix<BaseFloat> &linear_params = linear_layer.Params();
  priors->Resize(linear_params.NumRows());
  priors->AddMatVec(1.0, linear_params, kNoTrans, occupancy, 0.0);
  BaseFloat sum = priors->Sum();
  if (sum <= 0.0) {
    KALDI_WARN << "Total occupancy for category " << category << " is "
               << sum;
    priors->Set(1.0 / priors->Dim());
  } else {
    priors->Scale(1.0 / sum);
  }
}

void Nnet1::AdjustLearningRates(
    const Nnet1ProgressInfo &info, // this info is the dot prod of
    // the validation-set gradient *after* the step, with the delta
    // of parameters.
    BaseFloat ratio, // e.g. ratio = 1.1.
    BaseFloat max_lrate) { 
  KALDI_ASSERT(info.tanh_dot_prod.size() ==
               initial_layers_.size());
  BaseFloat inv_ratio = 1.0 / ratio;
  for (int32 i = 0; i < initial_layers_.size(); i++) {
    GenericLayer *layer = initial_layers_[i].tanh_layer;
    layer->SetLearningRateMax((info.tanh_dot_prod[i] > 0 ?
                               ratio : inv_ratio)
                              * layer->GetLearningRate(), max_lrate);
  }
  KALDI_ASSERT(info.softmax_dot_prod.size() ==
               final_layers_.size());
  for (int32 i = 0; i < final_layers_.size(); i++) {
    GenericLayer *layer = final_layers_[i].softmax_layer;
    layer->SetLearningRateMax((info.softmax_dot_prod[i] > 0 ?
                               ratio : inv_ratio)
                              * layer->GetLearningRate(), max_lrate);
    layer = final_layers_[i].linear_layer;
    layer->SetLearningRateMax((info.linear_dot_prod[i] > 0 ?
                               ratio : inv_ratio)
                              * layer->GetLearningRate(), max_lrate);
  }
}

void Nnet1::AdjustLearningRates(
    const Nnet1ProgressInfo &info, // this info is the dot prod of
    // the validation-set gradient *after* the step, with the delta
    // of parameters.
    const std::vector<std::vector<int32> > &final_layer_sets,
    BaseFloat ratio, // e.g. ratio = 1.1.
    BaseFloat max_lrate) {
  KALDI_ASSERT(info.tanh_dot_prod.size() ==
               initial_layers_.size());
  BaseFloat inv_ratio = 1.0 / ratio;
  for (int32 i = 0; i < initial_layers_.size(); i++) {
    GenericLayer *layer = initial_layers_[i].tanh_layer;
    layer->SetLearningRateMax((info.tanh_dot_prod[i] > 0 ?
                               ratio : inv_ratio)
                              * layer->GetLearningRate(), max_lrate);
  }
  KALDI_ASSERT(info.softmax_dot_prod.size() ==
               final_layers_.size());
  for (int32 i = 0; i < final_layer_sets.size(); i++) {
    BaseFloat total_dot_prod = 0.0;
    KALDI_ASSERT(final_layer_sets[i].size() > 0);
    for (int32 j = 0; j < final_layer_sets[i].size(); j++) {
      int32 idx = final_layer_sets[i][j];
      KALDI_ASSERT(idx >= 0 && idx < final_layers_.size());
      total_dot_prod += info.softmax_dot_prod[idx];
    }
    BaseFloat learning_rate_ratio = (total_dot_prod > 0 ?
                                     ratio : inv_ratio);
    for (int32 j = 0; j < final_layer_sets[i].size(); j++) {
      int32 idx = final_layer_sets[i][j];
      GenericLayer *layer = final_layers_[idx].softmax_layer;
      layer->SetLearningRateMax(learning_rate_ratio
                                * layer->GetLearningRate(), max_lrate);
    }
  }
  KALDI_ASSERT(info.linear_dot_prod.size() ==
               final_layers_.size());
  for (int32 i = 0; i < final_layer_sets.size(); i++) {
    BaseFloat total_dot_prod = 0.0;
    KALDI_ASSERT(final_layer_sets[i].size() > 0);
    for (int32 j = 0; j < final_layer_sets[i].size(); j++) {
      int32 idx = final_layer_sets[i][j];
      KALDI_ASSERT(idx >= 0 && idx < final_layers_.size());
      total_dot_prod += info.linear_dot_prod[idx];
    }
    BaseFloat learning_rate_ratio = (total_dot_prod > 0 ?
                                     ratio : inv_ratio);
    for (int32 j = 0; j < final_layer_sets[i].size(); j++) {
      int32 idx = final_layer_sets[i][j];
      GenericLayer *layer = final_layers_[idx].linear_layer;
      layer->SetLearningRateMax(learning_rate_ratio
                                * layer->GetLearningRate(), max_lrate);
    }
  }
}


void Nnet1::AddTanhLayer(int32 num_nodes,
                         int32 left_context,
                         int32 right_context,
                         BaseFloat learning_rate) {
  KALDI_ASSERT(!initial_layers_.empty());
  KALDI_ASSERT(learning_rate > 0 && left_context >= 0 && right_context >= 0);
  int32 output_size = (num_nodes > 0 ? num_nodes :
                       initial_layers_.back().tanh_layer->OutputDim());
  int32 input_size = (1 + left_context + right_context) *
      initial_layers_.back().tanh_layer->OutputDim() + 1; // + 1 for bias.
  InitialLayerInfo new_info;
  new_info.left_context = left_context;
  new_info.right_context = right_context;
  BaseFloat parameter_stddev = 0.0; // Initialize the parameters to zero.
  new_info.tanh_layer = new TanhLayer(input_size, output_size, learning_rate,
                                      parameter_stddev);
  initial_layers_.push_back(new_info);
}

std::string Nnet1::LrateInfo() const {
  std::ostringstream os;
  os << "tanh: ";
  for (int32 i = 0; i < initial_layers_.size(); i++)
    os << initial_layers_[i].tanh_layer->GetLearningRate() << ' ';
  os << ", softmax [avg]: ";
  double sum = 0.0;
  for (int32 i = 0; i < final_layers_.size(); i++)
    sum += final_layers_[i].softmax_layer->GetLearningRate();
  os << (sum / final_layers_.size()) << " ";
  sum = 0.0;
  for (int32 i = 0; i < final_layers_.size(); i++)
    sum += final_layers_[i].linear_layer->GetLearningRate();
  os << ", linear [avg]: " << (sum / final_layers_.size()) << " ";
  return os.str();
}

std::string Nnet1::LrateInfo(
    const std::vector<std::vector<int32> > &final_sets) const {
  std::ostringstream os;
  os << "tanh: ";
  for (int32 i = 0; i < initial_layers_.size(); i++)
    os << initial_layers_[i].tanh_layer->GetLearningRate() << ' ';
  os << ", softmax [per set]: ";
  for (int32 i = 0; i < final_sets.size(); i++) {
    double sum = 0.0;
    for (int32 j = 0; j < final_sets[i].size(); j++) {
      int32 idx = final_sets[i][j];
      KALDI_ASSERT(idx >= 0 && idx < final_layers_.size());
      sum += final_layers_[idx].softmax_layer->GetLearningRate();
    }
    double avg = sum / final_sets[i].size();
    os << avg << " ";
  }
  os << ", linear [per set]: ";
  for (int32 i = 0; i < final_sets.size(); i++) {
    double sum = 0.0;
   for (int32 j = 0; j < final_sets[i].size(); j++) {
      int32 idx = final_sets[i][j];
      KALDI_ASSERT(idx >= 0 && idx < final_layers_.size());
      sum += final_layers_[idx].linear_layer->GetLearningRate();
   }
    double avg = sum / final_sets[i].size();
    os << avg << " ";
  }
  return os.str();
}


std::string Nnet1::Info(
    const std::vector<std::vector<int32> > &final_layer_sets) const {
  std::ostringstream os;
  for (int32 i = 0; i < initial_layers_.size(); i++) {
    os << i << "'th tanh layer:\n";
    os << initial_layers_[i].tanh_layer->Info();
  }
  int32 tot_softmax_output_dim = 0,tot_linear_output_dim = 0;
  for (int32 i = 0; i < final_layers_.size(); i++) {
    tot_softmax_output_dim += final_layers_[i].softmax_layer->OutputDim();
    tot_linear_output_dim += final_layers_[i].linear_layer->OutputDim();    
  }
  os << "Total output dim of all softmax layers is "
     << tot_softmax_output_dim << "\n";
  os << "Total output dim of all linear layers is "
     << tot_linear_output_dim << "\n";
  
  // Now print out parameter stddevs and learning rates,
  // averaged over sets.

  for (int32 i = 0; i < final_layer_sets.size(); i++) {
    os << i << "'th set of softmax layers: ";
    BaseFloat sum_lrate = 0.0, sumsq_parameters = 0.0;
    int32 tot_parameters = 0;
    for (int32 j = 0; j < final_layer_sets[i].size(); j++) {
      int32 idx = final_layer_sets[i][j];
      SoftmaxLayer *layer = final_layers_[idx].softmax_layer;
      sum_lrate += layer->GetLearningRate();
      const Matrix<BaseFloat> &params = layer->Params();
      sumsq_parameters += pow(params.FrobeniusNorm(), 2.0);
      tot_parameters += params.NumRows() * params.NumCols();
    }
    BaseFloat avg_lrate = sum_lrate / final_layer_sets[i].size(),
        param_stddev = std::sqrt(sumsq_parameters / tot_parameters);
    os << "lrate=" << avg_lrate << ", param-stddev=" << param_stddev << std::endl;
  }
  
  for (int32 i = 0; i < final_layer_sets.size(); i++) {
    os << i << "'th set of linear layers: ";
    BaseFloat sum_lrate = 0.0, sumsq_parameters = 0.0;
    int32 tot_parameters = 0;
    for (int32 j = 0; j < final_layer_sets[i].size(); j++) {
      int32 idx = final_layer_sets[i][j];
      LinearLayer *layer = final_layers_[idx].linear_layer;
      sum_lrate += layer->GetLearningRate();
      const Matrix<BaseFloat> &params = layer->Params();
      sumsq_parameters += pow(params.FrobeniusNorm(), 2.0);
      tot_parameters += params.NumRows() * params.NumCols();
    }
    BaseFloat avg_lrate = sum_lrate / final_layer_sets[i].size(),
        param_stddev = std::sqrt(sumsq_parameters / tot_parameters);
    os << "lrate=" << avg_lrate << ", param-stddev=" << param_stddev << std::endl;
  }
  return os.str();
}




std::string Nnet1::Info() const {
  std::ostringstream os;
  for (int32 i = 0; i < initial_layers_.size(); i++) {
    os << i << "'th tanh layer:\n";
    os << initial_layers_[i].tanh_layer->Info();
  }
  int32 tot_softmax_output_dim = 0,tot_linear_output_dim = 0;
  for (int32 i = 0; i < final_layers_.size(); i++) {
    tot_softmax_output_dim += final_layers_[i].softmax_layer->OutputDim();
    tot_linear_output_dim += final_layers_[i].linear_layer->OutputDim();    
  }
  os << "Total output dim of all softmax layers is "
     << tot_softmax_output_dim << "\n";
  os << "Total output dim of all linear layers is "
     << tot_linear_output_dim << "\n";
  for (int32 i = 0; i < final_layers_.size(); i++) {
    os << i << "'th softmax layer\n";
    os << final_layers_[i].softmax_layer->Info();
    os << "[Occupancy of " << i << "'th softmax layer is "
       << final_layers_[i].softmax_layer->Occupancy().Sum() << " ]\n";
    os << i << "'th linear layer\n";
    os << final_layers_[i].linear_layer->Info();
  }
  return os.str();
}

void Nnet1::ComputeProgressInfo(
    const Nnet1 &previous_value,
    const Nnet1 &valid_gradient,
    Nnet1ProgressInfo *info) const {
  const Nnet1 &current_value = *this;
  info->tanh_dot_prod.resize(initial_layers_.size());
  for (int32 layer = 0; layer < initial_layers_.size(); layer++) {
    info->tanh_dot_prod[layer] =
        TraceMatMat(current_value.initial_layers_[layer].tanh_layer->Params(),
                    valid_gradient.initial_layers_[layer].tanh_layer->Params(),
                    kTrans) -
        TraceMatMat(previous_value.initial_layers_[layer].tanh_layer->Params(),
                    valid_gradient.initial_layers_[layer].tanh_layer->Params(),
                    kTrans);
  }
  info->softmax_dot_prod.resize(final_layers_.size());
  info->linear_dot_prod.resize(final_layers_.size());  
  for (int32 layer = 0; layer < final_layers_.size(); layer++) {
    info->softmax_dot_prod[layer] =
        TraceMatMat(current_value.final_layers_[layer].softmax_layer->Params(),
                    valid_gradient.final_layers_[layer].softmax_layer->Params(),
                    kTrans) -
        TraceMatMat(previous_value.final_layers_[layer].softmax_layer->Params(),
                    valid_gradient.final_layers_[layer].softmax_layer->Params(),
                    kTrans);
    info->linear_dot_prod[layer] =
        TraceMatMat(current_value.final_layers_[layer].linear_layer->Params(),
                    valid_gradient.final_layers_[layer].linear_layer->Params(),
                    kTrans) -
        TraceMatMat(previous_value.final_layers_[layer].linear_layer->Params(),
                    valid_gradient.final_layers_[layer].linear_layer->Params(),
                    kTrans);
  }
}

void UpdateProgressStats(const Nnet1ProgressInfo &progress_at_start,
                         const Nnet1ProgressInfo &progress_at_end,
                         Nnet1ProgressInfo *stats) {
  // make sure stats has correct size.
  stats->tanh_dot_prod.resize(progress_at_start.tanh_dot_prod.size(), 0.0);
  stats->softmax_dot_prod.resize(progress_at_start.softmax_dot_prod.size(), 0.0);
  stats->linear_dot_prod.resize(progress_at_start.linear_dot_prod.size(), 0.0);
  
  // Now update stats->  progress_at_start mean respectively:
  // (parameter-delta) . validation_gradient_before_change
  // and
  // (parameter-delta) . validation_gradient_after_change
  // If we have a quadratic model of the objective function, the
  // increase in the objective function attributed to each of the layers
  // is just 0.5 * (progress_at_start + progress_at_end).
  for (int32 i = 0; i < stats->tanh_dot_prod.size(); i++) {
    stats->tanh_dot_prod[i] += 0.5 * (progress_at_start.tanh_dot_prod[i] +
                                      progress_at_end.tanh_dot_prod[i]);
  }
  for (int32 i = 0; i < stats->softmax_dot_prod.size(); i++) {
    stats->softmax_dot_prod[i] += 0.5 * (progress_at_start.softmax_dot_prod[i] +
                                         progress_at_end.softmax_dot_prod[i]);
    stats->linear_dot_prod[i] += 0.5 * (progress_at_start.linear_dot_prod[i] +
                                        progress_at_end.linear_dot_prod[i]);
  }
}

std::string Nnet1ProgressInfo::Info() const {
  BaseFloat total = 0.0;;
  std::ostringstream os;
  os << "tanh: ";
  for (int32 i = 0; i < tanh_dot_prod.size(); i++) {
    os << tanh_dot_prod[i] << ' ';
    total += tanh_dot_prod[i];
  }
  os << '\n';
  os << "softmax: ";
  for (int32 i = 0; i < softmax_dot_prod.size(); i++) {
    os << softmax_dot_prod[i] << ' ';
    total += softmax_dot_prod[i];
  }
  os << '\n';
  os << "linear: ";
  for (int32 i = 0; i < linear_dot_prod.size(); i++) {
    os << linear_dot_prod[i] << ' ';
    total += linear_dot_prod[i];
  }
  os << '\n' << "total: " << total << '\n';
  return os.str();
}

std::string Nnet1ProgressInfo::Info(
    const std::vector<std::vector<int32> > &final_layer_sets) const {
  BaseFloat total = 0.0;;
  std::ostringstream os;
  os << "tanh: ";
  for (int32 i = 0; i < tanh_dot_prod.size(); i++) {
    os << tanh_dot_prod[i] << ' ';
    total += tanh_dot_prod[i];
  }
  os << '\n';
  os << "softmax (per set): ";
  for (int32 i = 0; i < final_layer_sets.size(); i++) {
    BaseFloat this_tot = 0.0;
    for (int32 j = 0; j < final_layer_sets[i].size(); j++) {
      int32 idx = final_layer_sets[i][j];
      KALDI_ASSERT(idx >= 0 && idx < linear_dot_prod.size());
      this_tot += softmax_dot_prod[idx];
    }
    os << this_tot << ' ';
    total += this_tot;
  }
  os << '\n';
  os << "linear (per set): ";
  for (int32 i = 0; i < final_layer_sets.size(); i++) {
    BaseFloat this_tot = 0.0;
    for (int32 j = 0; j < final_layer_sets[i].size(); j++) {
      int32 idx = final_layer_sets[i][j];
      KALDI_ASSERT(idx >= 0 && idx < linear_dot_prod.size());
      this_tot += linear_dot_prod[idx];
    }
    os << this_tot << ' ';
    total += this_tot;
  }
  os << '\n' << "total: " << total << '\n';
  return os.str();
}


} // namespace kaldi
