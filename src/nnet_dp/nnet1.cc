// nnet_dp/nnet1.cc

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

#include "nnet_dp/nnet1.h"
#include "thread/kaldi-thread.h"

namespace kaldi {


Nnet1InitInfo::Nnet1InitInfo(const Nnet1InitConfig &config,
                             const std::vector<int32> &category_sizes_in) {
  SplitStringToIntegers(config.layer_sizes, ":", false, &layer_sizes);
  if (layer_sizes.size() < 2 ||
      *std::min_element(layer_sizes.begin(), layer_sizes.end()) < 1)
    KALDI_ERR << "Invalid option --layer-sizes="
              << config.layer_sizes;
  std::vector<std::string> context_frames_vec;
  SplitStringToVector(config.context_frames, ":", &context_frames_vec, false);
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
  // category_sizes gives number of
  // labels in each category.  For single-language, will be [# first-level nodes in tree],
  // then for each first-level node that has >1 leaf, the # second-level nodes for that
  // first-level node.

  
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

Nnet1::Nnet1(const Nnet1 &other){
  int32 num_tanh_layers = other.NumTanhLayers();
  for (int32 layer = 0; layer < num_tanh_layers; layer++) {
    InitialLayerInfo layer_info ;
    layer_info.left_context = other.initial_layers_[layer].left_context;
    layer_info.right_context = other.initial_layers_[layer].right_context;
    // using default copy constructor here for tanh_layer. 
    layer_info.tanh_layer = new TanhLayer(
                          *other.initial_layers_[layer].tanh_layer);
    initial_layers_.push_back(layer_info);
  }

  int32 num_categories = other.NumCategories(); 
  for (int32 categ = 0; categ < num_categories; categ++) {
    FinalLayerInfo categ_info ;
    // using default copy constructor here for tanh_categ. 
    categ_info.softmax_layer = new SoftmaxLayer(
                          *other.final_layers_[categ].softmax_layer);
    categ_info.linear_layer = new LinearLayer(
                          *other.final_layers_[categ].linear_layer);
    final_layers_.push_back(categ_info);
  }
}
 
void Nnet1Trainer::TrainStep(const std::vector<TrainingExample> &data) {
  FormatInput(data);
  ForwardTanh();
  ForwardAndBackwardFinal(data);
  BackwardTanh();
}

void Nnet1Trainer::ForwardTanh() {
  // Does the forward computation for the initial (tanh) layers.
  for (int32 layer = 0; layer < nnet_.initial_layers_.size(); layer++) {

    Matrix<BaseFloat> temp_input; 
    if (nnet_.LayerIsSpliced(layer)) {
      // Splice together the data appropriately, if needed...
      temp_input.Resize(tanh_forward_data_[layer+1].NumRows(),
                        nnet_.initial_layers_[layer].tanh_layer->InputDim(),
                        kUndefined);
      SpliceFrames(tanh_forward_data_[layer], num_chunks_,
                   &temp_input);
    }
    Matrix<BaseFloat> &this_input = (nnet_.LayerIsSpliced(layer) ? temp_input :
                                     tanh_forward_data_[layer]);
    nnet_.initial_layers_[layer].tanh_layer->Forward(this_input,
                                                     &(tanh_forward_data_[layer+1]));
  }  
}

void Nnet1Trainer::ListCategories(
    const std::vector<TrainingExample> &data,
    std::vector<int32> *common_categories,
    std::vector<int32> *other_categories) {
  unordered_map<int32, int32> category_count;
  int32 num_frames = 0;
  for (int32 chunk = 0; chunk < data.size(); chunk++) {
    for (int32 t = 0; t < data[chunk].labels.size(); t++) {
      num_frames++;
      std::vector<int32> check_vec;
      for (int32 idx = 0; idx < data[chunk].labels[t].size(); idx++) { // iterate
        // over the list..
        int32 category = data[chunk].labels[t][idx].first;
        check_vec.push_back(category);
        if (category_count.find(category) == category_count.end())
          category_count[category] = 1;
        else
          category_count[category]++;
      }
      SortAndUniq(&check_vec);
      KALDI_ASSERT(check_vec.size() == data[chunk].labels[t].size()); // If this
      // assert fails, it means a category was repeated in the labels.  This
      // doesn't make sense.
    }
  }
  common_categories->clear();
  other_categories->clear();
  std::vector<std::pair<int32, int32> > frequency_list; 
  for (unordered_map<int32, int32>::iterator iter = category_count.begin();
       iter != category_count.end(); ++iter) {
    int32 category = iter->first, count = iter->second;
    if (count == num_frames) { // seen on every frame -> add to "common_categories".
      common_categories->push_back(category);
    } else {
      frequency_list.push_back(std::pair<int32, int32>(count, category));
    }
  }
  std::sort(frequency_list.begin(), frequency_list.end()); // sorts in increasing
  // order of count.
  for (size_t i = 0; i < frequency_list.size(); i++) {
    int32 category = frequency_list[i].second;
    other_categories->push_back(category);
  }
}

class Nnet1Trainer::ForwardAndBackwardFinalClass {
 public:
  ForwardAndBackwardFinalClass(Nnet1Trainer &nnet_trainer,
                               const std::vector<TrainingExample> &data,
                               Mutex *mutex,
                               std::vector<int32> *category_list,
                               double *tot_like_ptr):
      nnet_trainer_(nnet_trainer), data_(data), mutex_(mutex),
      category_list_(category_list), tot_like_(0), tot_like_ptr_(tot_like_ptr)
  { }
  
  void operator () () { // does the job of the class...
    while (1) { // while there are remaining categories in category_list_ to process,
      // process them.
      mutex_->Lock(); // Lock the mutex prior to modifying category_list_.
      int32 category = -1;
      if (!category_list_->empty()) {
        category = category_list_->back();
        category_list_->pop_back();
      }
      mutex_->Unlock();
      if (category == -1) return; // No more categories to process;
      // all threads will return once they see this.
      tot_like_ +=
          nnet_trainer_.ForwardAndBackwardFinalForCategory(data_, category, false);
    }   
  }

  ~ForwardAndBackwardFinalClass() {
    // The destructor is responsible for collecting the "tot_like" quantities.
    KALDI_ASSERT(tot_like_ptr_ != NULL);
    *tot_like_ptr_ += tot_like_;
  }
  
  // This function should be provided. Give it this exact implementation, with
  // the class name replaced with your own class's name.
  static void *run(void *c_in) {
    ForwardAndBackwardFinalClass *c = static_cast<ForwardAndBackwardFinalClass*>(c_in);
    (*c)(); // call operator () on it.
    return NULL;
  }  


  // The following class members are not actually needed, as the threads don't
  // need to know this information, but they are required by the
  // RunMultiThreaded function.
  int32 thread_id_; // 0 <= thread_number < num_threads
  int32 num_threads_;
 private:
  Nnet1Trainer &nnet_trainer_;
  const std::vector<TrainingExample> &data_;
  Mutex *mutex_; // pointer to mutex that guards category_list_
  std::vector<int32> *category_list_;
  double tot_like_;
  double *tot_like_ptr_;
  
};

double Nnet1Trainer::ForwardAndBackwardFinal(
    const std::vector<TrainingExample> &data) {
  // returns the objective function summed over all frames.
  
  // (1) detect whether there are any categories that are called
  // every time.  If so, we
  // first do the forward + backward on those, without threading.
  
  // Then for the remaining categories, we'll have a bunch
  // of threads take tasks from a queue.

  std::vector<int32> common_categories, other_categories;
  
  ListCategories(data, &common_categories, &other_categories);
  
  last_tanh_backward_.SetZero(); // Set this matrix containing
  // the "backward" derivative of the tanh layer, to all zeros.


  double ans = 0.0;
  
  // First call ForwardAndBackwardFinalForCategory for all the
  // categories (probably just 1) that appear on every single frame.
  // this will be multi-threaded by ATLAS.
  for (int32 i = 0; i < common_categories.size(); i++) {
    int32 category = common_categories[i];
    ans += ForwardAndBackwardFinalForCategory(data, category, true);
  }

  if (!other_categories.empty()) {
    Mutex mutex; // guards "other_categories".
    double tot_like = 0.0;
    ForwardAndBackwardFinalClass c(*this, data, &mutex,
                                   &other_categories, &tot_like);
    RunMultiThreaded(c); // will run with #threads = g_num_threads.
    ans += tot_like;
  }
  return ans;
}

// Does the forward and backward computation for the final two layers (softmax
// and linear), but just considering one of the categories of output labels.
double Nnet1Trainer::ForwardAndBackwardFinalForCategory(
    const std::vector<TrainingExample> &data,
    int32 category,
    bool common_category) {      
  Matrix<BaseFloat> temp_input_matrix; // if common_category is true,
  // temp_feature_matrix will be empty, else will be those features
  // for which we have that category.  Input to softmax layer.
  std::vector<int32> labels;
  std::vector<BaseFloat> weights;
  std::vector<int32> orig_index; // Index of this row in "full_input"
  Matrix<BaseFloat> &full_input = tanh_forward_data_.back(); // input to
  // softmax layer.
  { // set up labels, weights, orig_index, possibly temp_input_matrix.
    int32 chunk_size = tanh_forward_data_.back().NumRows() / num_chunks_,
        input_dim = full_input.NumCols();
    for (int32 i = 0; i < data.size(); i++)
      for (int32 j = 0; j < data[i].labels.size(); j++)
        for (int32 k = 0; k < data[i].labels[j].size(); k++)
          if (data[i].labels[j][k].first == category) {
            labels.push_back(data[i].labels[j][k].first);
            weights.push_back(data[i].weight);
            orig_index.push_back(chunk_size*i + j);
          }
    int32 num_examples = labels.size();
    KALDI_ASSERT((common_category && num_examples == full_input.NumRows()) ||
                 (!common_category && num_examples < full_input.NumRows()));
    if (!common_category) {
      temp_input_matrix.Resize(num_examples, input_dim);
      for (int32 t = 0; t < orig_index.size(); t++) {
        SubVector<BaseFloat> dest(temp_input_matrix, t),
            src(full_input, orig_index[t]);
        dest.CopyFromVec(src);
      }
    }
  }
  Matrix<BaseFloat> &input_matrix = (common_category ? temp_input_matrix
                                     : full_input);
  // set up labels and weights.
  
  
  Matrix<BaseFloat> input_deriv(input_matrix.NumRows(),
                                input_matrix.NumCols());

  double ans = ForwardAndBackwardFinalInternal(
      input_matrix, category, weights, labels, &input_deriv);
  
  // Now propagate the derivative from its temporary location to last_tanh_backward_.
  if (!common_category) {
    for (int32 t = 0; t < orig_index.size(); t++) {
      SubVector<BaseFloat> dest(last_tanh_backward_, orig_index[t]),
          src(input_deriv, t);
      dest.AddVec(1.0, src);
    }
  } else {
    last_tanh_backward_.AddMat(1.0, input_deriv);
  }
  return ans; // total log-like.
};

double Nnet1Trainer::ForwardAndBackwardFinalInternal(
    const Matrix<BaseFloat> &input, // input to softmax layer
    int32 category,
    const std::vector<BaseFloat> &weights, // one per example.
    const std::vector<int32> &labels, // one per example
    Matrix<BaseFloat> *input_deriv) { //derivative w.r.t "input".
  KALDI_ASSERT(weights.size() == input.NumRows());
  KALDI_ASSERT(weights.size() == labels.size());
  KALDI_ASSERT(input.NumRows() == input_deriv->NumRows());
  KALDI_ASSERT(input.NumCols() == input_deriv->NumCols());
  KALDI_ASSERT(category >= 0 && category < nnet_.NumCategories());

  const SoftmaxLayer &softmax_layer =
      *nnet_.final_layers_[category].softmax_layer;
  SoftmaxLayer *softmax_layer_to_update =
      nnet_to_update_->final_layers_[category].softmax_layer;
  const LinearLayer &linear_layer =
      *nnet_.final_layers_[category].linear_layer;
  LinearLayer *linear_layer_to_update =
      nnet_to_update_->final_layers_[category].linear_layer;
  
  // Forward data for softmax.
  Matrix<BaseFloat> softmax_forward(input.NumRows(),
                                    softmax_layer.OutputDim());
  
  softmax_layer.Forward(input,
                        &softmax_forward);
  
  // Forward data for linear layer.
  Matrix<BaseFloat> linear_forward(input.NumRows(),
                                   linear_layer.OutputDim());
  linear_layer.Forward(input,
                       &linear_forward);
  
  double ans = 0.0;
  // Compute objective function and its derivative.
  Matrix<BaseFloat> linear_backward(linear_forward.NumRows(),
                                    linear_forward.NumCols());
  for (int32 i = 0; i < linear_backward.NumRows(); i++) {
    int32 label = labels[i];
    BaseFloat weight = weights[i];
    KALDI_ASSERT(label < linear_backward.NumCols());
    BaseFloat prob = linear_backward(i, label);
    if (prob <= 0.0) {
      KALDI_WARN << "Zero probability in neural net training: " << prob;
      prob = 1.0e-20;
    }
    linear_backward(i, label) = weight / prob;
    ans += weight * log(prob);
  }

  Matrix<BaseFloat> softmax_backward(softmax_forward.NumRows(),
                                     softmax_backward.NumRows(),
                                     kUndefined);
  linear_layer.Backward(softmax_forward,
                        linear_backward,
                        &softmax_backward,
                        linear_layer_to_update);

  softmax_layer.Backward(input,
                         softmax_forward,
                         softmax_backward,
                         input_deriv,
                         softmax_layer_to_update);
  return ans;
}

void Nnet1Trainer::BackwardTanh() {
  Matrix<BaseFloat> cur_output_deriv;
  int32 num_layers = nnet_.initial_layers_.size();
  for (int32 layer = num_layers - 1; layer >= 0; layer--) {
    TanhLayer &tanh_layer = *nnet_.initial_layers_[layer].tanh_layer,
        *tanh_layer_to_update = nnet_to_update_->initial_layers_[layer].tanh_layer;
    Matrix<BaseFloat> &output_deriv = (layer == num_layers - 1 ?
                                       last_tanh_backward_ : cur_output_deriv);
    // spliced_input_deriv is the derivative w.r.t. the input of this
    // layer, but w.r.t. the possibly-spliced input that directly feeds into
    // the layer, not the original un-spliced input.
    Matrix<BaseFloat> spliced_input_deriv(output_deriv.NumRows(),
                                          tanh_layer.InputDim()),
        spliced_input(output_deriv.NumRows(),
                      tanh_layer.InputDim());
    SpliceFrames(tanh_forward_data_[layer],
                 num_chunks_,
                 &spliced_input);
    tanh_layer.Backward(spliced_input, // input to this layer.
                        tanh_forward_data_[layer+1], // output of this layer.
                        output_deriv, // deriv w.r.t. output of this layer
                        (layer == 0 ? NULL : &spliced_input_deriv),
                        tanh_layer_to_update);
    // Now we have to "un-splice" the input derivative.
    // This goes in a variable called "cur_output_deriv" which will
    // be the derivative w.r.t. the output of the previous layer.
    if (layer != 0) {
      cur_output_deriv.Resize(tanh_forward_data_[layer-1].NumRows(),
                              nnet_.initial_layers_[layer-1].tanh_layer->OutputDim());
      UnSpliceDerivative(spliced_input_deriv, num_chunks_,
                         &cur_output_deriv);
    }
  }
}


void Nnet1::ClearInitialLayers() {
  for (std::vector<InitialLayerInfo>::iterator iter = initial_layers_.begin();
       iter != initial_layers_.end(); ++iter) {
    if (iter->tanh_layer != NULL) {
      delete iter->tanh_layer ; 
    }
  }
  initial_layers_.clear() ; 
}

void Nnet1::ClearFinalLayers() {
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

void Nnet1::Destroy() {
  ClearInitialLayers();
  ClearFinalLayers();
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
  for (int32 i =0; i < num_categories; i++){
    FinalLayerInfo layer_info;
    // see comment above about the way TanhLayer is constructed
    // using a constructor taking the input stream as argument.
    layer_info.softmax_layer = new SoftmaxLayer(is, binary);
    layer_info.linear_layer = new LinearLayer(is, binary);
    final_layers_.push_back(layer_info);
  }
}


} // namespace kaldi
