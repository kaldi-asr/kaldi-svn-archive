// nnet_dp/update-nnet1.cc

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

#include "nnet_dp/update-nnet1.h"
#include "thread/kaldi-thread.h"

namespace kaldi {


Nnet1Updater::Nnet1Updater(
    const Nnet1 &nnet,
    int32 chunk_size_at_output, // size of chunks (number of output labels).
    int32 num_chunks, // number of chunks we process at the same time.
    Nnet1 *nnet_to_update): nnet_(nnet), num_chunks_(num_chunks),
                            nnet_to_update_(nnet_to_update) {
  last_tanh_backward_.Resize(num_chunks * chunk_size_at_output,
                             nnet_.initial_layers_.back().tanh_layer->OutputDim());
  
  int32 cur_chunk_size = chunk_size_at_output; // as we go from the last to
  // the first tanh layer, this will be the chunk size at the current layer; it
  // increases as we go back, as we may need more context.

  int32 num_tanh_layers = nnet_.initial_layers_.size();
  // One for each layer, plus one for the input.
  tanh_forward_data_.resize(nnet_.initial_layers_.size() + 1);
  
  for (int32 l = num_tanh_layers-1; l >= 0; l--) {

    Matrix<BaseFloat> &output_forward_data = tanh_forward_data_[l+1];
    output_forward_data.Resize(cur_chunk_size * num_chunks,
                               nnet_.initial_layers_[l].tanh_layer->OutputDim());
    cur_chunk_size += nnet_.LeftContextForLayer(l) +
        nnet_.RightContextForLayer(l);
  }
  tanh_forward_data_[0].Resize(cur_chunk_size * num_chunks,
                               nnet_.InputDim());
}


void Nnet1Updater::FormatInput(const std::vector<TrainingExample> &data) {
  KALDI_ASSERT(data.size() == num_chunks_);
  int32 chunk_size = tanh_forward_data_[0].NumRows() / num_chunks_;
  int32 input_dim = tanh_forward_data_[0].NumCols(); // Dimension of input
  // data that neural net sees.
  int32 raw_input_dim = data[0].input.NumCols();
  if (input_dim != raw_input_dim && input_dim != raw_input_dim + 1) {
    KALDI_ERR << "Dimension mismatch: neural net expects data of dimension "
              << input_dim << " but is being trained on data of dim "
              << raw_input_dim;
  }
  for (int32 i = 0; i < num_chunks_; i++) {
    SubMatrix<BaseFloat> dest(tanh_forward_data_[0],
                              i * chunk_size, chunk_size, 0, raw_input_dim);
    if (input_dim == raw_input_dim + 1) // extend features with 1.0.
      for (int32 j = 0; j < chunk_size; j++)
        tanh_forward_data_[0](i * chunk_size + j, raw_input_dim) = 1.0;
  }
}

BaseFloat Nnet1Updater::TrainOnOneMinibatch(const std::vector<TrainingExample> &data) {
  BaseFloat ans;
  FormatInput(data);
  ForwardTanh();
  ans = ForwardAndBackwardFinal(data);
  BackwardTanh();
  return ans;
}

void Nnet1Updater::ForwardTanh() {
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

void Nnet1Updater::ListCategories(
    const std::vector<TrainingExample> &data,
    std::vector<int32> *common_categories,
    std::vector<int32> *other_categories,
    BaseFloat *tot_weight_ptr) {
  unordered_map<int32, int32> category_count;
  double tot_weight = 0.0;
  int32 num_frames = 0;
  for (int32 chunk = 0; chunk < data.size(); chunk++) {
    for (int32 t = 0; t < data[chunk].labels.size(); t++) {
      num_frames++;
      tot_weight += data[chunk].weight;
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
  *tot_weight_ptr = tot_weight;
}

class Nnet1Updater::ForwardAndBackwardFinalClass: public MultiThreadable {
 public:
  ForwardAndBackwardFinalClass(Nnet1Updater &nnet_trainer,
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
  
 private:
  Nnet1Updater &nnet_trainer_;
  const std::vector<TrainingExample> &data_;
  Mutex *mutex_; // pointer to mutex that guards category_list_
  std::vector<int32> *category_list_;
  double tot_like_;
  double *tot_like_ptr_;
  
};

BaseFloat Nnet1Updater::ForwardAndBackwardFinal(
    const std::vector<TrainingExample> &data) {
  // returns the objective function summed over all frames.
  
  // (1) detect whether there are any categories that are called
  // every time.  If so, we
  // first do the forward + backward on those, without threading.
  
  // Then for the remaining categories, we'll have a bunch
  // of threads take tasks from a queue.

  std::vector<int32> common_categories, other_categories;
  BaseFloat tot_weight; // normalizer for answer we return (the avg log-prob
  // per frame).
  
  ListCategories(data, &common_categories, &other_categories, &tot_weight);
  
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
    RunMultiThreadedPersistent(c); // will run with #threads = g_num_threads.
    ans += tot_like;
  }
  return ans / tot_weight;
}

// Does the forward and backward computation for the final two layers (softmax
// and linear), but just considering one of the categories of output labels.
BaseFloat Nnet1Updater::ForwardAndBackwardFinalForCategory(
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

BaseFloat Nnet1Updater::ForwardAndBackwardFinalInternal(
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

void Nnet1Updater::BackwardTanh() {
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


} // namespace kaldi
