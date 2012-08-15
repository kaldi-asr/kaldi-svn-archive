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

static void ListCategories(const std::vector<TrainingExample> &data,
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
                               std::vector<int32> *category_list):
      nnet_trainer_(nnet_trainer), data_(data), mutex_(mutex),
      category_list_(category_list), tot_like_(0), tot_like_ptr_(NULL) { }
  ForwardAndBackwardFinalClass(ForwardAndBackwardFinalClass &other):
      nnet_trainer_(other.nnet_trainer_), data_(other.data_),
      mutex_(other.mutex_), category_list_(other.category_list_), tot_like_(0.0),
      tot_like_ptr_(&other.tot_like_) { }

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

  double TotLike() { return tot_like_; }
  
  ~ForwardAndBackwardFinalClass() {
    // The destructor is responsible for collecting the "tot_like" quantities.
    if (tot_like_ptr_ != NULL)
      *tot_like_ptr_ += tot_like_;
  }
  
  // This function should be provided. Give it this exact implementation, with
  // the class name replaced with your own class's name.
  static void *run(void *c_in) {
    ForwardAndBackwardFinalClass *c = static_cast<ForwardAndBackwardFinalClass*>(c_in);
    (*c)(); // call operator () on it.
    return NULL;
  }  
  
 private:
  Nnet1Trainer &nnet_trainer_;
  const std::vector<TrainingExample> &data_;
  Mutex *mutex_; // pointer to mutex that guards category_list_
  std::vector<int32> *category_list_;
  double tot_like_;
  double *tot_like_ptr_;
  // The following class members are not actually needed, but are
  // required by the RunMultiThreaded function.
  int32 thread_id_; // 0 <= thread_number < num_threads
  int32 num_threads_;
  
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
    ForwardAndBackwardFinalClass c(*this, data, &mutex, &other_categories);
    RunMultiThreaded(c); // will run with #threads = g_num_threads.
    ans += c.TotLike();
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
    int32 chunk_size = tanh_forward_data_.back().NumRows() / num_chunks_;
        input_dim = full_input.NumCols();
    for (int32 i = 0; i < data.size(); i++)
      for (int32 j = 0; j < data[i].labels.size(); j++)
        for (int32 k = 0; k < data[i].labels[j].size(); k++)
          if (data[i].labels[j][k].first == category) {
            labels.push_back(data[i].labels[j][k].first);
            weights.push_back(data[i].weight);
            orig_index.push_back(chunk_size*i + j);
          }
    KALDI_ASSERT((common_category && num_examples == full_input.NumRows()) ||
                 (!common_category && num_examples < full_input.NumRows())); // or common_category
    // should be true.
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

double ForwardAndBackwardFinalInternal(
    const Matrix<BaseFloat> &input, // input to softmax layer
    int32 category,
    const std::vector<BaseFloat> &weights, // one per example.
    const std::vector<int32> &labels, // one per example
    Matrix<BaseFloat> *input_deriv) { //derivative w.r.t "input".
  KALDI_ASSERT(weights.size() == input.NumRows());
  KALDI_ASSERT(weights.size() == labels.size());
  KALDI_ASSERT(input.NumRows() == input_deriv->NumRows());
  KALDI_ASSERT(input.NumCols() == input_deriv->NumCols());
  KALDI_ASSERT(category >= 0 && category < nnet.NumCategories());

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
    linear_backward(i, label) = 1.0 / prob;
    ans += log(prob);
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
}

void Nnet1Trainer::BackwardTanh() {
  Matrix<BaseFloat> cur_output_deriv;
  
  for (int32 layer = initial_layers_.size() - 1; layer >= 0; layer--) {
    TanhLayer &tanh_layer = nnet_.initial_layers_[layer],
        *tanh_layer_to_update = &nnet_to_update_->initial_layers_[layer];
    Matrix<BaseFloat> &output_deriv = (layer == initial_layers_.size() - 1 ?
                                       last_tanh_backward_ : cur_output_deriv);
    // spliced_input_deriv is the derivative w.r.t. the input of this
    // layer, but w.r.t. the possibly-spliced input that directly feeds into
    // the layer, not the original un-spliced input.
    Matrix<BaseFloat> spliced_input_deriv(output_deriv.NumRows(),
                                          tanh_layer.InputDim());
    tanh_layer.Backward(tanh_forward_data_[i],

                                          
                                          
    
  }
}




} // namespace kaldi
