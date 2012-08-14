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
  
void Nnet1Trainer::ForwardTanh() {
  // Does the forward computation for the initial (tanh) layers.
  for (int32 layer = 0; layer < initial_layers_.size(); layer++) {
    Matrix<BaseFloat> temp_mat_(...); // Splice together the
    // data appropriately, if needed...

    // Do the forward pass; put in tanh_data_[layer+1]
    // This is just a call to something in layer.h
  }  
}

static void ListCategories(const std::vector<TrainingExample> &data,
                           std::vector<int32> *common_categories,
                           std::vector<int32> *other_categories) {
  std::unordered_map<int32, int32> category_count;
  int32 num_frames = 0;
  for (int32 chunk = 0; chunk < data.size(); chunk++) {
    for (int32 t = 0; t < data[chunk].labels.size(); t++) {
      num_frames++;
      std::vector<int32> check_vec;
      for (int32 idx = 0; idx < data[chunk].labels[t].size()) { // iterate
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
  for (std::unordered_map<int32, int32>::iterator iter = category_count.begin();
       iter != category_count.end(); ++iter) {
    int32 category = iter->first, count = iter->second;
    if (count == num_frames) { // seen on every frame -> add to "common_categories".
      common_categories->push_back(category);
    } else {
      frequency_list.push_back(std::pair<int32, int32>(-count, category));
    }
  }
  std::sort(frequency_list.begin(), frequency_list.end()); // sorts in increasing
  // order of -count, so decreasing order of count [biggest first].
  for (size_t i = 0; i < frequency_list.end(); i++) {
    int32 category = frequency_list[i].second;
    other_categories->push_back(category);
  }
}

double Nnet1Trainer::ForwardAndBackwardFinal(const std::vector<TrainingExample> &data) {
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

  // HERE-- call for all categories...
  
}

// Does the forward and backward computation for the final two layers (softmax
// and linear), but just considering one of the categories of output labels.
double ForwardAndBackwardFinalForCategory(const std::vector<TrainingExample> &data,
                                          int32 category,
                                          bool common_category) {
  const SoftmaxLayer &softmax_layer = nnet_.final_layers_[category].softmax_layer;
  SoftmaxLayer *softmax_layer_to_update = nnet_to_update->final_layers_[category].softmax_layer;
  const LinearLayer &linear_layer = nnet_.final_layers_[category].linear_layer;
  LinearLayer *linear_layer_to_update = nnet_to_update->final_layers_[category].linear_layer;
      

  
  Matrix<BaseFloat> temp_feature_matrix; // if common_category is true,
  // temp_feature_matrix will be empty, else will be those features
  // for which we have that category.
  vector<int32> labels;
  if (!common_category) {
    // TODO: set up temp_feature_matrix, and labels
  } else {
    // just set up labels.
  }
  
  Matrix<BaseFloat> &feature_matrix = (common_category ? temp_feature_matrix
                                       : tanh_data_.back());

  // Forward data for softmax.
  Matrix<BaseFloat> softmax_forward(feature_matrix.NumRows(),
                                    softmax_layer.OutputDim());
  
  softmax_layer.Forward(feature_matrix,
                        &softmax_forward);
  
  // Forward data for linear layer.
  Matrix<BaseFloat> linear_forward(feature_matrix.NumRows(),
                                   linear_layer.OutputDim());
  linear_layer.Forward(feature_matrix,
                       &linear_forward);

  double ans = 0.0;
  // Compute objective function and its derivative.
  Matrix<BaseFloat> linear_backward(linear_forward.NumRows(),
                                    linear_forward.NumCols());
  for (int32 i = 0; i < linear_backward.NumRows(); i++) {
    int32 label = labels[i];
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
  nnet_.final_layers_[category].linear_layer->Backward(softmax_forward,
                                                       linear_backward,
                                                       &softmax_backward,

                                                       
                                                 
  
  
  
}

} // namespace kaldi
