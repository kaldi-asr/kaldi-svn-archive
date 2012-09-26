// nnet/nnet-update.cc

// Copyright 2012   Johns Hopkins University (author: Daniel Povey)

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

#include "nnet-cpu/nnet-update.h"

namespace kaldi {


// This class NnetUpdater contains functions for updating the neural net or
// computing its gradient, given a set of NnetTrainingExamples.  Its functionality
// is exported by DoBackprop(), so we define it in the .cc file.
class NnetUpdater {
 public:
  // Note: in the case of training with SGD, "nnet" and "nnet_to_update" will
  // be identical.  They'll be different if we're accumulating the gradient
  // for a held-out set and don't want to update the model.
  NnetUpdater(const Nnet &nnet,
              int32 minibatch_size,
              Nnet *nnet_to_update);
  
  BaseFloat TrainOnMinibatch(const std::vector<NnetTrainingExample> &data);
  // returns average objective function over this minibatch.
  
 private:
  void FormatInput(const std::vector<NnetTrainingExample> &data); // takes the
  // input and formats as a single matrix, in forward_data_[0].
  
  // Possibly splices input together from forward_data_[component].
  //   MatrixBase<BaseFloat> &GetSplicedInput(int32 component, Matrix<BaseFloat> *temp_matrix);


  void Propagate();

  void Backprop();

  const Nnet &nnet_;
  int32 minibatch_size_;
  Nnet *nnet_to_update_;
  
  std::vector<Matrix<BaseFloat> > forward_data_; // The forward data
  // for the outputs of the tanh components; has a 1 appended (so larger
  // dimension than the actual output if the component in question will not
  // be spliced.  Indexed by [component][t][dim]; tanh_forward[i] is the input
  // of component i.  
};

NnetUpdater::NnetUpdater(const Nnet &nnet,
                         int32 minibatch_size,
                         Nnet *nnet_to_update):
    nnet_(nnet), minibatch_size_(minibatch_size),
    nnet_to_update_(nnet_to_update) {
  forward_data_.resize(nnet_.NumComponents() + 1); // one for the input of
  // each layer, and one for the output of the last layer.

  // TODO: remove the following.
  
  // Resize the matrices in forward_data_.  Each of these matrices will be the
  // output of the previous layer, except forward_data_[0] which will be the
  // input of the first layer.  Note: the output of layer n is not necessarily
  // equal to the input of layer n+1, because of splicing issues.

  // First handle forward_data_[0], which is the input of the 1st layer.
  forward_data_[0].Resize(minibatch_size * FullSplicingForComponent(1),
                          nnet_.Component(0).InputDim());

  // For each of the other indices i, the dimension is the dimension of the output
  // of i-1'th component.  The number of spliced frames is whichever number the
  // next component needed, and the dimension is the actual output dimension
  // of the i-1'th layer.
  for (int32 i = 1; i < forward_data_.size(); i++) {
    // forward_data_[i] ...
  }  
}

/**
  Explanation for SpliceData:
  First, "Next" and "Previous" are relative to the space "between" the
  components.  So the next and previous components are e.g. layer c+1 and c.
  Suppose for each sample, at the output of component c+1 we need frames
  [ -1, 0, 1 ] relative to each sample, and this component splices frames
  [ -5, 0, 5 ] at its input.  At the output of component c we'd need frames
  [ -6, -5, -4, -1, 0, 1, 4, 5, 6 ].
  As indexes into this array, the first of the three outputs (frame -1)
  would have spliced together relative frame indexes [ -6 -1 4 ], which
  expressed as indexes into the list are [ 0 3 6 ].  So rel_splicing
  (a variable used in this function) would be an array [ 0 3 6; 1 4 7; 2 5 8 ].
 */
void NnetUpdater::SpliceData(const MatrixBase<BaseFloat> &prev_output,
                             MatrixBase<BaseFloat> *next_input,
                             int32 c) { // c is component index of next_input.
  int32 prev_num_frames = prev_output.NumRows() / minibatch_size_,
      next_num_frames = next_input->NumRows() / minibatch_size_,
      prev_output_dim = prev_output.NumCols(),
      next_input_dim = next_input->NumCols();
      
  // This function handles the data splicing from the output of
  // component c-1 to the input of component c.
  if (RawSplicingForComponent(c).size() == 1) { // no splicing is going on,
    // so just copy data.  Note: even if this array is not [ 0 ] but something
    // like [ -1 ], this would not matter.
    next_input.CopyFromMat(prev_output);
  } else {
    const std::vector<std::vector<int32> > &rel_splicing =
        nnet_.RelativeSplicingForComponent(c);
    KALDI_ASSERT(next_num_frames == rel_splicing.size());
    int32 num_splice = rel_splicing[0].size(); // #times we splice old output
    KALDI_ASSERT(prev_output_dim * num_splice == next_input_dim);
    for (int32 m = 0; m < minibatch_size_; m++) {
      for (int32 f = 0; f < next_num_frames; f++) {
        SubVector<BaseFloat> next_frame(*next_input, m * next_num_frames + f);
        for (int32 s = 0; s < num_splice; s++) {
          int32 index = rel_splicing[f][s];
          KALDI_ASSERT(index >= 0 && index < prev_num_frames);
          SubVector<BaseFloat> next_frame_part(next_frame, prev_output_dim * s,
                                               prev_output_dim);
          SubVector<BaseFloat> prev_frame(prev_output,
                                          m * prev_num_frames + index);
          next_frame_part.CopyFromVec(prev_frame);
        }
      }
    }
  }
  
  


}

void NnetUpdater::FormatInput(const std::vector<NnetTrainingExample> &data) {
  // first, some checks.
  KALDI_ASSERT(data.size() > 0);
  KALDI_ASSERT(data[0].input_frames.NumRows() ==
               nnet_.FullSplicingForComponent(0).size());
  KALDI_ASSERT(data[0].input_frames.NumCols() == nnet_.FeatureDim());
  KALDI_ASSERT(data[0].spk_info.Dim() == nnet_.SpeakerDim());

  // frames_per_sample is the number of separate frames of output we'll
  // need from the zeroth layer, for each sample.
  int32 frames_per_sample = nnet_.FullSplicingForComponent(1).size();

  int32 feature_dim = nnet_.FeatureDim(); // Raw feature dim.
  forward_data_[0].Resize(minibatch_size_ * frames_per_sample,
                          nnet_.Component(0).InputDim());
  int32 speaker_dim = nnet_.SpeakerDim(); // Speaker-info dim; may be 0.

  const std::vector<std::vector<int32> > &relative_splicing =
      nnet_.RelativeSplicingForComponent(0);
  KALDI_ASSERT(relative_splicing.size() = frames_per_sample);

  for (int32 m = 0; m < minibatch_size_; m++) {
    for (int32 s = 0; s < frames_per_sample; s++) { // for each
      // of the output frames of component 0; for this minibatch...
      SubVector<BaseFloat> spliced(forward_data_[0], m * frames_per_sample + s);
      for (int32 t = 0; t < relative_splicing[s].size(); t++) {
        int32 input_index = relative_splicing[s][t];
        SubVector<BaseFloat> input(data[m].input_frames.Row(t));
        SubVector<BaseFloat> this_output(spliced, feature_dim * s, feature_dim);
        this_output.CopyFromVec(input);
      }
      int32 num_spliced = relative_splicing[s].size(); // actually doesn't vary
      // with s.
      KALDI_ASSERT(num_spliced * feature_dim + speaker_dim == spliced.Dim());
      if (speaker_dim != 0) {
        SubVector<BaseFloat> this_output(spliced, num_spliced * feature_dim,
                                         speaker_dim);
        this_output.CopyFromVec(data[m].spk_info);
      }
    }
  }
  
  
}


  
  
} // namespace
