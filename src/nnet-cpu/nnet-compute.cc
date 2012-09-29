// nnet/nnet-compute.cc

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

#include "nnet-cpu/nnet-compute.h"

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
              Nnet *nnet_to_update);
  
  BaseFloat ComputeForMinibatch(const std::vector<NnetTrainingExample> &data);
  // returns average objective function over this minibatch.
  
 private:
  // Splices together the output of one layer into the
  // input of the next.
  void SpliceData(const MatrixBase<BaseFloat> &prev_output,
                  int32 c, // component index of component for which
                           // "input" is the input
                  MatrixBase<BaseFloat> *input);

  // UnSpliceDerivatives is like SpliceData but it's for Backprop;
  // the "copy" operation becomes a "sum" operation in reverse.
  void UnSpliceDerivatives(const MatrixBase<BaseFloat> &input_deriv,
                           int32 c, // component index of component for which
                           // "input_deriv" is input derivative.
                           MatrixBase<BaseFloat> *preb_output_deriv);
  
  void FormatInput(const std::vector<NnetTrainingExample> &data); // takes the
  // input and formats as a single matrix, in forward_data_[0].
  
  // Possibly splices input together from forward_data_[component].
  //   MatrixBase<BaseFloat> &GetSplicedInput(int32 component, Matrix<BaseFloat> *temp_matrix);


  void Propagate();

  /// Computes objective function and derivative at output layer.
  BaseFloat ComputeObjfAndDeriv(const std::vector<NnetTrainingExample> &data,
                                Matrix<BaseFloat> *deriv) const;
  
  /// Returns objf summed (and weighted) over samples.
  /// Note: "deriv" will contain, at input, the derivative w.r.t. the
  /// output layer but will be used as a temporary variable by
  /// this function.
  void Backprop(const std::vector<NnetTrainingExample> &data,
                Matrix<BaseFloat> *deriv);

  const Nnet &nnet_;
  Nnet *nnet_to_update_;
  int32 minibatch_size_;
  
  std::vector<Matrix<BaseFloat> > forward_data_; // The forward data
  // for the outputs of the tanh components; has a 1 appended (so larger
  // dimension than the actual output if the component in question will not
  // be spliced.  Indexed by [component][t][dim]; tanh_forward[i] is the input
  // of component i.
  
};


/*
   Explanation of variable names:
    num_frames = the number of context frames for each training data sample,
       at a particular layer.  This is required due to to the context splicing.
       For a component c, num_frames will equal nnet.FullSplicingForComponent(c+1).size();
       this is the number of separate frames of data we process at this layer,
       for each data sample.
     frame_idx ranges from 0 to num_frames-1.
    num_splice = the number of separate input frames (at different times)
       that we splice together for component c.  This will equal
       nnet.RawSplicingForComponent(c).sze().
     splice_idx ranges from 0 to num_splice-1.       
*/

NnetUpdater::NnetUpdater(const Nnet &nnet,
                         Nnet *nnet_to_update):
    nnet_(nnet), nnet_to_update_(nnet_to_update) { }
 

BaseFloat NnetUpdater::ComputeForMinibatch(
    const std::vector<NnetTrainingExample> &data) {
  minibatch_size_ = data.size();
  FormatInput(data);
  Propagate();
  Matrix<BaseFloat> tmp_deriv;
  BaseFloat ans = ComputeObjfAndDeriv(data, &tmp_deriv);
  Backprop(data, &tmp_deriv); // this is summed (after weighting), not averaged.
  return ans;
}

void NnetUpdater::Propagate() {
  int32 num_components = nnet_.NumComponents();
  for (int32 c = 0; c < num_components; c++) {
    const Component &component = nnet_.GetComponent(c);
    Matrix<BaseFloat> temp_input; // input to this layer.
    bool do_splicing = (c != 0 && nnet_.RawSplicingForComponent(c).size() != 1);
    if (do_splicing) {
      temp_input.Resize(
          nnet_.FullSplicingForComponent(c+1).size() * minibatch_size_,
          component.InputDim());
      SpliceData(forward_data_[c], c, &temp_input);
    }
    Matrix<BaseFloat> &input = (do_splicing ? temp_input : forward_data_[c]),
                     &output = forward_data_[c+1];
    component.Propagate(input, &output);
    // If we won't need the output of the previous layer for
    // backprop, delete it to save memory.
    if (!(c == num_components-1 ||
          (c>0 && nnet_.GetComponent(c-1).BackpropNeedsOutput()) ||
          component.BackpropNeedsInput())) {
      forward_data_[c].Resize(0, 0); // We won't need this data.
    }
  }
}

BaseFloat NnetUpdater::ComputeObjfAndDeriv(
    const std::vector<NnetTrainingExample> &data,
    Matrix<BaseFloat> *deriv) const {
  BaseFloat floor = 1.0e-20; // Avoids division by zero.
  double tot_objf = 0, tot_weight = 0;
  int32 num_components = nnet_.NumComponents();  
  deriv->Resize(minibatch_size_, nnet_.OutputDim()); // sets to zero.
  const Matrix<BaseFloat> &output(forward_data_[num_components]);
  KALDI_ASSERT(SameDim(output, *deriv));
  for (int32 m = 0; m < minibatch_size_; m++) {
    int32 label = data[m].label;
    BaseFloat weight = data[m].weight;
    KALDI_ASSERT(label >= 0 && label < nnet_.OutputDim());
    BaseFloat this_prob = output(m, label);
    if (this_prob < floor) {
      KALDI_WARN << "Probability is " << this_prob << ", flooring to "
                 << floor;
      this_prob = floor;
    }
    tot_objf += weight * log(this_prob);
    tot_weight += weight;
    (*deriv)(m, label) = 1.0 / this_prob;
    
  }
  KALDI_VLOG(4) << "Objective function is " << (tot_objf/tot_weight) << " over "
                << tot_weight << " samples (weighted).";
  return tot_objf;
}


void NnetUpdater::Backprop(const std::vector<NnetTrainingExample> &data,
                           Matrix<BaseFloat> *deriv) {
  BaseFloat tot_weight = 0.0;
  for (size_t i = 0; i < data.size(); i++)
    tot_weight += data[i].weight;
  // First compute gradient at output.  We assume it's cross-entropy,
  // i.e. the probability of the correct class.

  for (int32 c = nnet_.NumComponents() - 1; c >= 0; c--) {
    const Component &component = nnet_.GetComponent(c);
    Component *component_to_update = (nnet_to_update_ == NULL ? NULL :
                                      &(nnet_to_update_->GetComponent(c)));
    bool backprop_needs_input = component.BackpropNeedsInput();
    Matrix<BaseFloat> temp_input; // input to this layer.
    bool do_splicing = (c != 0 && nnet_.RawSplicingForComponent(c).size() != 1);
    if (do_splicing && backprop_needs_input) {
      temp_input.Resize(
          nnet_.FullSplicingForComponent(c+1).size() * minibatch_size_,
          component.InputDim());
      SpliceData(forward_data_[c], c, &temp_input);
    }
    Matrix<BaseFloat> &input = (do_splicing ? temp_input : forward_data_[c]),
                     &output = forward_data_[c+1];
    // Note: one or both of input and output may be the empty matrix,
    // depending on the values of BackpropNeedsInput() and BackpropNeedsOutput()
    // for this component.
    Matrix<BaseFloat> input_deriv(minibatch_size_ *
                                  nnet_.FullSplicingForComponent(c+1).size(),
                                  component.InputDim());
    const Matrix<BaseFloat> &output_deriv(*deriv); // *deriv is currently derivative of objf
    // w.r.t. the output of this layer.

    component.Backprop(input, output, output_deriv, tot_weight,
                       component_to_update, &input_deriv);

    if (!do_splicing) {
      // Overwrite *deriv with input_deriv, which
      // is derivative of objective function w.r.t. output previous layer    
      *deriv = input_deriv;
    } else {
      UnSpliceDerivatives(input_deriv, c, deriv); // This does the reverse of
      // SpliceData().
    }
  }
}

/**
  Explanation for SpliceData:
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
                             int32 c, // c is component index of input.
                             MatrixBase<BaseFloat> *input) {
  int32 prev_num_frames = prev_output.NumRows() / minibatch_size_,
             num_frames = input->NumRows() / minibatch_size_,
        prev_output_dim = prev_output.NumCols(),
              input_dim = input->NumCols();
      
  // This function handles the data splicing from the output of
  // component c-1 to the input of component c.
  if (nnet_.RawSplicingForComponent(c).size() == 1) { // no splicing is going on,
    // so just copy data.  Note: even if this array is not [ 0 ] but something
    // like [ -1 ], this would not matter.
    input->CopyFromMat(prev_output);
  } else {
    const std::vector<std::vector<int32> > &rel_splicing =
        nnet_.RelativeSplicingForComponent(c);
    KALDI_ASSERT(num_frames == rel_splicing.size());
    int32 num_splice = rel_splicing[0].size(); // #times we splice old output
    KALDI_ASSERT(prev_output_dim * num_splice == input_dim);
    for (int32 m = 0; m < minibatch_size_; m++) {
      for (int32 frame_idx = 0; frame_idx < num_frames; frame_idx++) {
        SubVector<BaseFloat> frame(*input, m * num_frames + frame_idx);
        for (int32 splice_idx = 0;
             splice_idx < num_splice;
             splice_idx++) {
          int32 prev_frame_idx = rel_splicing[frame_idx][splice_idx];
          KALDI_ASSERT(prev_frame_idx >= 0 && prev_frame_idx < prev_num_frames);
          SubVector<BaseFloat> frame_part(frame, prev_output_dim * splice_idx,
                                          prev_output_dim);
          SubVector<BaseFloat> prev_frame(prev_output,
                                          m * prev_num_frames + prev_frame_idx);
          frame_part.CopyFromVec(prev_frame);
        }
      }
    }
  }
}

void NnetUpdater::UnSpliceDerivatives(
    const MatrixBase<BaseFloat> &input_deriv,
    int32 c, // c is component index of layer for which input_deriv is input.
    MatrixBase<BaseFloat> *prev_output_deriv) {
  int32 prev_num_frames = prev_output_deriv->NumRows() / minibatch_size_,
             num_frames = input_deriv.NumRows() / minibatch_size_,
        prev_output_dim = prev_output_deriv->NumCols(),
              input_dim = input_deriv.NumCols();
  
  // This function handles the data splicing from the output of
  // component c-1 to the input of component c.
  if (nnet_.RawSplicingForComponent(c).size() == 1) { // no splicing is going on,
    // so just copy data.  Note: even if this array is not [ 0 ] but something
    // like [ -1 ], this would not matter.
    prev_output_deriv->CopyFromMat(input_deriv);
  } else {
    const std::vector<std::vector<int32> > &rel_splicing =
        nnet_.RelativeSplicingForComponent(c);
    KALDI_ASSERT(num_frames == rel_splicing.size());
    int32 num_splice = rel_splicing[0].size(); // #times we splice old output
    KALDI_ASSERT(prev_output_dim * num_splice == input_dim);
    prev_output_deriv->SetZero();
    for (int32 m = 0; m < minibatch_size_; m++) {
      for (int32 frame_idx = 0; frame_idx < num_frames; frame_idx++) {
        SubVector<BaseFloat> frame(input_deriv, m * num_frames + frame_idx);
        for (int32 splice_idx = 0;
             splice_idx < num_splice;
             splice_idx++) {
          int32 prev_frame_idx = rel_splicing[frame_idx][splice_idx];
          KALDI_ASSERT(prev_frame_idx >= 0 && prev_frame_idx < prev_num_frames);
          SubVector<BaseFloat> frame_part(frame, prev_output_dim * splice_idx,
                                          prev_output_dim);
          SubVector<BaseFloat> prev_frame(*prev_output_deriv,
                                          m * prev_num_frames + prev_frame_idx);
          prev_frame.AddVec(1.0, frame_part);
          // The "forward" version of the code that just does the splicing did:
          // frame_part.CopyFromVec(prev_frame);
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
  
  // num_frames is the number of separate frames of output we'll
  // need from the zeroth layer, for each sample.
  int32 num_frames = nnet_.FullSplicingForComponent(1).size();
  
  int32 feature_dim = nnet_.FeatureDim(); // Raw feature dim.

  forward_data_.resize(nnet_.NumComponents() + 1); // one for the input of
  // each layer, and one for the output of the last layer.

  forward_data_[0].Resize(minibatch_size_ * num_frames,
                          nnet_.GetComponent(0).InputDim());

  int32 speaker_dim = nnet_.SpeakerDim(); // Speaker-info dim; may be 0.

  const std::vector<std::vector<int32> > &relative_splicing =
      nnet_.RelativeSplicingForComponent(0);
  KALDI_ASSERT(relative_splicing.size() == num_frames);
  int32 num_splice = relative_splicing[0].size();
  KALDI_ASSERT(num_splice * feature_dim + speaker_dim ==
               forward_data_[0].NumCols());
  
  for (int32 m = 0; m < minibatch_size_; m++) {
    for (int32 frame_idx = 0; frame_idx < num_frames; frame_idx++) { // for each
      // of the output frames of component 0, for this minibatch...
      SubVector<BaseFloat> frame(forward_data_[0], m * num_frames + frame_idx);
      for (int32 splice_idx = 0; splice_idx < num_splice; splice_idx++) {
        int32 input_frame_idx = relative_splicing[frame_idx][splice_idx];
        SubVector<BaseFloat> input(data[m].input_frames.Row(input_frame_idx));
        SubVector<BaseFloat> this_output(frame,
                                         feature_dim * splice_idx, feature_dim);
        this_output.CopyFromVec(input);
      }
      if (speaker_dim != 0) {
        SubVector<BaseFloat> this_output(frame, num_splice * feature_dim,
                                         speaker_dim);
        this_output.CopyFromVec(data[m].spk_info);
      }
    }
  }
}

BaseFloat TotalNnetTrainingWeight(const std::vector<NnetTrainingExample> &egs) {
  double ans = 0.0;
  for (size_t i = 0; i < egs.size(); i++)
    ans += egs[i].weight;
  return ans;
}

BaseFloat DoBackprop(const Nnet &nnet,
                     const std::vector<NnetTrainingExample> &examples,
                     Nnet *nnet_to_update) {
  NnetUpdater updater(nnet, nnet_to_update);
  return updater.ComputeForMinibatch(examples);  
}



  
  
} // namespace
