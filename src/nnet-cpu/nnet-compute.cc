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


class NnetComputer {
 public:
  // Note: this class maintains a reference to nnet.  Don't delete it
  // while this class still exists.
  NnetComputer(const Nnet &nnet,
               const MatrixBase<BaseFloat> &input_feats,
               const VectorBase<BaseFloat> &spk_info,
               bool pad,
               Nnet *nnet_to_update); // nnet_to_update may be NULL.

  // The forward-through-the-layers part of the computation.
  void Propagate(bool will_do_backprop);

  
  // Backprop does not currently support weights on the frames.  We could easily
  // add support for weights if we needed it.  Returns objective function,
  // summed over the labels.
  BaseFloat Backprop(const std::vector<int32> &labels);
  
  MatrixBase<BaseFloat> &GetOutput() { return forward_data_.back(); }
  
 private:
  // Computes objf derivative at last layer, and returns objective
  // function summed over labels.
  BaseFloat ComputeLastLayerDeriv(const std::vector<int32> &labels,
                                  Matrix<BaseFloat> *deriv) const;
  
  // Splices together the output of one layer into the input of the next,
  // taking care of splicing.
  void SpliceData(const MatrixBase<BaseFloat> &prev_output,
                  int32 c, // component index of component for which
                           // "input" is the input
                  MatrixBase<BaseFloat> *input) const;

  // This is like SpliceData but it's what we do when we're propagating
  // the derivatives back.
  void UnSpliceDerivatives(const MatrixBase<BaseFloat> &input_deriv,
                           int32 c, // component index of component for which
                           // "input_deriv" is the derivative of the input
                           MatrixBase<BaseFloat> *prev_output_deriv) const;

  // This function takes the raw feature data in forward_data_[0], and
  // splices it together as needed, together with the speaker information.
  void SpliceFirstLayerInput(Matrix<BaseFloat> *spliced_input) const;
  
  const Nnet &nnet_;
  Vector<BaseFloat> spk_info_;
  std::vector<Matrix<BaseFloat> > forward_data_;
  Nnet *nnet_to_update_; // May be NULL, if just want objective function
  // but no gradient info or SGD.
};

NnetComputer::NnetComputer(const Nnet &nnet,
                           const MatrixBase<BaseFloat> &input_feats,
                           const VectorBase<BaseFloat> &spk_info,
                           bool pad,
                           Nnet *nnet_to_update):
    nnet_(nnet), spk_info_(spk_info), nnet_to_update_(nnet_to_update) {
  KALDI_ASSERT(input_feats.NumRows() != 0 &&
               input_feats.NumCols() == nnet_.FeatureDim());
  forward_data_.resize(nnet.NumComponents() + 1);
  if (!pad) {
    forward_data_[0] = input_feats;
  } else {
    int32 num_rows = nnet_.LeftContext() + input_feats.NumRows() +
                     nnet_.RightContext();
    Matrix<BaseFloat> &padded_input(forward_data_[0]);
    padded_input.Resize(num_rows, input_feats.NumCols());
    padded_input.Range(nnet_.LeftContext(), input_feats.NumRows(),
                           0, input_feats.NumCols()).CopyFromMat(input_feats);
    for (int32 i = 0; i < nnet_.LeftContext(); i++)
      padded_input.Row(i).CopyFromVec(input_feats.Row(0));
    int32 last_row = input_feats.NumRows() - 1;
    for (int32 i = 0; i < nnet_.RightContext(); i++)
      padded_input.Row(num_rows - i - 1).CopyFromVec(input_feats.Row(last_row));
  }
  KALDI_ASSERT(spk_info_.Dim() == nnet_.SpeakerDim());
}


// Splices together the output of one layer into the input of the next,
// taking care of splicing.  Caution: "input" is the input of the next
// layer; it's actually the output of this function.
void NnetComputer::SpliceData(const MatrixBase<BaseFloat> &prev_output,
                              int32 c, // component index of component for which
                                       // "input" is the input
                              MatrixBase<BaseFloat> *input) const {
  const std::vector<int32> &raw_splicing = nnet_.RawSplicingForComponent(c);
  int32 left_context = -raw_splicing.front(),
       right_context = raw_splicing.back();
  if (left_context == 0 && right_context == 0) {
    input->CopyFromMat(prev_output);
    return;
  }
  KALDI_ASSERT(input->NumRows() == prev_output.NumRows() - left_context - right_context);
  int32 num_splice = raw_splicing.size();
  int32 raw_dim = prev_output.NumCols();
  KALDI_ASSERT(input->NumCols() == raw_dim * num_splice);
  for (int32 splice_idx = 0; splice_idx < num_splice; splice_idx++) {
    int32 frame_offset = raw_splicing[splice_idx];
    int32 num_frames = input->NumRows();  // the smaller #frames.
    SubMatrix<BaseFloat> dest(*input, 0, num_frames,
                              raw_dim * splice_idx, raw_dim),
        src(prev_output, frame_offset, num_frames,
            0, raw_dim);
    dest.CopyFromMat(src);
  }
}


// This is like SpliceData but for backpropagating derivatives.
void NnetComputer::UnSpliceDerivatives(
    const MatrixBase<BaseFloat> &input_deriv,
    int32 c, // component index of component for which "input_deriv" is the
             // derivative w.r.t. the input
    MatrixBase<BaseFloat> *prev_output_deriv) const {
  const std::vector<int32> &raw_splicing = nnet_.RawSplicingForComponent(c);
  int32 left_context = -raw_splicing.front(),
       right_context = raw_splicing.back();
  if (left_context == 0 && right_context == 0) {
    prev_output_deriv->CopyFromMat(input_deriv);
    return;
  }
  KALDI_ASSERT(input_deriv.NumRows() ==
               prev_output_deriv->NumRows() - left_context - right_context);
  prev_output_deriv->SetZero();
  int32 num_splice = raw_splicing.size();
  int32 raw_dim = prev_output_deriv->NumCols();
  KALDI_ASSERT(input_deriv.NumCols() == raw_dim * num_splice);
  for (int32 splice_idx = 0; splice_idx < num_splice; splice_idx++) {
    int32 frame_offset = raw_splicing[splice_idx];
    int32 num_frames = input_deriv.NumRows(); // the smaller #frames.
    SubMatrix<BaseFloat> input_deriv_part(input_deriv, 0, num_frames,
                                          raw_dim * splice_idx, raw_dim),
        prev_output_deriv_part(*prev_output_deriv, frame_offset, num_frames,
                               0, raw_dim);
    prev_output_deriv_part.AddMat(1.0, input_deriv_part);
  }
}

// Splices together the input to the whole neural net, from
// the possibly-padded (but not spliced) feature data, into
// "spliced_input" which is the input to the first layer.
void NnetComputer::SpliceFirstLayerInput(Matrix<BaseFloat> *spliced_input) const {
  const std::vector<int32> &raw_splicing = nnet_.RawSplicingForComponent(0);
  int32 left_context = -raw_splicing.front(),
       right_context = raw_splicing.back();
  const MatrixBase<BaseFloat> &raw_feats = forward_data_[0];
  int32 num_splice = raw_splicing.size(),
       feature_dim = nnet_.FeatureDim(),
       speaker_dim = nnet_.SpeakerDim(),
        num_frames = raw_feats.NumRows(),
smaller_num_frames = num_frames - left_context - right_context;
  int32 total_feature_dim = num_splice * feature_dim + speaker_dim;
  KALDI_ASSERT(nnet_.GetComponent(0).InputDim() == total_feature_dim);
  spliced_input->Resize(smaller_num_frames, total_feature_dim);
  SubMatrix<BaseFloat> feature_part(*spliced_input, 0, smaller_num_frames,
                                    0, num_splice * feature_dim);
  SpliceData(raw_feats, 0, &feature_part); // Splice the "real" part of the
  // features...

  SubMatrix<BaseFloat> speaker_part(*spliced_input, 0, smaller_num_frames,
                                    0, speaker_dim);
  speaker_part.CopyRowsFromVec(spk_info_);
}

/// This is the forward part of the computation.
void NnetComputer::Propagate(bool will_do_backprop) {
  for (int32 c = 0; c < nnet_.NumComponents(); c++) {
    const Component &component = nnet_.GetComponent(c);
    const std::vector<int32> &raw_splicing =
        nnet_.RawSplicingForComponent(c);
    Matrix<BaseFloat> &prev_output = forward_data_[c];
    int32 left_context = -raw_splicing.front(),
         right_context = raw_splicing.back();
    Matrix<BaseFloat> temp_input; // input to this layer.
    bool use_temp_input = (c == 0 || left_context != 0 || right_context != 0);
    if (use_temp_input) {
      if (c == 0) {
        SpliceFirstLayerInput(&temp_input);
      } else {
        temp_input.Resize(prev_output.NumRows() - left_context - right_context,
                          component.InputDim());
        SpliceData(prev_output, c, &temp_input);
      }
    }
    const Matrix<BaseFloat> &input = (use_temp_input ? temp_input : prev_output);
    Matrix<BaseFloat> &output = forward_data_[c+1];
    component.Propagate(input, &output);
    bool need_last_output =
        will_do_backprop &&
        (c>0 && nnet_.GetComponent(c-1).BackpropNeedsOutput()) ||
        component.BackpropNeedsInput();
    if (!need_last_output)
      forward_data_[c].Resize(0, 0); // We won't need this data.
  }
}

BaseFloat NnetComputer::ComputeLastLayerDeriv(const std::vector<int32> &labels,
                                              Matrix<BaseFloat> *deriv) const {
  int32 num_components = nnet_.NumComponents();
  // We're writing this code in such a way that if you want to
  // incorporate per-frame weights later on, it's easy.
  double tot_objf = 0.0, tot_weight = 0.0;
  const BaseFloat floor = 1.0e-20; // Avoids division by zero.
  const Matrix<BaseFloat> &last_layer_output = forward_data_[num_components];
  int32 num_frames = last_layer_output.NumRows(),
          num_pdfs = last_layer_output.NumCols();
  deriv->Resize(num_frames, num_pdfs); // will zero it.
  for (int32 i = 0; i < deriv->NumRows(); i++) {
    const BaseFloat weight = 1.0;
    int32 label = labels[i];
    KALDI_ASSERT(label >= 0 && label < num_pdfs);
    BaseFloat this_prob = last_layer_output(i, label);
    if (this_prob < floor) {
      KALDI_WARN << "Probability is " << this_prob << ", flooring to "
                 << floor;
      this_prob = floor;
    }
    tot_objf += weight * log(this_prob);
    tot_weight += weight;
    (*deriv)(i, label) = weight / this_prob;
  }
  KALDI_VLOG(4) << "Objective function is " << (tot_objf/tot_weight) << " over "
                << tot_weight << " samples.";
  return tot_objf;  
}


BaseFloat NnetComputer::Backprop(const std::vector<int32> &labels) {
  BaseFloat tot_weight = labels.size();
  Matrix<BaseFloat> temp_deriv;
  BaseFloat ans = ComputeLastLayerDeriv(labels, &temp_deriv);
  
  for (int32 c = nnet_.NumComponents() - 1; c >= 0; c--) {
    const Component &component = nnet_.GetComponent(c);
    Component *component_to_update = (nnet_to_update_ == NULL ? NULL :
                                      &(nnet_to_update_->GetComponent(c)));
    const std::vector<int32> &raw_splicing =
        nnet_.RawSplicingForComponent(c);
    int32 left_context = -raw_splicing.front(),
         right_context = raw_splicing.back();
    const Matrix<BaseFloat> &output = forward_data_[c+1],
                      &output_deriv = temp_deriv;
    KALDI_ASSERT(SameDim(output, output_deriv));
    
    bool backprop_needs_input = component.BackpropNeedsInput();
    bool do_splicing = (c == 0 || left_context != 0 || right_context != 0);
    Matrix<BaseFloat> temp_input; // input to this layer.
    if (do_splicing && backprop_needs_input) {
      if (c == 0) {
        SpliceFirstLayerInput(&temp_input);
      } else {
        temp_input.Resize(output_deriv.NumRows(),
                          component.InputDim());
        SpliceData(forward_data_[c], c, &temp_input);
      }
    }
    Matrix<BaseFloat> &input = (do_splicing ? temp_input : forward_data_[c]);
    // Note: at this point, if !backprop_needs_input, "input" will be empty.
    
    Matrix<BaseFloat> input_deriv(output.NumRows(), component.InputDim());
    
    component.Backprop(input, output, output_deriv, tot_weight,
                       component_to_update, &input_deriv);

    if (!do_splicing) {
      // Overwrite temp_deriv with input_deriv, which
      // is derivative of objective function w.r.t. output previous layer    
      temp_deriv = input_deriv;
    } else {
      UnSpliceDerivatives(input_deriv, c, &temp_deriv); // This does the reverse of
      // SpliceData().
    }
  }
  return ans;
}

void NnetComputation(const Nnet &nnet,
                     const MatrixBase<BaseFloat> &input,  // features
                     const VectorBase<BaseFloat> &spk_info,
                     bool pad_input,
                     MatrixBase<BaseFloat> *output) {
  NnetComputer nnet_computer(nnet, input, spk_info, pad_input, NULL);
  nnet_computer.Propagate(false);
  const MatrixBase<BaseFloat> &temp_output(nnet_computer.GetOutput());
  if (!SameDim(*output, temp_output)) {
    KALDI_ERR << "Mismatch in size of output (issue with padding, or "
              << "mismatch in #pdfs?) " << output->NumRows() << " by "
              << output->NumCols() << " vs. " << temp_output.NumRows()
              << " by " << temp_output.NumCols();
  }
  output->CopyFromMat(temp_output);  
}

BaseFloat NnetGradientComputation(const Nnet &nnet,
                                  const MatrixBase<BaseFloat> &input,
                                  const VectorBase<BaseFloat> &spk_info,
                                  bool pad_input,
                                  std::vector<int32> &labels,
                                  Nnet *nnet_to_update) {
  NnetComputer nnet_computer(nnet, input, spk_info, pad_input, nnet_to_update);
  nnet_computer.Propagate(true);
  return nnet_computer.Backprop(labels);
}


  
  
} // namespace
