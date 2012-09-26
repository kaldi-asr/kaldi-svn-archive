// nnet/nnet-component.cc

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

#include "nnet-cpu/nnet-component.h"

namespace kaldi {

// static
Component* Component::ReadNew(std::istream &is, bool binary) {
  int i = PeekToken(is, binary), c = static_cast<char>(i);
  Component *ans = NULL;
  if (i == -1) {
    KALDI_ERR << "Unexpected end of file";
  } else if (c == 'S') {
    ans = new SigmoidComponent();
  } else if (c == 'T') {
    ans = new TanhComponent();
  } else if (c == 'A') {
    ans = new AffineComponent();
  } else if (c == 'M') {
    ans = new MixtureProbComponent();
  } else if (c == 'B') {
    ans = new BlockAffineComponent();
  } else if (c == 'P') {
    ans = new PermuteComponent();
  } else if (c == 's') {
    ans = new SoftmaxComponent();
  } else {
    KALDI_ERR << "Unexpected character " << CharToString(c);
  }
  ans->Read(is, binary);
  return ans;
}

// static
Component* Component::NewFromString(const std::string &initializer_line) {
  std::istringstream istr(initializer_line);
  std::string component_type; // e.g. "SigmoidComponent".
  istr >> component_type >> std::ws; 
  std::string args = istr.str();
  Component *ans = NULL;
  if (component_type == "SigmoidComponent") {
    ans = new SigmoidComponent();
  } else if (component_type == "TanhComponent") {
    ans = new TanhComponent();
  } else if (component_type == "SoftmaxComponent") {
    ans = new SoftmaxComponent();
  } else if (component_type == "AffineComponent") {
    ans = new AffineComponent();
  } else if (component_type == "MixtureProbComponent") {
    ans = new MixtureProbComponent();
  } else if (component_type == "BlockAffineComponent") {
    ans = new BlockAffineComponent();
  } else if (component_type == "PermuteComponent") {
    ans = new PermuteComponent();
  } else {
    KALDI_ERR << "Bad initializer line for component: "
              << initializer_line;
  }
  ans->InitFromString(args);
  return ans;
}

Component *PermuteComponent::Copy() const {
  PermuteComponent *ans = new PermuteComponent();
  ans->reorder_ = reorder_;
  return ans;
}

void NonlinearComponent::Read(std::istream &is, bool binary) {
  std::ostringstream ostr_beg, ostr_end;
  ostr_beg << "<" << Type() << ">"; // e.g. "<SigmoidComponent>"
  ostr_end << "</" << Type() << ">"; // e.g. "</SigmoidComponent>"
  ExpectToken(is, binary, ostr_beg.str());
  ExpectToken(is, binary, "<Dim>");
  ReadBasicType(is, binary, &dim_); // Read dimension.
  ExpectToken(is, binary, ostr_end.str());  
}

void NonlinearComponent::Write(std::ostream &os, bool binary) const {
  std::ostringstream ostr_beg, ostr_end;
  ostr_beg << "<" << Type() << ">"; // e.g. "<SigmoidComponent>"
  ostr_end << "</" << Type() << ">"; // e.g. "</SigmoidComponent>"
  WriteToken(os, binary, ostr_beg.str());
  WriteToken(os, binary, "<Dim>");
  WriteBasicType(os, binary, dim_);
  WriteToken(os, binary, ostr_end.str());  
}

void NonlinearComponent::InitFromString(std::string args) {
  std::istringstream istr(args);
  istr >> dim_ >> std::ws;
  if (!istr || !istr.eof() || dim_ <= 0) {
    KALDI_ERR << "Invalid initializer for layer of type "
              << Type() << ": \"" << args << "\"";
  }
}

void SigmoidComponent::Propagate(const MatrixBase<BaseFloat> &in,
                                 MatrixBase<BaseFloat> *out) const {
  KALDI_ASSERT(SameDim(in, *out));
  int32 num_rows = in.NumRows(), num_cols = in.NumCols();
  for(int32 r = 0; r < num_rows; r++) {
    const BaseFloat *in_data = in.RowData(r),
        *in_data_end = in_data + num_cols;
    BaseFloat *out_data = out->RowData(r);
    for (; in_data != in_data_end; ++in_data, ++out_data) {
      if (*in_data > 0.0) {
        *out_data = 1.0 / (1.0 + exp(- *in_data));
      } else { // avoid exponentiating positive number; instead,
        // use 1/(1+exp(-x)) = exp(x) / (exp(x)+1)
        BaseFloat f = exp(*in_data);
        *out_data = f / (f + 1.0);
      }
    }
  }
}

void SigmoidComponent::Backprop(const MatrixBase<BaseFloat> &in_value,
                                const MatrixBase<BaseFloat> &out_value,
                                const MatrixBase<BaseFloat> &out_deriv,
                                BaseFloat tot_weight,
                                Component *to_update,
                                MatrixBase<BaseFloat> *in_deriv) const {
  // we ignore in_value and to_update.

  // The element by element equation would be:
  // in_deriv = out_deriv * out_value * (1.0 - out_value);
  // We can accomplish this via calls to the matrix library.

  in_deriv->Set(1.0);
  in_deriv->AddMat(-1.0, out_value);
  // now in_deriv = 1.0 - out_value [element by element]
  in_deriv->MulElements(out_value);
  // now in_deriv = out_value * (1.0 - out_value) [element by element]
  in_deriv->MulElements(out_deriv);
  // now in_deriv = out_deriv * out_value * (1.0 - out_value) [element by element]
}


void TanhComponent::Propagate(const MatrixBase<BaseFloat> &in,
                              MatrixBase<BaseFloat> *out) const {
  // Apply tanh function to each element of the output...
  // the tanh function may be written as -1 + ( 2 / (1 + e^{-2 x})),
  // which is a scaled and shifted sigmoid.
  
  KALDI_ASSERT(SameDim(in, *out));
  int32 num_rows = in.NumRows(), num_cols = in.NumCols();
  for(int32 r = 0; r < num_rows; r++) {
    const BaseFloat *in_data = in.RowData(r),
        *in_data_end = in_data + num_cols;
    BaseFloat *out_data = out->RowData(r);
    for (; in_data != in_data_end; ++in_data, ++out_data) {
      if (*in_data > 0.0) {
        *out_data = -1.0 + 2.0 / (1.0 + exp(-2.0 * *in_data));
      } else {
        *out_data = 1.0 - 2.0 / (1.0 + exp(2.0 * *in_data));
      }
    }
  }
}

void TanhComponent::Backprop(const MatrixBase<BaseFloat> &, // in_value
                             const MatrixBase<BaseFloat> &out_value,
                             const MatrixBase<BaseFloat> &out_deriv,
                             BaseFloat, //  tot_weight,
                             Component *, // to_update
                             MatrixBase<BaseFloat> *in_deriv) const {
  /*
    Note on the derivative of the tanh function:
    tanh'(x) = sech^2(x) = -(tanh(x)+1) (tanh(x)-1) = 1 - tanh^2(x)

    The element by element equation of what we're doing would be:
    in_deriv = out_deriv * (1.0 - out_value^2).
    We can accomplish this via calls to the matrix library. */    
  in_deriv->CopyFromMat(out_value);
  in_deriv->ApplyPow(2.0);
  in_deriv->Scale(-1.0);
  in_deriv->Add(1.0); // now in_deriv = (1.0 - out_value^2).
  in_deriv->MulElements(out_deriv);
}  

void SoftmaxComponent::Propagate(const MatrixBase<BaseFloat> &in,
                                 MatrixBase<BaseFloat> *out) const {
  // Apply softmax function to each row of the output...
  // for that row, we do
  // x_i = exp(x_i) / sum_j exp(x_j).
  
  out->CopyFromMat(in);
  int32 num_rows = out->NumRows();
  for(int32 r = 0; r < num_rows; r++) {
    SubVector<BaseFloat> row(*out, r);
    row.ApplySoftMax();
  }
}

void SoftmaxComponent::Backprop(const MatrixBase<BaseFloat> &in_value,
                                const MatrixBase<BaseFloat> &out_value,
                                const MatrixBase<BaseFloat> &out_deriv,
                                BaseFloat, //  tot_weight
                                Component *to_update,
                                MatrixBase<BaseFloat> *in_deriv) const {
  /*
    Note on the derivative of the softmax function: let it be
    p_i = exp(x_i) / sum_i exp_i
    The [matrix-valued] Jacobian of this function is
    diag(p) - p p^T
    Let the derivative vector at the output be e, and at the input be
    d.  We have
    d = diag(p) e - p (p^T e).
    d_i = p_i e_i - p_i (p^T e).    
  */
  KALDI_ASSERT(SameDim(out_value, out_deriv) && SameDim(out_value, *in_deriv));
  const MatrixBase<BaseFloat> &P(out_value), &E(out_deriv);
  MatrixBase<BaseFloat> &D (*in_deriv);
  
  for (int32 r = 0; r < P.NumRows(); r++) {
    SubVector<BaseFloat> p(P, r), e(E, r), d(D, r);
    d.AddVecVec(1.0, p, e, 0.0); // d_i = p_i e_i.
    BaseFloat pT_e = VecVec(p, e); // p^T e.
    d.AddVec(-pT_e, p); // d_i -= (p^T e) p_i
  }
}

void AffineComponent::SetZero(bool treat_as_gradient) {
  if (treat_as_gradient) {
    SetLearningRate(1.0);
    SetL2Penalty(0.0);
  }
  linear_params_.SetZero();
  bias_params_.SetZero();
}

void AffineComponent::PerturbParams(BaseFloat stddev) {
  Matrix<BaseFloat> temp_linear_params(linear_params_);
  temp_linear_params.SetRandn();
  linear_params_.AddMat(stddev, temp_linear_params);
  
  Vector<BaseFloat> temp_bias_params(bias_params_);
  temp_bias_params.SetRandn();
  bias_params_.AddVec(stddev, temp_bias_params);
}

Component* AffineComponent::Copy() const {
  AffineComponent *ans = new AffineComponent();
  ans->learning_rate_ = learning_rate_;
  ans->l2_penalty_ = l2_penalty_;
  ans->linear_params_ = linear_params_;
  ans->bias_params_ = bias_params_;
  return ans;
}

BaseFloat AffineComponent::DotProduct(const UpdatableComponent &other_in) const {
  const AffineComponent *other =
      dynamic_cast<const AffineComponent*>(&other_in);
  return TraceMatMat(linear_params_, other->linear_params_, kTrans)
      + VecVec(bias_params_, other->bias_params_);
}

void AffineComponent::Init(BaseFloat learning_rate, BaseFloat l2_penalty,
                           int32 input_dim, int32 output_dim,
                           BaseFloat param_stddev) {
  UpdatableComponent::Init(learning_rate, l2_penalty);
  linear_params_.Resize(output_dim, input_dim);
  bias_params_.Resize(output_dim);
  KALDI_ASSERT(output_dim > 0 && input_dim > 0 && param_stddev >= 0.0);
  linear_params_.SetRandn(); // sets to random normally distributed noise.
  linear_params_.Scale(param_stddev);
  bias_params_.SetRandn();
  bias_params_.Scale(param_stddev);
}

void AffineComponent::InitFromString(std::string args) {
  std::istringstream is(args);
  BaseFloat learning_rate, l2_penalty, param_stddev;
  int32 input_dim, output_dim;
  is >> learning_rate >> l2_penalty >> input_dim >> output_dim
     >> param_stddev >> std::ws;
  if (!is || !is.eof())
    KALDI_ERR << "Bad initializer " << args;
  Init(learning_rate, l2_penalty, input_dim, output_dim, param_stddev);
}
void AffineComponent::Propagate(const MatrixBase<BaseFloat> &in,
                                MatrixBase<BaseFloat> *out) const {
  // No need for asserts as they'll happen within the matrix operations.
  out->CopyRowsFromVec(bias_params_); // copies bias_params_ to each row
  // of *out.
  out->AddMatMat(1.0, in, kNoTrans, linear_params_, kTrans, 1.0);
}

void AffineComponent::Backprop(const MatrixBase<BaseFloat> &in_value,
                               const MatrixBase<BaseFloat> &, // out_value
                               const MatrixBase<BaseFloat> &out_deriv,
                               BaseFloat tot_weight,
                               Component *to_update_in,
                               MatrixBase<BaseFloat> *in_deriv) const {
  AffineComponent *to_update = dynamic_cast<AffineComponent*>(to_update_in);
  // Propagate the derivative back to the input.
  in_deriv->AddMatMat(1.0, out_deriv, kNoTrans, linear_params_, kNoTrans,
                      0.0);

  if (to_update) {
    BaseFloat old_weight = to_update->OldWeight(tot_weight);
    // Next update the model (must do this 2nd so the derivatives we propagate
    // are accurate, in case this == to_update_in.)
    // add the sum of the rows of out_deriv, to the bias_params_.
    to_update->bias_params_.AddRowSumMat(to_update->learning_rate_, out_deriv,
                                         old_weight);
    to_update->linear_params_.AddMatMat(to_update->learning_rate_,
                                        out_deriv, kTrans, in_value, kNoTrans,
                                        old_weight);
  }
}

void AffineComponent::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "<AffineComponent>");
  ExpectToken(is, binary, "<LearningRate>");
  ReadBasicType(is, binary, &learning_rate_);
  ExpectToken(is, binary, "<L2Penalty>");
  ReadBasicType(is, binary, &l2_penalty_);
  ExpectToken(is, binary, "<LinearParams>");
  linear_params_.Read(is, binary);
  ExpectToken(is, binary, "<BiasParams>");
  bias_params_.Read(is, binary);
  ExpectToken(is, binary, "</AffineComponent>");  
}

void AffineComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<AffineComponent>");
  WriteToken(os, binary, "<LearningRate>");
  WriteBasicType(os, binary, learning_rate_);
  WriteToken(os, binary, "<L2Penalty>");
  WriteBasicType(os, binary, l2_penalty_);
  WriteToken(os, binary, "<LinearParams>");
  linear_params_.Write(os, binary);
  WriteToken(os, binary, "<BiasParams>");
  bias_params_.Write(os, binary);
  WriteToken(os, binary, "</AffineComponent>");  
}

void BlockAffineComponent::SetZero(bool treat_as_gradient) {
  if (treat_as_gradient) {
    SetLearningRate(1.0);
    SetL2Penalty(0.0);
  }
  linear_params_.SetZero();
  bias_params_.SetZero();
}

void BlockAffineComponent::PerturbParams(BaseFloat stddev) {
  Matrix<BaseFloat> temp_linear_params(linear_params_);
  temp_linear_params.SetRandn();
  linear_params_.AddMat(stddev, temp_linear_params);
  
  Vector<BaseFloat> temp_bias_params(bias_params_);
  temp_bias_params.SetRandn();
  bias_params_.AddVec(stddev, temp_bias_params);
}

BaseFloat BlockAffineComponent::DotProduct(
    const UpdatableComponent &other_in) const {
  const BlockAffineComponent *other =
      dynamic_cast<const BlockAffineComponent*>(&other_in);
  return TraceMatMat(linear_params_, other->linear_params_, kTrans)
      + VecVec(bias_params_, other->bias_params_);
}

Component* BlockAffineComponent::Copy() const {
  BlockAffineComponent *ans = new BlockAffineComponent();
  ans->learning_rate_ = learning_rate_;
  ans->l2_penalty_ = l2_penalty_;
  ans->linear_params_ = linear_params_;
  ans->bias_params_ = bias_params_;
  ans->num_blocks_ = num_blocks_;
  return ans;
}

void BlockAffineComponent::Propagate(const MatrixBase<BaseFloat> &in,
                                     MatrixBase<BaseFloat> *out) const {
  out->CopyRowsFromVec(bias_params_); // copies bias_params_ to each row
  // of *out.

  // The matrix has a block structure where each matrix has input dim
  // (#rows) equal to input_block_dim.  The blocks are stored in linear_params_
  // as [ M
  //      N
  //      O ] but we actually treat it as:
  // [ M 0 0
  //   0 N 0
  //   0 0 O ]
  int32 input_block_dim = linear_params_.NumCols(),
       output_block_dim = linear_params_.NumRows() / num_blocks_,
             num_frames = in.NumRows();
  KALDI_ASSERT(in.NumCols() == input_block_dim * num_blocks_);
  KALDI_ASSERT(out->NumCols() == output_block_dim * num_blocks_);
  KALDI_ASSERT(in.NumRows() == out->NumRows());

  out->CopyRowsFromVec(bias_params_); // copies bias_params_ to each row
  // of *out.
  
  for (int32 b = 0; b < num_blocks_; b++) {
    SubMatrix<BaseFloat> in_block(in, 0, num_frames,
                                  b * input_block_dim, input_block_dim),
        out_block(*out, 0, num_frames,
                  b * output_block_dim, output_block_dim),
        param_block(linear_params_,
                    b * output_block_dim, output_block_dim,
                    0, input_block_dim);
    out_block.AddMatMat(1.0, in_block, kNoTrans, param_block, kTrans, 1.0);
  }
}

void BlockAffineComponent::Backprop(
    const MatrixBase<BaseFloat> &in_value,
    const MatrixBase<BaseFloat> &, // out_value
    const MatrixBase<BaseFloat> &out_deriv,
    BaseFloat tot_weight,
    Component *to_update_in,
    MatrixBase<BaseFloat> *in_deriv) const {
  // This code mirrors the code in Propagate().
  int32 num_frames = in_value.NumRows();
  BlockAffineComponent *to_update = dynamic_cast<BlockAffineComponent*>(
      to_update_in);
  
  int32 input_block_dim = linear_params_.NumCols(),
       output_block_dim = linear_params_.NumRows() / num_blocks_;
  KALDI_ASSERT(in_value.NumCols() == input_block_dim * num_blocks_);
  KALDI_ASSERT(out_deriv.NumCols() == output_block_dim * num_blocks_);

  // add the sum of the rows of out_deriv, to the bias_params_.
  if (to_update)
    to_update->bias_params_.AddRowSumMat(to_update->learning_rate_, out_deriv,
                                         to_update->OldWeight(tot_weight));
  
  for (int32 b = 0; b < num_blocks_; b++) {
    SubMatrix<BaseFloat> in_value_block(in_value, 0, num_frames,
                                        b * input_block_dim,
                                        input_block_dim),
        in_deriv_block(*in_deriv, 0, num_frames,
                       b * input_block_dim, input_block_dim),
        out_deriv_block(out_deriv, 0, num_frames,
                        b * output_block_dim, output_block_dim),
        param_block(linear_params_,
                    b * output_block_dim, output_block_dim,
                    0, input_block_dim);

    // Propagate the derivative back to the input.
    in_deriv_block.AddMatMat(1.0, out_deriv_block, kNoTrans,
                             param_block, kNoTrans, 0.0);
    
    if (to_update) {
      SubMatrix<BaseFloat> param_block_to_update(
          to_update->linear_params_,
          b * output_block_dim, output_block_dim,
          0, input_block_dim);
      // Update the parameters.
      param_block_to_update.AddMatMat(
          to_update->learning_rate_,
          out_deriv_block, kTrans, in_value_block, kNoTrans,
          to_update->OldWeight(tot_weight));
    }
  }  
}

BaseFloat UpdatableComponent::OldWeight(BaseFloat num_frames) const {
  return std::pow(static_cast<BaseFloat>(1.0 - 2.0 * learning_rate_ * l2_penalty_),
                  static_cast<BaseFloat>(num_frames));
}

void BlockAffineComponent::Init(BaseFloat learning_rate, BaseFloat l2_penalty,
                                int32 input_dim, int32 output_dim,
                                BaseFloat param_stddev, int32 num_blocks) {
  UpdatableComponent::Init(learning_rate, l2_penalty);
  KALDI_ASSERT(output_dim > 0 && input_dim > 0 && param_stddev >= 0.0);
  KALDI_ASSERT(input_dim % num_blocks == 0 && output_dim % num_blocks == 0);

  linear_params_.Resize(output_dim, input_dim / num_blocks);
  bias_params_.Resize(output_dim);

  linear_params_.SetRandn(); // sets to random normally distributed noise.
  linear_params_.Scale(param_stddev);
  bias_params_.SetRandn();
  bias_params_.Scale(param_stddev);
  num_blocks_ = num_blocks;
}

void BlockAffineComponent::InitFromString(std::string args) {
  std::istringstream is(args);
  BaseFloat learning_rate, l2_penalty, param_stddev;
  int32 input_dim, output_dim, num_blocks;
  is >> learning_rate >> l2_penalty >> input_dim >> output_dim
     >> param_stddev >> num_blocks >> std::ws;
  if (!is || !is.eof())
    KALDI_ERR << "Bad initializer " << args;
  Init(learning_rate, l2_penalty, input_dim, output_dim,
       param_stddev, num_blocks);
}

void BlockAffineComponent::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "<BlockAffineComponent>");
  ExpectToken(is, binary, "<LearningRate>");
  ReadBasicType(is, binary, &learning_rate_);
  ExpectToken(is, binary, "<L2Penalty>");
  ReadBasicType(is, binary, &l2_penalty_);
  ExpectToken(is, binary, "<NumBlocks>");
  ReadBasicType(is, binary, &num_blocks_);
  ExpectToken(is, binary, "<LinearParams>");
  linear_params_.Read(is, binary);
  ExpectToken(is, binary, "<BiasParams>");
  bias_params_.Read(is, binary);
  ExpectToken(is, binary, "</BlockAffineComponent>");  
}

void BlockAffineComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<BlockAffineComponent>");
  WriteToken(os, binary, "<LearningRate>");
  WriteBasicType(os, binary, learning_rate_);
  WriteToken(os, binary, "<L2Penalty>");
  WriteBasicType(os, binary, l2_penalty_);
  WriteToken(os, binary, "<NumBlocks>");
  WriteBasicType(os, binary, num_blocks_);
  WriteToken(os, binary, "<LinearParams>");
  linear_params_.Write(os, binary);
  WriteToken(os, binary, "<BiasParams>");
  bias_params_.Write(os, binary);
  WriteToken(os, binary, "</BlockAffineComponent>");  
}


void PermuteComponent::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "<PermuteComponent>");
  ReadIntegerVector(is, binary, &reorder_);
  ExpectToken(is, binary, "</PermuteComponent>");
}

void PermuteComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<PermuteComponent>");
  WriteIntegerVector(os, binary, reorder_);
  WriteToken(os, binary, "</PermuteComponent>");
}

void PermuteComponent::Init(int32 dim) {
  KALDI_ASSERT(dim > 0);
  reorder_.resize(dim);
  for (int32 i = 0; i < dim; i++) reorder_[i] = i;
  std::random_shuffle(reorder_.begin(), reorder_.end());
}

void PermuteComponent::InitFromString(std::string args) {
  std::istringstream is(args);
  int32 dim;
  is >> dim >> std::ws;
  if (!is || !is.eof())
    KALDI_ERR << "Bad initializer: " << args;
  Init(dim);
}

void PermuteComponent::Propagate(const MatrixBase<BaseFloat> &in,
                                 MatrixBase<BaseFloat> *out) const {
  KALDI_ASSERT(SameDim(in, *out) && in.NumCols() == OutputDim());
  
  int32 num_rows = in.NumRows(), num_cols = in.NumCols();
  for (int32 r = 0; r < num_rows; r++) {
    const BaseFloat *in_data = in.RowData(r);
    BaseFloat *out_data = out->RowData(r);
    for (int32 c = 0; c < num_cols; c++)
      out_data[reorder_[c]] = in_data[c];
  }
}

void PermuteComponent::Backprop(const MatrixBase<BaseFloat> &in_value,
                                const MatrixBase<BaseFloat> &out_value,
                                const MatrixBase<BaseFloat> &out_deriv,
                                BaseFloat, //  tot_weight
                                Component *to_update,
                                MatrixBase<BaseFloat> *in_deriv) const {
  KALDI_ASSERT(SameDim(out_deriv, *in_deriv) &&
               out_deriv.NumCols() == OutputDim());
  
  int32 num_rows = in_deriv->NumRows(), num_cols = in_deriv->NumCols();
  for (int32 r = 0; r < num_rows; r++) {
    const BaseFloat *out_deriv_data = out_deriv.RowData(r);
    BaseFloat *in_deriv_data = in_deriv->RowData(r);
    for (int32 c = 0; c < num_cols; c++)
      in_deriv_data[c] = out_deriv_data[reorder_[c]];
  }
}


void MixtureProbComponent::PerturbParams(BaseFloat stddev) {
  for (size_t i = 0; i < params_.size(); i++) {
    Matrix<BaseFloat> temp_params(params_[i]);
    temp_params.SetRandn();
    params_[i].AddMat(stddev, temp_params);
  }
}


Component* MixtureProbComponent::Copy() const {
  MixtureProbComponent *ans = new MixtureProbComponent();
  ans->learning_rate_ = learning_rate_;
  ans->l2_penalty_ = l2_penalty_;
  ans->params_ = params_;
  ans->input_dim_ = input_dim_;
  ans->output_dim_ = output_dim_;
  ans->is_gradient_ = is_gradient_;
  return ans;
}


BaseFloat MixtureProbComponent::DotProduct(
    const UpdatableComponent &other_in) const {
  const MixtureProbComponent *other =
      dynamic_cast<const MixtureProbComponent*>(&other_in);
  BaseFloat ans = 0.0;
  KALDI_ASSERT(params_.size() == other->params_.size());
  for (size_t i = 0; i < params_.size(); i++)
    ans += TraceMatMat(params_[i], other->params_[i], kTrans);
  return ans;
}

void MixtureProbComponent::Init(BaseFloat learning_rate, BaseFloat l2_penalty,
                                BaseFloat diag_element,
                                const std::vector<int32> &sizes) {
  UpdatableComponent::Init(learning_rate, l2_penalty);
  is_gradient_ = false;
  input_dim_ = 0;
  output_dim_ = 0;
  params_.resize(sizes.size());
  KALDI_ASSERT(diag_element > 0.0 && diag_element <= 1.0);
  // Initialize to a block-diagonal matrix consisting of a series of square
  // blocks, with sizes specified in "sizes".  Note: each block will typically
  // correspond to a number of clustered states, so this whole thing implements
  // an idea similar to the "state clustered tied mixture" system.
  for (size_t i = 0; i < sizes.size(); i++) {
    KALDI_ASSERT(sizes[i] > 0);
    int32 size = sizes[i];
    params_[i].Resize(size, size);
    input_dim_ += size;
    output_dim_ += size;
    if (size == 1) {
      params_[i](0,0) = 1.0;
    } else {
      BaseFloat off_diag_element = (1.0 - diag_element) / (size - 0.999999);
      params_[i].Set(off_diag_element);
      for (int32 j = 0; j < size; j++)
        params_[i](j, j) = diag_element;
    }
  }
}  

void MixtureProbComponent::InitFromString(std::string args) {
  std::istringstream is(args);
  BaseFloat learning_rate, l2_penalty, diag_element;
  is >> learning_rate >> l2_penalty >> diag_element;
  if (!is)
    KALDI_ERR << "Invalid initializer";
  std::vector<int32> sizes;
  int32 i;
  while (is >> i) sizes.push_back(i);
  is >> std::ws;
  if (!is.eof() || sizes.empty())
    KALDI_ERR << "Invalid initializer";
  Init(learning_rate, l2_penalty, diag_element, sizes);
}

void MixtureProbComponent::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "<MixtureProbComponent>");
  ExpectToken(is, binary, "<LearningRate>");
  ReadBasicType(is, binary, &learning_rate_);
  ExpectToken(is, binary, "<L2Penalty>");
  ReadBasicType(is, binary, &l2_penalty_);
  ExpectToken(is, binary, "<Params>");
  int32 size;
  ReadBasicType(is, binary, &size);
  input_dim_ = 0;
  output_dim_ = 0;
  KALDI_ASSERT(size >= 0);
  params_.resize(size);
  for (int32 i = 0; i < size; i++) {
    params_[i].Read(is, binary);
    input_dim_ += params_[i].NumCols();
    output_dim_ += params_[i].NumRows();
  }        
  ExpectToken(is, binary, "<IsGradient>");
  ReadBasicType(is, binary, &is_gradient_);
  ExpectToken(is, binary, "</MixtureProbComponent>");  
}

void MixtureProbComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<MixtureProbComponent>");
  WriteToken(os, binary, "<LearningRate>");
  WriteBasicType(os, binary, learning_rate_);
  WriteToken(os, binary, "<L2Penalty>");
  WriteBasicType(os, binary, l2_penalty_);
  WriteToken(os, binary, "<Params>");
  int32 size = params_.size();
  WriteBasicType(os, binary, size);
  for (int32 i = 0; i < size; i++)
    params_[i].Write(os, binary);
  WriteToken(os, binary, "<IsGradient>");
  WriteBasicType(os, binary, is_gradient_);
  WriteToken(os, binary, "</MixtureProbComponent>");  
}

void MixtureProbComponent::SetZero(bool treat_as_gradient) {
  if (treat_as_gradient) {
    SetLearningRate(1.0);
    SetL2Penalty(0.0);
    is_gradient_ = true;
  }
  for (size_t i = 0; i < params_.size(); i++)
    params_[i].SetZero();
}

void MixtureProbComponent::Propagate(const MatrixBase<BaseFloat> &in,
                                     MatrixBase<BaseFloat> *out) const {
  KALDI_ASSERT(in.NumRows() == out->NumRows() &&
               in.NumCols() == InputDim() && out->NumCols() == OutputDim());

  int32 num_frames = in.NumRows(),
      input_offset = 0,
     output_offset = 0;

  for (size_t i = 0; i < params_.size(); i++) {
    int32 this_input_dim = params_[i].NumCols(), // input dim of this block.
         this_output_dim = params_[i].NumRows();
    KALDI_ASSERT(this_input_dim > 0 && this_output_dim > 0);
    SubMatrix<BaseFloat> in_block(in, 0, num_frames,
                                  input_offset, this_input_dim),
        out_block(*out, 0, num_frames, output_offset, this_output_dim);
    const Matrix<BaseFloat> &param_block(params_[i]);
    out_block.AddMatMat(1.0, in_block, kNoTrans, param_block, kTrans, 0.0);
    input_offset += this_input_dim;
    output_offset += this_input_dim;   
  }
}

void MixtureProbComponent::Backprop(const MatrixBase<BaseFloat> &in_value,
                                    const MatrixBase<BaseFloat> &,// out_value
                                    const MatrixBase<BaseFloat> &out_deriv,
                                    BaseFloat tot_weight,
                                    Component *to_update_in,
                                    MatrixBase<BaseFloat> *in_deriv) const {
  MixtureProbComponent *to_update = dynamic_cast<MixtureProbComponent*>(
      to_update_in);

  KALDI_ASSERT(in_value.NumRows() == out_deriv.NumRows() &&
               in_value.NumCols() == InputDim() && out_deriv.NumCols() == OutputDim());
  int32 num_frames = in_value.NumRows(),
      input_offset = 0,
     output_offset = 0;
  
  for (size_t i = 0; i < params_.size(); i++) {
    int32 this_input_dim = params_[i].NumCols(), // input dim of this block.
         this_output_dim = params_[i].NumRows();   
    KALDI_ASSERT(this_input_dim > 0 && this_output_dim > 0);
    SubMatrix<BaseFloat> in_value_block(in_value, 0, num_frames,
                                        input_offset, this_input_dim),
        in_deriv_block(*in_deriv, 0, num_frames,
                       input_offset, this_input_dim),
        out_deriv_block(out_deriv, 0, num_frames,
                        output_offset, this_output_dim);
    const Matrix<BaseFloat> &param_block(params_[i]);

    // Propagate gradient back to in_deriv.
    in_deriv_block.AddMatMat(1.0, out_deriv_block, kNoTrans, param_block,
                             kNoTrans, 0.0);

    if (to_update) {
      Matrix<BaseFloat> &param_block_to_update(to_update->params_[i]);
      if (to_update->is_gradient_) { // We're just storing
        // the gradient there, so it's a linear update rule as for any other layer.
        // Note: most likely the learning_rate_ will be 1.0 and OldWeight() will
        // be 1.0 because of zero l2_penalty_.
        KALDI_ASSERT(to_update->OldWeight(tot_weight) == 1.0 &&
                     to_update->learning_rate_ == 1.0);
        param_block_to_update.AddMatMat(1.0, out_deriv_block, kTrans, in_value_block,
                                        kNoTrans, 1.0);
      } else {
        /*
          We do gradient descent in the space of log probabilities.  We enforce the
          sum-to-one constraint; this affects the gradient (I think you can derive
          this using lagrangian multipliers).
        
          For a column c of the matrix, we have a gradient g.
          Let l be the vector of unnormalized log-probs of that row; it has an arbitrary
          offset, but we just choose the point where it coincides with correctly normalized
          log-probs, so for each element:
          l_i = log(c_i).
          The functional relationship between l_i and c_i is:
          c_i = exp(l_i) / sum_j exp(l_j) . [softmax function from l to c.]
          Let h_i be the gradient w.r.t l_i.  We can compute this as follows.  The softmax function
          has a Jacobian equal to diag(c) - c c^T.  We have:
          h = (diag(c) - c c^T)  g
          We do the gradient-descent step on h, and when we convert back to c, we renormalize.
          [note: the renormalization would not even be necessary if the step size were infinitesimal;
          it's only needed due to second-order effects which slightly unnormalize each column.]
        */        
        int32 num_rows = this_output_dim, num_cols = this_input_dim;
        Matrix<BaseFloat> gradient(num_rows, num_cols);
        gradient.AddMatMat(1.0, out_deriv_block, kTrans, in_value_block, kNoTrans,
                           0.0);
        BaseFloat old_weight = to_update->OldWeight(tot_weight);
        for (int32 col = 0; col < num_cols; col++) {
          Vector<BaseFloat> param_col(num_rows);
          param_col.CopyColFromMat(param_block_to_update, col);
          Vector<BaseFloat> log_param_col(param_col);
          log_param_col.ApplyLog(); // note: works even for zero, but may have -inf
          log_param_col.Scale(old_weight); // relates to l2 regularization-- applied at log
          // parameter level.
          for (int32 i = 0; i < num_rows; i++)
            if (log_param_col(i) < -1.0e+20)
              log_param_col(i) = -1.0e+20; // get rid of -inf's,as
          // as we're not sure exactly how BLAS will deal with them.
          Vector<BaseFloat> gradient_col(num_rows);
          gradient_col.CopyColFromMat(gradient, col);
          Vector<BaseFloat> log_gradient(num_rows);
          log_gradient.AddVecVec(1.0, param_col, gradient_col, 0.0); // h <-- diag(c) g.
          BaseFloat cT_g = VecVec(param_col, gradient_col);
          log_gradient.AddVec(-cT_g, param_col); // h -= (c^T g) c .
          log_param_col.AddVec(learning_rate_, log_gradient); // Gradient step,
          // in unnormalized log-prob space.      
          log_param_col.ApplySoftMax(); // Go back to probabilities, renormalizing.
          param_block_to_update.CopyColFromVec(log_param_col, col); // Write back.
        }
      }
    }
    input_offset += this_input_dim;
    output_offset += this_input_dim;   
  }
}



} // namespace kaldi
