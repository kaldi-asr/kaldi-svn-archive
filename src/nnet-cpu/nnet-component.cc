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

void NonlinearComponent::Read(std::istream &is, bool binary) {
  std::ostringstream ostr_beg, ostr_end;
  ostr_beg << "<" << Type() << ">"; // e.g. "<SigmoidLayer>"
  ostr_end << "</" << Type() << ">"; // e.g. "</SigmoidLayer>"
  ExpectToken(is, binary, ostr_beg.str());
  ExpectToken(is, binary, "<Dim>");
  ReadBasicType(is, binary, &dim_); // Read dimension.
  ExpectToken(is, binary, ostr_end.str());  
}

void NonlinearComponent::Write(std::ostream &os, bool binary) const {
  std::ostringstream ostr_beg, ostr_end;
  ostr_beg << "<" << Type() << ">"; // e.g. "<SigmoidLayer>"
  ostr_end << "</" << Type() << ">"; // e.g. "</SigmoidLayer>"
  WriteToken(os, binary, ostr_beg.str());
  WriteToken(os, binary, "<Dim>");
  WriteBasicType(os, binary, dim_);
  WriteToken(os, binary, ostr_end.str());  
}

void NonlinearComponent::InitFromString(std::string args) {
  std::istringstream istr(args);
  istr >> dim_ >> std::ws;
  if (!istr || !istr.str().empty() || dim_ <= 0) {
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
                                Component *to_update,
                                MatrixBase<BaseFloat> *in_deriv) {
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
      } else { // avoid exponentiating positive number; instead,
        // use 1/(1+exp(-x)) = exp(x) / (exp(x)+1)
        BaseFloat f = exp(*in_data);
        *out_data = -1.0 + 2.0 * f / (f + 1.0);
      }
    }
  }
}

void TanhComponent::Backprop(const MatrixBase<BaseFloat> &in_value,
                             const MatrixBase<BaseFloat> &out_value,
                             const MatrixBase<BaseFloat> &out_deriv,
                             Component *to_update,
                             MatrixBase<BaseFloat> *in_deriv) {
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
                                Component *to_update,
                                MatrixBase<BaseFloat> *in_deriv) {
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
  KALDI_ASSERT(in.NumRows() == out->NumRows() &&
               in.NumCols() == linear_params_.NumCols() &&
               out->NumCols() == linear_params_.NumRows());
  out->CopyRowsFromVec(bias_params_); // copies bias_params_ to each row
  // of *out.
  out->AddMatMat(1.0, in, kNoTrans, linear_params_, kTrans, 1.0);
}

void AffineComponent::Backprop(const MatrixBase<BaseFloat> &in_value,
                               const MatrixBase<BaseFloat> &out_value, // dummy
                               const MatrixBase<BaseFloat> &out_deriv, 
                               Component *to_update,
                               MatrixBase<BaseFloat> *in_deriv) {
  // First update the model.
  // add the sum of the rows of out_deriv, to the bias_params_.
  to_update->bias_params_.AddRowSumMat(to_update->learning_rate_,
                                       out_deriv);
  to_update->linear_params_.AddMatMat(to_update->learning_rate_,
                                      out_deriv, kTrans, in_value, kNoTrans,
                                      1.0);
  // Next propagate the derivative back to the input.
  in_deriv->AddMatMat(1.0, out_deriv, kNoTrans, linear_params_, kNoTrans,
                      0.0);
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
                                Component *to_update,
                                MatrixBase<BaseFloat> *in_deriv) {
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


} // namespace kaldi
