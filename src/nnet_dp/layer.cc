// nnet_dp/layer.cc

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

#include "nnet_dp/layer.h"

namespace kaldi {

static void SpliceFrames(const MatrixBase<BaseFloat> &input,
                         MatrixBase<BaseFloat> *spliced_out) {
  // This function [specific to this .cc file] splices
  // the rows of "input" into "spliced_out".  E.g. we put the first
  // 3 rows of "input", spliced together as the 1st row of spliced_out,
  // and then rows 2,3,4 of "input" spliced together as the 2nd row of
  // spliced_out.
  // Actually this is probably not as memory-local as it could be...
  // we'll redo it if a lot of time is spent here. [TODO: try to make this
  // row-by-row, not using sub-matrices; or do it using pointers.]
  int small_dim = input.NumCols(), large_dim = spliced_out->NumCols();
  int num_splice = large_dim / small_dim, num_output_rows = spliced_out->NumRows();
  KALDI_ASSERT (small_dim * num_splice == large_dim); // or not a multiple...
  KALDI_ASSERT(spliced_out->NumRows() == input.NumRows() - num_splice + 1);
  for (int32 s = 0; s < num_splice; s++) {
    SubMatrix<BaseFloat> src(input, s, num_output_rows, 0, small_dim);
    SubMatrix<BaseFloat> dst(*spliced_out, 0, num_output_rows,
                             small_dim * s, small_dim * (s+1));
    dst.CopyFromMat(src);
  }
}

LinearLayer::LinearLayer(int size, BaseFloat diagonal_element,
                         BaseFloat learning_rate):
    learning_rate_(learning_rate) {
  KALDI_ASSERT(learning_rate >= 0.0 && learning_rate <= 1.0); // > 1.0 doesn't make sense.
  // 0 may be used just to disable learning.
  // Note: if diagonal_element == 1.0, learning will never move away from that
  // point (this is due to the preconditioning).
  KALDI_ASSERT(size > 0 && diagonal_element > 0.0 && diagonal_element <= 1.0);
  params_.Resize(size, size);
  double off_diag = (1.0 - diagonal_element) / (size - 1); // value of
  // off-diagonal elements that's needed if we want each column to sum to one.
  params_.Set(off_diag);
  for (int32 i = 0; i < size; i++)
    params_(i, i) = diagonal_element;
}

void LinearLayerStats::AccStats(const MatrixBase<BaseFloat> &input,
                                const MatrixBase<BaseFloat> &output_deriv) {
  KALDI_ASSERT(output_deriv.NumRows() == config_.chunk_size
               && input.NumRows() == config_.chunk_size);
  KALDI_ASSERT(input.NumCols() == input_stats_.NumCols());
  KALDI_ASSERT(output_deriv.NumCols() == output_stats_.NumCols());

  while (1) { // We try to store the stuff we've been given, but we may have to update
    // the model first, to make space for it.
    int32 chunk1, chunk2;
    chunk_manager_.GetTask(&chunk1, &chunk2);
    if (chunk2 == -1) { // Store the stats, and return.
      int32 chunk = chunk1, chunk_size = config_.chunk_size;
      SubMatrix<BaseFloat> input_chunk(input_stats_, chunk * chunk_size,
                                       chunk_size, 0, input_stats_.NumCols());
      input_chunk.CopyFromMat(input);
      SubMatrix<BaseFloat> output_chunk(output_stats_, chunk * chunk_size,
                                        chunk_size, 0, output_chunk.NumCols());
      output_chunk.CopyFromMat(output_deriv);
      chunk_manager_.SetToFull(chunk);
      break;
    } else { // We need to update the model parameters, to free space.
      Update(chunk1, chunk2);
      chunk_manager_.SetToEmpty(chunk1, chunk2);
    }
  }
}

// Update model parameters.
void LinearLayerStats::Update(int32 chunk_start, int32 chunk_end) {
  /*
    Note: for this particular type of layer [which transforms probabilities], we
    store the stats as they relate to the actual probabilities in the matrix, with
    a sum-to-one constraint on each row; but we do the gradient descent in the
    space of unnormalized log probabilities.  This is useful both in terms of
    preconditioning and in order to easily enforce the sum-to-one constraint on
    each column.

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
  
  KALDI_ASSERT(chunk_start >= 0 && chunk_end >= 0 && chunk_end > chunk_start &&
               chunk_end < config_.num_chunks);
  int32 chunk_size = config_.chunk_size;
  SubMatrix<BaseFloat> input_chunks(input_stats_, chunk_size * chunk_start,
                                    chunk_size * (chunk_end - chunk_start),
                                    0, input_stats_.NumCols()),
      output_chunks(output_stats_, chunk_size * chunk_start,
                    chunk_size * (chunk_end - chunk_start),
                    0, output_stats_.NumCols());

  if (layer_to_update_->is_gradient_) {
    // We just want the gradient: do a "vanilla" SGD type of update as
    // we would do on any layer.
    layer_to_update_->params_.AddMatMat(layer_to_update_->learning_rate,
                                        output_chunks, kTrans,
                                        input_chunks, kNoTrans, 1.0); 
  } else { // the "real" update which takes place in unnormalized-log
    // parameter space.
    Matrix<BaseFloat> &params(layer_to_update_->params_);
    int32 num_rows = params.NumRows(), num_cols = params.NumCols();
    Matrix<BaseFloat> gradient(num_rows, num_cols); // objf gradient on this chunk.
    gradient.AddMatMat(1.0, output_chunks, kTrans,
                       input_chunks, kNoTrans, 1.0); 
    
    for (int32 col = 0; col < num_cols; col++) {
      Vector<BaseFloat> param_col(num_rows);
      param_col.CopyColFromMat(params, col);
      Vector<BaseFloat> log_param_col(param_col);
      log_param_col.ApplyLog(); // note: works even for zero, but may have -inf
      for (int32 i = 0; i < num_rows; i++)
        if (log_param_col(i) < -1.0e+20)
          log_param_col(i) = -1.0e+20; // get rid of -inf's,as
      // as we're not sure exactly how BLAS will deal with them.
      Vector<BaseFloat> gradient_col(num_rows);
      gradient_col.CopyColFromMat(gradient, col);
      Vector<BaseFloat> log_gradient(num_rows);
      log_gradient.AddVecVec(1.0, param_col, gradient_col, 0.0); // h <-- diag(c) g.
      BaseFloat cT_g = VecVec(param_col, gradient_col);
      log_gradient.AddVec(-cT_g, param_col); // h += (c^T g) c .
      log_param_col.AddVec(layer_to_update_->learning_rate, log_gradient); // Gradient step,
      // in unnormalized log-prob space.
      log_param_col.ApplySoftMax(); // Go back to probabilities.
      params.CopyColFromVec(log_param_col, col); // Write back to parameters.
    }
  }
}


// We initialize the weights to be uniformly distributed on
// [-1/sqrt(n), +1/sqrt(n)], where n is the input dimension.
// Apparently this is widely used: see  glorot10a.pdf (search term), 
// Glorot and Bengio, "Understanding the difficulty of training deep feedforward networks".
TanhLayer::TanhLayer(int input_size, int output_size):
    params_(output_size, input_size) {
  KALDI_ASSERT(input_size > 0 && output_size > 0);
  BaseFloat end_range = 1.0 / sqrt(input_size),
      begin_range = -end_range;
  for (int32 i = 0; i < output_size; i++) {
    for (int32 j = 0; j < input_size; j++) {
      BaseFloat r = begin_range + RandUniform() * (end_range - begin_range);
      params_(i, j) = r;
    }
  }
}

void TanhLayer::Write(std::ostream &out, bool binary) const {
  WriteToken(out, binary, "<TanhLayer>");
  params_.Write(out, binary);
  WriteToken(out, binary, "</TanhLayer>");  
}

void TanhLayer::Read(std::istream &in, bool binary) {
  ExpectToken(in, binary, "<TanhLayer>");
  params_.Read(in, binary);
  ExpectToken(in, binary, "</TanhLayer>");  
}

void TanhLayer::Forward(const MatrixBase<BaseFloat> &input,
                        MatrixBase<BaseFloat> *output) {
  // This would be simpler if we didn't allow frame splicing.  For
  // the case with no frame splicing, assume input_dim == full_input_dim.
  
  int32 chunk_size = output->NumRows(); // Number of frames in this chunk.  This is fixed
  // during training; the stats take it as part of the initializer.
  int32 input_dim = input.NumCols(), full_input_dim = params_.NumCols(),
      output_dim = output->NumCols();
  int32 num_spliced = full_input_dim / input_dim;
  KALDI_ASSERT(output_dim == params_.NumRows());
  KALDI_ASSERT(full_input_dim == num_spliced * input_dim);
  KALDI_ASSERT(chunk_size + num_spliced - 1 == input.NumRows()); // We'll shift the
  // input row by 1 each time... this equality has to hold for the splicing to work.

  for (int32 s = 0; s < num_spliced; s++) { // without frame splicing, only do this once.
    BaseFloat add_old_output = (s == 0 ? 0.0 : 1.0);
    SubMatrix<BaseFloat> input_part(input, s, chunk_size, 0, input_dim);
    SubMatrix<BaseFloat> params_part(params_, 0, output_dim,
                                     input_dim * s, input_dim);
    output->AddMatMat(1.0, input_part, kNoTrans, params_part, kTrans,
                      add_old_output);
  }
  ApplyTanh(output);
}

// Propagate the derivative back through the nonlinearity.
void TanhLayer::ComputeSumDeriv(const MatrixBase<BaseFloat> &output,
                                const MatrixBase<BaseFloat> &output_deriv,
                                MatrixBase<BaseFloat> *sum_deriv) {
  /*
    Note on the derivative of the tanh function:
    tanh'(x) = sech^2(x) = -(tanh(x)+1) (tanh(x)-1) = 1 - tanh^2(x)
  */
  sum_deriv->CopyFromMat(output);
  sum_deriv->ApplyPow(2.0);
  sum_deriv->Scale(-1.0);
  sum_deriv->Add(1.0);
  // now sum_deriv is 1.0 - [output^2], which is the derivative of the tanh function.
  sum_deriv->MulElements(output_deriv);
  // now each element of sum_deriv is the derivative of the objective function
  // w.r.t. the input to the tanh function.
}


// The backward pass.  Similar note about sizes and frame-splicing
// applies as in "Forward" [this affects "input" and "input_deriv"].
void TanhLayer::Backward(
    const MatrixBase<BaseFloat> &input,
    const MatrixBase<BaseFloat> &output,
    const MatrixBase<BaseFloat> &output_deriv,
    MatrixBase<BaseFloat> *input_deriv, // derivative w.r.t. input, which we add to.
    TanhLayerStats *stats) {

  // sum_deriv will be the derivative of the objective function w.r.t. the sum
  // before the sigmoid is applied.
  Matrix<BaseFloat> sum_deriv(output.NumRows(), output.NumCols(),
                              kUndefined);
  ComputeSumDeriv(output, output_deriv, &sum_deriv);
  
  ComputeInputDeriv(output, sum_deriv, input_deriv);

  stats->AccStats(input, sum_deriv);  
}

// Called from Backward().  Computes "input_deriv".
void TanhLayer::ComputeInputDeriv(const MatrixBase<BaseFloat> &output,
                                  const MatrixBase<BaseFloat> &sum_deriv,
                                  MatrixBase<BaseFloat> *input_deriv) {
  
  // This would be simpler if we didn't allow frame splicing.  For
  // the case with no frame splicing, assume input_dim == full_input_dim.
  
  int32 chunk_size = output.NumRows(); // Number of frames in this chunk.  This is fixed
  // during training; the stats take it as part of the initializer.
  int32 input_dim = input_deriv->NumCols(), full_input_dim = params_.NumCols(),
      output_dim = output.NumCols();
  int32 num_spliced = full_input_dim / input_dim;
  KALDI_ASSERT(output_dim == params_.NumRows());
  KALDI_ASSERT(full_input_dim == num_spliced * input_dim);
  KALDI_ASSERT(chunk_size + num_spliced - 1 == input_deriv->NumRows()); // We'll shift the
  // input row by 1 each time... this equality has to hold for the splicing to work.

  input_deriv->SetZero();
  for (int32 s = 0; s < num_spliced; s++) { // without frame splicing, only do this once.
    SubMatrix<BaseFloat> input_deriv_part(*input_deriv, s, chunk_size, 0, input_dim);
    SubMatrix<BaseFloat> params_part(params_, 0, output_dim,
                                     input_dim * s, input_dim);
    input_deriv_part.AddMatMat(1.0, output, kNoTrans, params_part, kNoTrans,
                               1.0);
  }
}

void TanhLayer::ApplyTanh(MatrixBase<BaseFloat> *output) {
  // Apply tanh function to each element of the output...
  // function is -1 + 2 ( 1 / (1 + e^{-2 x}))
  
  int32 num_rows = output->NumRows(), num_cols = output->NumCols(),
      stride = output->Stride();
  
  BaseFloat *data = output->Data();
  for (int32 row = 0; row < num_rows; row++, data += stride) {
    for (int32 col = 0; col < num_cols; col++) {
      // This if-statement is intended to avoid overflow caused by exponentiating
      // large positive values.
      if (*data >= 0.0) {
        *data = -1.0 + 2.0 / (1 + exp(-2.0 * *data));
      } else {
        *data = 1.0 - 2.0 / (1 + exp(2.0 * *data));
      }
    }    
  }
}


void TanhLayerStats::AccStats(const MatrixBase<BaseFloat> &input,
                              const MatrixBase<BaseFloat> &output_deriv) {
  KALDI_ASSERT(output_deriv.NumRows() == config_.chunk_size);
  KALDI_ASSERT(output_deriv.NumCols() == output_stats_.NumCols());
  // Don't check input size is correct, as affected by splicing:
  // will be checked in SpliceFrames().
  
  while (1) { // We try to store the stuff we've been given, but we may have to update
    // the model first, to make space for it.
    int32 chunk1, chunk2;
    chunk_manager_.GetTask(&chunk1, &chunk2);
    if (chunk2 == -1) { // Store the stats, and return.
      int32 chunk = chunk1, chunk_size = config_.chunk_size;
      SubMatrix<BaseFloat> input_chunk(input_stats_, chunk * chunk_size,
                                       chunk_size, 0, input_stats_.NumCols());
      SpliceFrames(input, &input_chunk);
      SubMatrix<BaseFloat> output_chunk(output_stats_, chunk * chunk_size,
                                        chunk_size, 0, output_stats_.NumCols());
      output_chunk.CopyFromMat(output_deriv);
      chunk_manager_.SetToFull(chunk);
      break;
    } else { // We need to update the model parameters, to free space.
      Update(chunk1, chunk2);
      chunk_manager_.SetToEmpty(chunk1, chunk2);
    }
  }
}

<<<<<<< .mine
SoftmaxLayer::SoftmaxLayer(int input_size, int output_size, BaseFloat learning_rate):
    params_(output_size, input_size), occupancy_(output_size),
    learning_rate_(learning_rate) {
  KALDI_ASSERT(learning_rate_ > 0.0 && learning_rate_ <= 1.0); // Note:
  // learning rate of zero may be used to disable learning for a particular
  // layer, but for this type of layer that doesn't really make sense, in
  // the usage situations we envisage.
}

void SoftmaxLayer::Write(std::ostream &out, bool binary) const {
  WriteToken(out, binary, "<SoftmaxLayer>");
  params_.Write(out, binary);
  occupancy_.Write(out, binary);
  WriteToken(out, binary, "</SoftmaxLayer>");  
}

void SoftmaxLayer::Read(std::istream &in, bool binary) {
  ExpectToken(in, binary, "<SoftmaxLayer>");
  params_.Read(in, binary);
  occupancy_.Read(in, binary);
  ExpectToken(in, binary, "</SoftmaxLayer>");  
}

void SoftmaxLayer::Forward(const MatrixBase<BaseFloat> &input,
                           MatrixBase<BaseFloat> *output) {
  // This would be simpler if we didn't allow frame splicing.  For
  // the case with no frame splicing, assume input_dim == full_input_dim.
  
  int32 chunk_size = output->NumRows(); // Number of frames in this chunk.  This is fixed
  // during training; the stats take it as part of the initializer.
  int32 input_dim = input.NumCols(), full_input_dim = params_.NumCols(),
      output_dim = output->NumCols();
  int32 num_spliced = full_input_dim / input_dim;
  KALDI_ASSERT(output_dim == params_.NumRows());
  KALDI_ASSERT(full_input_dim == num_spliced * input_dim);
  KALDI_ASSERT(chunk_size + num_spliced - 1 == input.NumRows()); // We'll shift the
  // input row by 1 each time... this equality has to hold for the splicing to work.

  for (int32 s = 0; s < num_spliced; s++) { // without frame splicing, only do this once.
    BaseFloat add_old_output = (s == 0 ? 0.0 : 1.0);
    SubMatrix<BaseFloat> input_part(input, s, chunk_size, 0, input_dim);
    SubMatrix<BaseFloat> params_part(params_, 0, output_dim,
                                     input_dim * s, input_dim);
    output->AddMatMat(1.0, input_part, kNoTrans, params_part, kTrans,
                      add_old_output);
  }
  ApplySoftmax(output);
}

void SoftmaxLayer::ApplySoftmax(MatrixBase<BaseFloat> *output) {
  // Apply softmax to each row of the output.
  for (int32 r = 0; r < output->NumRows(); r++) {
    SubVector<BaseFloat> row(*output, r);
    row.ApplySoftMax();
  }
}

// Propagate the derivative back through the nonlinearity.
void SoftmaxLayer::ComputeSumDeriv(const MatrixBase<BaseFloat> &output,
                                   const MatrixBase<BaseFloat> &output_deriv,
                                   MatrixBase<BaseFloat> *sum_deriv) {
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
  const MatrixBase<BaseFloat> &P(output), &E(output_deriv);
  MatrixBase<BaseFloat> &D (*sum_deriv);
  for (int32 r = 0; r < P.NumRows(); r++) {
    SubVector<BaseFloat> p(P, r), e(E, r), d(D, r);
    d.AddVecVec(1.0, p, e, 0.0); // d_i = p_i e_i.
    BaseFloat pT_e = VecVec(p, e); // p^T e.
    d.AddVec(-pT_e, p); // d_i -= (p^T e) p_i
  }
}

// Called from Backward().  Computes "input_deriv".
void SoftmaxLayer::ComputeInputDeriv(const MatrixBase<BaseFloat> &output,
                                     const MatrixBase<BaseFloat> &sum_deriv,
                                     MatrixBase<BaseFloat> *input_deriv) {
  
  // This would be simpler if we didn't allow frame splicing.  For
  // the case with no frame splicing, assume input_dim == full_input_dim.
  
  int32 chunk_size = output.NumRows(); // Number of frames in this chunk.  This is fixed
  // during training; the stats take it as part of the initializer.
  int32 input_dim = input_deriv->NumCols(), full_input_dim = params_.NumCols(),
      output_dim = output.NumCols();
  int32 num_spliced = full_input_dim / input_dim;
  KALDI_ASSERT(output_dim == params_.NumRows());
  KALDI_ASSERT(full_input_dim == num_spliced * input_dim);
  KALDI_ASSERT(chunk_size + num_spliced - 1 == input_deriv->NumRows()); // We'll shift the
  // input row by 1 each time... this equality has to hold for the splicing to work.

  input_deriv->SetZero();
  for (int32 s = 0; s < num_spliced; s++) { // without frame splicing, only do this once.
    SubMatrix<BaseFloat> input_deriv_part(*input_deriv, s, chunk_size, 0, input_dim);
    SubMatrix<BaseFloat> params_part(params_, 0, output_dim,
                                     input_dim * s, input_dim);
    input_deriv_part.AddMatMat(1.0, output, kNoTrans, params_part, kNoTrans,
                               1.0);
  }
}

// The backward pass.  Similar note about sizes and frame-splicing
// applies as in "Forward" [this affects "input" and "input_deriv"].
void SoftmaxLayer::Backward(
    const MatrixBase<BaseFloat> &input,
    const MatrixBase<BaseFloat> &output,
    const MatrixBase<BaseFloat> &output_deriv,
    MatrixBase<BaseFloat> *input_deriv, // derivative w.r.t. input, which we add to.
    SoftmaxLayerStats *stats) {

  // sum_deriv will be the derivative of the objective function w.r.t. the sum
  // before the sigmoid is applied.
  Matrix<BaseFloat> sum_deriv(output.NumRows(), output.NumCols(),
                              kUndefined);
  ComputeSumDeriv(output, output_deriv, &sum_deriv);
  
  ComputeInputDeriv(output, sum_deriv, input_deriv);

  stats->AccStats(input, output, sum_deriv);  
}


void SoftmaxLayerStats::Update(int32 chunk_start, int32 chunk_end) {
  // Update model parameters.  Vanilla SGD.
  KALDI_ASSERT(chunk_start >= 0 && chunk_end >= 0 && chunk_end > chunk_start &&
               chunk_end < config_.num_chunks);
  int32 chunk_size = config_.chunk_size;
  SubMatrix<BaseFloat> input_chunks(input_stats_, chunk_size * chunk_start,
                                    chunk_size * (chunk_end - chunk_start),
                                    0, input_stats_.NumCols()),
      output_chunks(output_stats_, chunk_size * chunk_start,
                    chunk_size * (chunk_end - chunk_start),
                    0, output_stats_.NumCols()),
      output_sums_part(output_sums_, chunk_start, (chunk_end - chunk_start),
                       0, output_sums_.NumCols());
  
  layer_to_update_->params_.AddMatMat(layer_to_update_->learning_rate,
                                      output_chunks, kTrans,
                                      input_chunks, kNoTrans, 1.0);
  layer_to_update_->occupancy_.AddRowSumMat(output_sums_part);  
}


void SoftmaxLayerStats::AccStats(const MatrixBase<BaseFloat> &input,
                                 const MatrixBase<BaseFloat> &output,
                                 const MatrixBase<BaseFloat> &sum_deriv) {
  KALDI_ASSERT(sum_deriv.NumRows() == config_.chunk_size);
  KALDI_ASSERT(sum_deriv.NumCols() == output_stats_.NumCols());
  // Don't check input size is correct, as it's affected by splicing:
  // will be checked in SpliceFrames().
  
  while (1) { // We try to store the stuff we've been given, but we may have to update
    // the model first, to make space for it.
    int32 chunk1, chunk2;
    chunk_manager_.GetTask(&chunk1, &chunk2);
    if (chunk2 == -1) { // Store the stats, and return.
      int32 chunk = chunk1, chunk_size = config_.chunk_size;
      SubMatrix<BaseFloat> input_chunk(input_stats_, chunk * chunk_size,
                                       chunk_size, 0, input_stats_.NumCols());
      SpliceFrames(input, &input_chunk);
      SubMatrix<BaseFloat> output_chunk(output_stats_, chunk * chunk_size,
                                        chunk_size, 0, output_stats_.NumCols());
      output_chunk.CopyFromMat(sum_deriv);

      // Store the sum of the output-- this is an occupancy-like quantity.
      SubVector<BaseFloat> output_sum(output_sums_, chunk);
      output_sum.AddRowSumMat(output);
      chunk_manager_.SetToFull(chunk);
      break;
    } else { // We need to update the model parameters, to free space.
      Update(chunk1, chunk2);
      chunk_manager_.SetToEmpty(chunk1, chunk2);
    }
  }
}

bool ChunkManager::TaskIsAvailable() {
  bool saw_full = false, saw_emptying = false;
  for (size_t i = 0; i < chunk_status_.size(); i++) {
    if (chunk_status_[i] == kEmpty) return true;
    if (chunk_status_[i] == kFull) saw_full = true;
    if (chunk_status_[i] == saw_emptying) saw_emptying = true;
  }
  return (saw_full && !saw_emptying);
}

void ChunkManager::SetToFull(int32 chunk) {
  mutex_.Lock(); // we lock this mutex before changing anything...
  KALDI_ASSERT(chunk >= 0 && chunk < NumChunks());
  KALDI_ASSERT(chunk_status_[chunk] == kFilling);
  bool task_was_available = TaskIsAvailable();
  chunk_status_[chunk] = kFull;
  if (!task_was_available) { // Task was not available and is now,
    // so need to unlock get_task_mutex_.
    get_task_mutex_.Unlock();
  }
  mutex_.Unlock();
}

// Sets chunks in range begin ... end-1
// [which should be in status kEmptying] to status kEmpty.
void ChunkManager::SetToEmpty(int32 begin, int32 end) {
  mutex_.Lock(); // we lock this mutex before changing anything..
  KALDI_ASSERT(begin >= 0 && end > begin && end <= NumChunks());
  bool task_was_available = TaskIsAvailable();
  for (int32 i = begin; i < end; i++) {
    KALDI_ASSERT(chunk_status_[i] == kEmptying);
    chunk_status_[i] = kEmpty;
  }
  if (!task_was_available) { // Task was not available and is now,
    // so need to unlock get_task_mutex_.
    get_task_mutex_.Unlock();
  }
  mutex_.Unlock();
}

// If we end up spending a lot of time in this routine, we'll have to
// modify this class so as to store the counts directly in the class.
// We did it like this for simplicity and ease of programming.
void ChunkManager::GetCounts(int32 *num_full_ptr,
                             int32 *num_empty_ptr,
                             int32 *num_emptying_ptr) {
  int32 num_full = 0, num_empty = 0, num_emptying = 0;
  for (size_t i = 0; i < NumChunks(); i++) {
    if (chunk_status_[i] == kFull) num_full++;
    if (chunk_status_[i] == kEmpty) num_empty++;
    if (chunk_status_[i] == kEmptying) num_emptying++;
  }
  *num_full_ptr = num_full;
  *num_empty_ptr = num_empty;
  *num_emptying_ptr = num_emptying;
}

// This function sets a and b to the [begin, end] of the
// largest range of chunks that are all in state kFull.
// Dies if resulting range is empty.
void ChunkManager::GetLargestFullRange(int32 *a, int32 *b) {
  int32 cur_start = -1;
  int32 largest_range_size = 0;
  for (int32 i = 0; i < NumChunks(); i++) {
    if (chunk_status_[i] != kFull) {
      cur_start = -1;
    } else {
      if (cur_start == -1) { cur_start = i; }
      int32 this_range_size = i - cur_start + 1;
      if (this_range_size > largest_range_size) {
        *a = cur_start;
        *b = i + 1;
        largest_range_size = i - cur_start + 1;
      }
    }
  }
  KALDI_ASSERT(largest_range_size != 0);
}

void ChunkManager::GetTask(int32 *chunk1, int32 *chunk2) {
  get_task_mutex_.Lock();
  mutex_.Lock();
  // If we are more than two thirds full and nothing is in state kEmptying
  // (i.e. no other process is updating the model), then the task
  // is to update the model [so set chunk1 and chunk2, meaning a range
  // of chunks to write to the model]; else give a chunk to fill.
  int32 num_full, num_empty, num_emptying;
  GetCounts(&num_full, &num_empty, &num_emptying);
  // A task must be availbale, or we shouldn't have been able
  // to lock get_task_mutex_.
  bool task_is_avail = (num_empty > 0 || (num_full != 0 && num_emptying == 0));
  KALDI_ASSERT(task_is_avail); // Or we shouldn't have been able to
  // acquire the lock "get_task_mutex_".
  
  if (num_empty == 0 ||
      (num_full + num_full/2 > NumChunks() && num_emptying == 0)) {
    // task is to empty a range of chunks.
    GetLargestFullRange(chunk1, chunk2);
    for (int32 i = *chunk1; i < *chunk2; i++) {
      KALDI_ASSERT(chunk_status_[i] == kFull);
      chunk_status_[i] = kEmptying;
    }
    task_is_avail = (num_empty != 0); // The task of emptying a range
    // is not available now, so only possible task is to fill a chunk.
  } else {
    *chunk2 = -1; // indicates we're returning the task to full a chunk.
    *chunk1 = GetNextEmptyChunk();
    num_empty--;
    task_is_avail = (num_empty > 0 || (num_full != 0 && num_emptying == 0));
  }
  if (task_is_avail)
    get_task_mutex_.Unlock();
  // else leave it locked-- no tasks is available so no other process
  // should be allowed to enter this function until a task becomes
  // available.
  mutex_.Unlock();
}

int32 ChunkManager::GetNextEmptyChunk() {
  /* This might be a little hard to understand.  It's a mechanism to
     cycle backward and forward, e.g. suppose NumChunks() = 4, we'd
     cycle through positions 0, 1, 2, 3, 2, 1, 0, 1, ...
     This is necessary when finding an empty chunk to fill, due to
     the interaction with GetLargestFullRange(): we need to ensure
     that not too much time elapses between
     filling a chunk and having it emptied.  If we always filled
     the leftmost available chunk, there would be positions on the
     right that would be only rarely committed to the model.
     
     next_chunk_to_fill_ encodes both the position and direction, so
     it cycles through 0, 1, 2, 3, -2, -1, 0, 1, 2, 3, etc.
  */
  int32 num_chunks = NumChunks(), max_loop = num_chunks * 2;
  for (int32 loop = 0; loop < max_loop; loop++) { // This for
    // statement could be while(1); it's a for loop in order
    // to detect an infinite loop which would be a bug.
    int32 cur_sign = (next_chunk_to_fill_ >= 0 ? 1 : -1);
    int32 cur_pos = abs(next_chunk_to_fill_);

    int32 next_pos = cur_sign * (cur_sign + cur_pos);
    if (next_pos == num_chunks)
      next_pos = -(num_chunks-2); // bounce off the end; change
    // direction.
    next_chunk_to_fill_ = next_pos;
    if (chunk_status_[cur_pos] == kEmpty)
      return cur_pos;
  }
  KALDI_ERR << "Parallel programming error: empty chunk requested but none exists.";
  return 0; // silence warning.
}




} // namespace kaldi
