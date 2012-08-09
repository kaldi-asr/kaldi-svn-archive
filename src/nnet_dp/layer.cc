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
    SubMatrix<BaseFloat> dst(spliced_out, 0, num_output_rows,
                             small_dim * s, small_dim * (s+1));
    dst.CopyFromMat(src);
  }
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
      SubMatrix input_chunk(input_stats_, chunk * chunk_size,
                            chunk_size, 0, input_stats_.NumCols());
      input_chunk.CopyFromMat(input);
      SubMatrix output_chunk(output_stats_, chunk * chunk_size,
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
  KALDI_ASSERT(chunk_start >= 0 && chunk_end >= 0 && chunk_end > chunk_start &&
               chunk_end < config_.num_chunks);
  int32 chunk_size = config_.chunk_size;
  SubMatrix<BaseFloat> input_chunks(input_stats_, chunk_size * chunk_start,
                                    chunk_size * (chunk_end - chunk_start),
                                    0, input_stats_.NumCols()),
      output_chunks(output_stats_, chunk_size * chunk_start,
                    chunk_size * (chunk_end - chunk_start),
                    0, output_stats_.NumCols());

  // layer_to_update->params_ += learning_rate * output_chunks^T * input_chunks
  layer_to_update->params_.AddMatMat(config_.learning_rate, output_chunks, kTrans,
                                     input_chunks, kNoTrans, 1.0);
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
      output_dim = output->NumCols();
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
    input_deriv_part->AddMatMat(1.0, output, kNoTrans, params_part, kNoTrans,
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
      SubMatrix input_chunk(input_stats_, chunk * chunk_size,
                            chunk_size, 0, input_stats_.NumCols());
      SpliceFrames(input, &input_chunk);
      SubMatrix output_chunk(output_stats_, chunk * chunk_size,
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
  for (int32 row = 0; row < output->NumRows(); row++) {
    SubVector<BaseFloat> row(*output, row);
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
    The [matrix-valued] 2nd derivative of this function is
       diag(p) - p p^T
    Let the derivative vector at the output be e, and at the input be
    d.  We have
       d = diag(p) e - p (p^T e).
    d_i = p_i e_i - p_i (p^T e).    
  */
  const MatrixBase<BaseFloat> &p(output), &e(output_deriv);
  MatrixBase<BaseFloat> &d (*sum_deriv);
  d.AddVecVec(1.0, p, e, 0.0); // d_i = p_i e_i.
  BaseFloat pT_e = VecVec(p, e); // p^T e.
  d.AddVec(-pT_e, p); // d_i -= (p^T e) p_i
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
      output_dim = output->NumCols();
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
    input_deriv_part->AddMatMat(1.0, output, kNoTrans, params_part, kNoTrans,
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
    TanhLayerStats *stats) {

  // sum_deriv will be the derivative of the objective function w.r.t. the sum
  // before the sigmoid is applied.
  Matrix<BaseFloat> sum_deriv(output.NumRows(), output.NumCols(),
                              kUndefined);
  ComputeSumDeriv(output, output_deriv, &sum_deriv);
  
  ComputeInputDeriv(output, sum_deriv, input_deriv);

  stats->AccStats(input, output, sum_deriv);  
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
      SubMatrix input_chunk(input_stats_, chunk * chunk_size,
                            chunk_size, 0, input_stats_.NumCols());
      SpliceFrames(input, &input_chunk);
      SubMatrix output_chunk(output_stats_, chunk * chunk_size,
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



}  // namespace kaldi
