// feat/signal-process.cc

// Copyright 2013-  Arnab Ghoshal

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

#include <algorithm>

#include "feat/signal-process.h"

namespace kaldi {

void AutoCorrelate(const VectorBase<BaseFloat> &signal_window,
                   int32 autoc_order,
                   VectorBase<BaseFloat> *autoc,
                   BaseFloat energy_floor) {
  KALDI_ASSERT(autoc != NULL);
  KALDI_ASSERT(autoc->Dim() == autoc_order);
  int32 num_samples = signal_window.Dim();
  if (num_samples <= autoc_order) {
    KALDI_ERR << "Cannot produce autocorelation coefficients of order: "
              << autoc_order << ", since signal window has " << num_samples
              << " samples.";
  }

  autoc->SetZero();
  BaseFloat total_energy = signal_window.SumPower(2);

  // Handle the degenerate case of having no signal in the window. The output
  // is set to delta(0). This is motivated by the fact that when used in the
  // Levinson-Durbin recursion, the delta(0) autocorrelation will produce LP
  // coefficients that are 0.
  if (total_energy < energy_floor) {
    std::ostringstream oss;
    signal_window.Write(oss, false /*not binary write*/);
    KALDI_WARN << "Zero energy in signal window: " << oss.str()
               << "\n Setting autocorrelation to delta(0).";
    (*autoc)(0) = 1;  // The rest are initialized to 0
    return;
  }

  (*autoc)(0) = total_energy;
  for (int32 i = 1; i < autoc_order; ++i) {
    double acc = 0.0;
    for (int32 j = 0; j < num_samples - i; ++j)
      acc += signal_window(j) * signal_window(j+i);
    (*autoc)(i) = acc;
  }
}


BaseFloat NormalizedAutoCorrelate(const VectorBase<BaseFloat> &signal_window,
                                  int32 autoc_order,
                                  VectorBase<BaseFloat> *autoc,
                                  BaseFloat energy_floor) {
  KALDI_ASSERT(autoc != NULL);
  KALDI_ASSERT(autoc->Dim() == autoc_order);
  int32 num_samples = signal_window.Dim();
  if (num_samples <= autoc_order) {
    KALDI_ERR << "Cannot produce autocorelation coefficients of order: "
              << autoc_order << ", since signal window has " << num_samples
              << " samples.";
  }

  autoc->SetZero();
  (*autoc)(0) = 1;  // Normalized energy = 1
  BaseFloat total_energy = signal_window.SumPower(2);

  // Handle the degenerate case of having no signal in the window. The output
  // is set to delta(0). This is motivated by the fact that when used in the
  // Levinson-Durbin recursion, the delta(0) autocorrelation will produce LP
  // coefficients that are 0.
  if (total_energy < energy_floor) {
    std::ostringstream oss;
    signal_window.Write(oss, false /*not binary write*/);
    KALDI_WARN << "Zero energy in signal window: " << oss.str()
               << "\n Setting autocorrelation to delta(0).";
    // Autocorrelation is already initialized to delta(0), so nothing to do.
    return energy_floor;
  }

  double norm = (1/total_energy);  // Normalize the coefficients by total energy
  for (int32 i = 1; i < autoc_order; ++i) {
    double acc = 0.0;
    for (int32 j = 0; j < num_samples - i; ++j)
      acc += signal_window(j) * signal_window(j+i);
    (*autoc)(i) = acc * norm;
  }
  return total_energy;
}


BaseFloat Levinson(const VectorBase<BaseFloat> &autoc, int32 lpc_order,
                   VectorBase<BaseFloat> *lp_coeffs,
                   BaseFloat energy_floor) {
  KALDI_ASSERT(lp_coeffs != NULL);
  KALDI_ASSERT(lp_coeffs->Dim() == lpc_order);
  if (autoc.Dim() <= lpc_order) {
    KALDI_ERR << "Cannot compute LPCs of order: " << lpc_order
              << ", since autocorrelation order is " << autoc.Dim();
  }

  lp_coeffs->SetZero();
  BaseFloat E = autoc(0);  // prediction error energy
  KALDI_ASSERT(E >= 0.0 && "Negative energy in autocorrelation.");

  if (E < energy_floor) {
    KALDI_WARN << "Zero energy is autocorrelation. Setting LPCs to 0";
    return energy_floor;
  }

  Vector<BaseFloat> new_lp_coeffs(lpc_order, kSetZero);
  for (int32 i = 0; i < lpc_order; i++) {
    BaseFloat k = autoc(i+1);  // reflection coefficient
    for (int32 j = 0; j < i; j++)
      k -= (*lp_coeffs)(j) * autoc(i-j);
    k /= E;
    E *= (1 - k*k);
    new_lp_coeffs(i) = k;
    for (int32 j = 0; j < i; j++)
      new_lp_coeffs(j) = (*lp_coeffs)(j) - k * (*lp_coeffs)(i-j);
    for (int32 j = 0; j < i; j++)
      (*lp_coeffs)(j) = new_lp_coeffs(j);
  }
  return E;
}


void FirFilter(const VectorBase<BaseFloat> &signal_window,
               const VectorBase<BaseFloat> &filter,
               VectorBase<BaseFloat> *output) {
  KALDI_ASSERT(output != NULL);
  KALDI_ASSERT(signal_window.Data() != filter.Data());
  KALDI_ASSERT(signal_window.Data() != output->Data());
  KALDI_ASSERT(signal_window.Dim() == output->Dim());

  const BaseFloat* __restrict__ signal_data = signal_window.Data();
  const BaseFloat* __restrict__ filter_data = filter.Data();
  BaseFloat* __restrict__ output_data = output->Data();
  int32 num_samples = signal_window.Dim(),
      filter_length = filter.Dim();
  output->SetZero();
  for (int32 n = 0; n < num_samples; n++) {
    for (int32 k = 0; k < std::min(filter_length, n); k++)
      output_data[n] += filter_data[k] * signal_data[n - k];
  }
}


void IirFilter(const VectorBase<BaseFloat> &signal_window,
               const VectorBase<BaseFloat> &feedforward_coeffs,
               const VectorBase<BaseFloat> &feedback_coeffs,
               VectorBase<BaseFloat> *output) {
  KALDI_ERR << "Not implemented yet";
}


}  // namespace kaldi
