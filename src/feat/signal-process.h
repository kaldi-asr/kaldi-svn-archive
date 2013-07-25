// feat/signal-process.h

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

#ifndef KALDI_FEAT_SIGNAL_PROCESS_H_
#define KALDI_FEAT_SIGNAL_PROCESS_H_

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/matrix-lib.h"

namespace kaldi {

/// Compute the autocorrelation coefficients for a windowed signal. The order
/// of autocorrelation (i.e. maximum lag) must be smaller than the number of
/// samples in the signal window. If the signal in the window has negligible
/// energy, then the autocorrelation is set to delta(0).
void AutoCorrelate(const VectorBase<BaseFloat> &signal_window,
                   int32 autoc_order,
                   VectorBase<BaseFloat> *autoc,
                   BaseFloat energy_floor = FLT_EPSILON);

/// Compute the autocorrelation coefficients for a windowed signal, normalized
/// by the total energy. The order of autocorrelation (i.e. maximum lag) must
/// be smaller than the number of samples in the signal window. If the signal
/// in the window has negligible energy, then output is set to delta(0).
BaseFloat NormalizedAutoCorrelate(const VectorBase<BaseFloat> &signal_window,
                                  int32 autoc_order,
                                  VectorBase<BaseFloat> *autoc,
                                  BaseFloat energy_floor = FLT_EPSILON);

/// Levinson-Durbin recursion for obtaining the linear predictor coefficients
/// from the autocorrelation values. Returns the LPC gain, i.e. the prediction
/// error energy.
BaseFloat Levinson(const VectorBase<BaseFloat> &autoc, int32 lpc_order,
                   VectorBase<BaseFloat> *lp_coeff,
                   BaseFloat energy_floor = FLT_EPSILON);

/// FIR or moving-average filter. The signal_window, filter_coeffs, and output
/// must point to different memory locations.
void FirFilter(const VectorBase<BaseFloat> &signal_window,
               const VectorBase<BaseFloat> &filter_coeffs,
               VectorBase<BaseFloat> *output);

/// Direct form II implemenation of an IIR filter.
void IirFilter(const VectorBase<BaseFloat> &signal_window,
               const VectorBase<BaseFloat> &feedforward_coeffs,
               const VectorBase<BaseFloat> &feedback_coeffs,
               VectorBase<BaseFloat> *output);

}  // namespace kaldi

#endif  // KALDI_FEAT_SIGNAL_PROCESS_H_
