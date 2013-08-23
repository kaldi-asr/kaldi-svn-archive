// feat/feature-lpc.h

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

#ifndef KALDI_FEAT_FEATURE_LPC_H_
#define KALDI_FEAT_FEATURE_LPC_H_

#include "feat/feature-functions.h"

namespace kaldi {

/// LpcOptions contains basic options for computing LPC features
struct LpcOptions {
  FrameExtractionOptions frame_opts;
  int32 lpc_order;  // e.g. 13: number of LP coefficients
  BaseFloat energy_floor;

  LpcOptions() : lpc_order(13),
                 energy_floor(FLT_EPSILON) {
    frame_opts.round_to_power_of_two = false;  // We are not doing FFT
  }

  void Register(OptionsItf *po) {
    frame_opts.Register(po);
    po->Register("lpc-order", &lpc_order,
                 "Number of LP coefficients.");
    po->Register("energy-floor", &energy_floor,
                 "Floor on energy (absolute, not relative) in LPC computation");
  }
};


/// Class for computing LPC features.
class Lpc {
 public:
  explicit Lpc(const LpcOptions &opts)
    : opts_(opts), feature_window_function_(opts.frame_opts) {}
  ~Lpc() {}

  int32 Dim() { return opts_.lpc_order; }

  void Compute(const VectorBase<BaseFloat> &wave,
               Matrix<BaseFloat> *lp_coeffs,
               Matrix<BaseFloat> *lp_residuals = NULL,
               Vector<BaseFloat> *wave_remainder = NULL);

 private:
  LpcOptions opts_;
  FeatureWindowFunction feature_window_function_;
  KALDI_DISALLOW_COPY_AND_ASSIGN(Lpc);
};

BaseFloat WaveToLpc(const VectorBase<BaseFloat> &wave_window,
                    int32 lpc_order,
                    BaseFloat energy_floor,
                    Vector<BaseFloat> *lp_coeffs,
                    Vector<BaseFloat> *lp_residuals = NULL);

}  // namespace kaldi

#endif  // KALDI_FEAT_FEATURE_LPC_H_
