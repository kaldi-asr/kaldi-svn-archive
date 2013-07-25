// feat/feature-lpc.cc

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


#include "feat/feature-lpc.h"
#include "feat/signal-process.h"


namespace kaldi {

void Lpc::Compute(const VectorBase<BaseFloat> &wave,
                  Matrix<BaseFloat> *lp_coeffs,
                  Matrix<BaseFloat> *lp_residuals,
                  Vector<BaseFloat> *wave_remainder) {
  KALDI_ASSERT(lp_coeffs != NULL);
  int32 rows_out = NumFrames(wave.Dim(), opts_.frame_opts),
      cols_out = opts_.lpc_order;
  if (rows_out == 0) {
    KALDI_ERR << "Waveform shorter than window length (#samples = "
              << wave.Dim() << "; window length = "
              << opts_.frame_opts.WindowSize() << ")";
  }

  lp_coeffs->Resize(rows_out, cols_out);

  if (wave_remainder != NULL)
    ExtractWaveformRemainder(wave, opts_.frame_opts, wave_remainder);
  if (lp_residuals != NULL)
    lp_residuals->Resize(rows_out, opts_.frame_opts.WindowSize(), kSetZero);

  Vector<BaseFloat> signal_window;  // windowed waveform.
  for (int32 r = 0; r < rows_out; r++) {  // r is the frame index
    SubVector<BaseFloat> this_lpc(lp_coeffs->Row(r));
    ExtractWindow(wave, r, opts_.frame_opts, feature_window_function_,
                  &signal_window);
    Vector<BaseFloat> autoc(opts_.lpc_order);
    BaseFloat total_energy = NormalizedAutoCorrelate(signal_window,
                                                     opts_.lpc_order,
                                                     &autoc);
    BaseFloat gain = Levinson(autoc, opts_.lpc_order, &this_lpc);
    //todo: add energy floor to levinson
    gain *= total_energy;  // Since the autocorrelations were normalized

    if (lp_residuals != NULL) {
      SubVector<BaseFloat> this_residual(lp_residuals->Row(r));
      if (total_energy < opts_.energy_floor) {
        this_residual.CopyFromVec(signal_window);
        return;
      }
      gain = 1/sqrt(gain);
      Vector<BaseFloat> filter(opts_.lpc_order+1, kUndefined);
      filter.Range(1, opts_.lpc_order).CopyFromVec(this_lpc);
      filter.Range(1, opts_.lpc_order).Scale(-1.0);
      filter(0) = 1;
      FirFilter(signal_window, filter, &this_residual);
      this_residual.Scale(gain);
    }
  }
}

}  // namespace kaldi
