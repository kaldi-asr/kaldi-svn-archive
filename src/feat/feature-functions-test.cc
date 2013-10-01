// feat/feature-functions-test.cc

// Copyright 2013   Johns Hopkins University (author: Daniel Povey);
//                  Arnab Ghoshal;

// See ../../COPYING for clarification regarding multiple authors
//
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

#include "feat/feature-functions.h"

using kaldi::int32;
using kaldi::BaseFloat;

// TODO: some of the other functions should be tested.
namespace kaldi {

template<class Real> void GenerateSinusoid(const VectorBase<Real> &sample_times,
                                           Real frequency,
                                           Vector<Real> *output,
                                           bool add) {
  int32 dim = sample_times.Dim();
  KALDI_ASSERT(output->Dim() == dim);
  for (int32 i = 0; i < dim; ++i) {
    Real sample = sin(M_2PI * frequency * sample_times(i));
    if (add)
      (*output)(i) += sample;
    else
      (*output)(i) = sample;
  }
}

template<class Real> static void UnitTestRealCepstrum() {
  Real frequency = 100.0 + 100.0 * RandUniform();  // random freq. in 100-200Hz
  Real sampling_freq = 10 * frequency;
  Real sample_period = 1/sampling_freq;
  int32 num_samples = static_cast<int32>(RandInt(1, 5)*frequency);  // 1-5s
  Vector<Real> sample_times(num_samples);
  for (int32 i = 0; i < num_samples; ++i)
    sample_times(i) = i * sample_period;
  Vector<Real> x(num_samples);
  GenerateSinusoid(sample_times, frequency, &x, false /*don't add*/);
  x.Scale(2.0);
  for (int32 i = 2; i <= 4; ++i)  // add some harmonics
    GenerateSinusoid(sample_times, (i*frequency), &x, true /*add to x*/);
  Vector<Real> y(RoundUpToNearestPowerOfTwo(num_samples), kSetZero);
  y.Range(0, num_samples).CopyFromVec(x);
  RealFft(&y, true);
  ComputePowerSpectrum(&y);
  Vector<Real> power_spectrum(y.Range(0, y.Dim()/2+1));
  PowerSpecToRealCeps(&y);
  Vector<Real> z(y);
  RealCepsToMagnitudeSpec(&y, false /*log magnitude*/);
  SubVector<Real> recons_spec(y, 0, y.Dim()/2+1);
  recons_spec.Scale(2.0);
  recons_spec.ApplyExp();
  KALDI_ASSERT(recons_spec.ApproxEqual(power_spectrum, 1e-5));

  RealCepsToMagnitudeSpec(&z, true /*magnitude*/);
  SubVector<Real> recons_spec2(z, 0, z.Dim()/2+1);
  recons_spec2.ApplyPow(2.0);
  KALDI_ASSERT(recons_spec2.ApproxEqual(power_spectrum, 1e-5));
}

void UnitTestOnlineCmvn() {
  for (int32 i = 0; i < 100; i++) {
    int32 num_frames = 1 + (rand() % 10 * 10);
    int32 dim = 1 + rand() % 10;
    SlidingWindowCmnOptions opts;
    opts.center = (rand() % 2 == 0);
    opts.normalize_variance = (rand() % 2 == 0);
    opts.cmn_window = 5 + rand() % 50;
    opts.min_window = 1 + rand() % 100;
    if (opts.min_window > opts.cmn_window)
      opts.min_window = opts.cmn_window;

    Matrix<BaseFloat> feats(num_frames, dim),
        output_feats(num_frames, dim),
        output_feats2(num_frames, dim);
    feats.SetRandn();
    SlidingWindowCmn(opts, feats, &output_feats);

    for (int32 t = 0; t < num_frames; t++) {
      int32 window_begin, window_end;
      if (opts.center) {
        window_begin = t - (opts.cmn_window / 2),
            window_end = window_begin + opts.cmn_window;
        int32 shift = 0;
        if (window_begin < 0)
          shift = -window_begin;
        else if (window_end > num_frames)
          shift = num_frames - window_end;
        window_end += shift;
        window_begin += shift;
      } else {
        window_begin = t - opts.cmn_window;
        window_end = t;
        if (window_end < opts.min_window)
            window_end = opts.min_window;
      }
      if (window_begin < 0) window_begin = 0;
      if (window_end > num_frames) window_end = num_frames;
      int32 window_size = window_end - window_begin;
      for (int32 d = 0; d < dim; d++) {
        double sum = 0.0, sumsq = 0.0;
        for (int32 t2 = window_begin; t2 < window_end; t2++) {
          sum += feats(t2, d);
          sumsq += feats(t2, d) * feats(t2, d);
        }
        double mean = sum / window_size, uncentered_covar = sumsq / window_size,
            covar = uncentered_covar - mean * mean;
        covar = std::max(covar, 1.0e-20);
        double data = feats(t, d),
            norm_data = data - mean;
        if (opts.normalize_variance)
          norm_data /= sqrt(covar);
        output_feats2(t, d) = norm_data;
      }
    }
    KALDI_ASSERT(output_feats.ApproxEqual(output_feats2, 0.0001));
  }
}

}

int main() {
  kaldi::g_kaldi_verbose_level = 3;
  kaldi::UnitTestOnlineCmvn();
  kaldi::UnitTestRealCepstrum<float>();
  kaldi::UnitTestRealCepstrum<double>();
  std::cout << "Test OK.\n";
  return 0;
}
