// feat/feature-functions-test.cc

// Copyright 2013   Arnab Ghoshal

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

}

int main() {
  kaldi::g_kaldi_verbose_level = 1;
  kaldi::UnitTestRealCepstrum<float>();
  kaldi::UnitTestRealCepstrum<double>();
  std::cout << "Test OK.\n";
  return 0;
}
