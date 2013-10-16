// feat/feature-aperiodic-test.cc

// Copyright 2013  Arnab Ghoshal

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


#include <iostream>

#include "feat/feature-aperiodic.h"
#include "feat/wave-reader.h"

namespace kaldi {

static void UnitTestAperiodic() {
  WaveData wave;
  {
    std::ifstream is("test_data/test.wav");
    wave.Read(is);
  }
  KALDI_ASSERT(wave.Data().NumRows() == 1);
  SubVector<BaseFloat> waveform(wave.Data(), 0);

  Matrix<BaseFloat> f0_info;
  {
    bool binary;
    Input in("test_data/test.f0", &binary);
    f0_info.Read(in.Stream(), binary);
  }

  AperiodicEnergyOptions opts;
  opts.frame_opts.frame_length_ms = 30;  // That's what was used for get_f0
  AperiodicEnergy ap_energy(opts);
  int32 num_frames = NumFrames(waveform.Dim(), opts.frame_opts);
  int32 num_frames_f0 = f0_info.NumRows();
  KALDI_LOG << "NF = " << num_frames << "; NR = " << f0_info.NumRows();
//  KALDI_ASSERT(f0_info.NumRows() == num_frames);
  Vector<BaseFloat> f0(num_frames_f0);
  f0.CopyColFromMat(f0_info, 0);
  Vector<BaseFloat> pov(num_frames_f0);
  pov.CopyColFromMat(f0_info, 1);

  Matrix<BaseFloat> m;
  ap_energy.Compute(waveform, pov, f0, &m, NULL);
}

}


int main() {
  try {
    for (int i = 0; i < 5; i++)
      kaldi::UnitTestAperiodic();
    std::cout << "Tests succeeded.\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return 1;
  }
}


