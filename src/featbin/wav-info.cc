// idlakfexbin/idlakfex.cc

// Copyright 2013 CereProc Ltd.  (Author: Matthew Aylett)

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
//

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "feat/wave-reader.h"
// Simple wrapper around class WaveData to extract sample rate and
// and duration of a riff wav file

int main(int argc, char *argv[]) {
  const char *usage =
      "Get information on a RIFF wav file\n"
      "Usage:  wav-info wavfile\n";
  std::string wavin;
  try {
    kaldi::ParseOptions po(usage);
    po.Read(argc, argv);
    // Must have input wavfile
    if (po.NumArgs() != 1) {
      po.PrintUsage();
      exit(1);
    }
    wavin = po.GetArg(1);
    bool binary;
    kaldi::Input ki(wavin, &binary);
    kaldi::Output kio("-", false);
    kaldi::WaveData wav;
    wav.Read(ki.Stream());
    kaldi::WriteToken(kio.Stream(), false, "SampleRate:");
    kaldi::WriteBasicType(kio.Stream(), false, wav.SampFreq());
    std::endl(kio.Stream());
    kaldi::WriteToken(kio.Stream(), false, "Duration:");
    kaldi::WriteBasicType(kio.Stream(), false, wav.Duration());
    std::endl(kio.Stream());
    kaldi::WriteToken(kio.Stream(), false, "Samples:");
    kaldi::WriteBasicType(kio.Stream(), false, wav.Data().NumCols());
    std::endl(kio.Stream());

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
