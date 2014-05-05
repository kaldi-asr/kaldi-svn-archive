// featbin/wav-add-noise.cc

// Copyright 2014  Vimal Manohar

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

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "feat/feature-mfcc.h"
#include "feat/wave-reader.h"

#include <sstream>

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    const char *usage =
        "Add random Gaussian noise of a particular SNR to a wave file "
        "and write the noisy wave file. \n"
        "Optionally can also write the noise file\n"
        "\n"
        "Usage:  wav-add-noise [options...] <wav-rspecifier> <wav-wspecifier> [<noise-wspcifier>]\n"
        "e.g. wav-add-noise --snr=10 scp:wav.scp scp:noisy_wav.scp scp:noise.scp\n"
        "See also: wav-copy\n";
    
    ParseOptions po(usage);
    BaseFloat snr = 10;

    po.Register("snr", &snr, "Add noise such that the output signal has this SNR in dB");

    po.Read(argc, argv);

    if (po.NumArgs() < 2 || po.NumArgs() > 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string wav_rspecifier = po.GetArg(1),
        wav_wspecifier = po.GetArg(2),
        noise_wspecifier = po.GetArg(3);

    int32 num_done = 0;
    
    SequentialTableReader<WaveHolder> wav_reader(wav_rspecifier);
    TableWriter<WaveHolder> wav_writer(wav_wspecifier);
    TableWriter<WaveHolder> noise_writer(noise_wspecifier);

    for (; !wav_reader.Done(); wav_reader.Next()) {
      std::string key = wav_reader.Key();
      std::stringstream noisy_key_ss;

      noisy_key_ss << key << "-" << snr << "dB";
      
      WaveData clean_wav = wav_reader.Value();
      int32 num_channels = clean_wav.NumChannels();
      int32 num_samples = clean_wav.NumSamples();
      BaseFloat samp_freq = clean_wav.SampFreq();

      Matrix<BaseFloat> data(clean_wav.Data());
      WaveData noise(samp_freq, num_samples, num_channels);

      for (int32 c = 0; c < num_channels; c++) {
        BaseFloat beta = pow(10, (clean_wav.MeanLoudness(c) - snr) / 20);
        SubVector<BaseFloat> this_channel_noise(noise.Data(), c);
        this_channel_noise.Scale(beta);
      }

      data.AddMat(1.0, noise.Data(), kNoTrans);

      WaveData noisy_wav(samp_freq, data);

      wav_writer.Write(noisy_key_ss.str(), noisy_wav);

      if (noise_wspecifier != "") 
        noise_writer.Write(noisy_key_ss.str(), noise);

      num_done++;
    }
    KALDI_LOG << "Added noise to " << num_done << " wave files " 
      << "and written noisy wave files.\n";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

