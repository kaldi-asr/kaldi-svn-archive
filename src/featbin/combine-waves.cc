// featbin/combine-waves.cc

// Copyright    2014 Pegah ghahremani

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


namespace kaldi {

/* 
  This function combines two recieved waves into a single wave.
  In this combining technique, we combine two waves with the first one being 
  greater by "s_factor". 
  If e1, and e2 would be energy of 1st and 2nd wave, the second wave is weighted 
  w.r.t specific signal gain sqrt(e1 / ("s_factor" * e2)) and then two signals are summed.
  So the enegry of 1st wave is "s_factor" times greater than the energy of 2nd one in combined wave.
  Also the lenght of the out_wave is the length of larger wave and the smaller wave is 
  is extended with zeros at the end.
*/
void CombineWaves(const VectorBase<BaseFloat> &wave1, 
             const VectorBase<BaseFloat> &wave2,
             BaseFloat s_factor,
             Matrix<BaseFloat> *out_wave) {
  int32 length1 = wave1.Dim(), length2 = wave2.Dim(),
    max_length = std::max(length1, length2);
  Vector<BaseFloat> wave1_tmp(wave1), wave2_tmp(wave2);
  // padd zero to wavefiles with smaller size. 
  // out_wave size is equal to larger wavefile.
  wave1_tmp.Resize(max_length, kCopyData);
  wave2_tmp.Resize(max_length, kCopyData);
  out_wave->Resize(1, max_length);

  // Compute energy of two waves
  BaseFloat energy1 = VecVec(wave1, wave1),
    energy2 = VecVec(wave2, wave2);
  
  // Compute scale_factor to scale second wave, so 1st wavefile would be s_factor times greater.
  // scale_factor = 0, out_wave is equal to the 1st wave.
  BaseFloat scale_factor = (s_factor == 0 ? 0 : std::pow(energy1 / (energy2 * s_factor), 0.5));
  out_wave->AddVecToRows(1, wave1_tmp);    
  if (scale_factor != 0) out_wave->AddVecToRows(scale_factor, wave2_tmp);    
}
} // namespace kaldi

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    const char *usage = 
      "Combines two utterance's wavefile into single wave,\n"
      "where, energy of the 1st wave is energy-factor times greater than\n"
      "the energy of 2nd wave in combined wave.\n"
      "If e1, and e2 would be energy of 1st and 2nd wave, the second wave is weighted\n"
      "w.r.t specific signal gain sqrt(e1 / (energy-factor * e2)) and then two signals are summed.\n"
      "The combined utterance id is (1st speaker)_(1st spoken text)_(2nd spoken text).\n"
      "\n"
      "Usage: combine-waves [options...] <wav1-rspecifier> <wav2-rspecifier> <wav-wspecifier>\n"
      "e.g.. combine-waves --energy-factor=10 scp:wav1.scp scp:wav2.scp ark:-\n";

    ParseOptions po(usage);
    
    BaseFloat energy_factor = 100;
    
    po.Register("energy-factor", &energy_factor, "Energy of 1st wave is energy_factor "
                "times greater than the energy of 2nd wave in combined wave."
                "if 0, the combined wave is equal to the 1st wave.");
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }
    std::string wav1_rspecifier = po.GetArg(1),
      wav2_rspecifier = po.GetArg(2),
      wav_wspecifier = po.GetArg(3);
    int32 num_done = 0, num_err = 0;

    SequentialTableReader<WaveHolder> wav1_reader(wav1_rspecifier),
      wav2_reader(wav2_rspecifier);
    TableWriter<WaveHolder> wav_writer(wav_wspecifier);     
    

    for (; !wav1_reader.Done(), !wav2_reader.Done(); wav1_reader.Next(), wav2_reader.Next()) {
      // Create utterance id for comined wave.
      std::string utt1 = wav1_reader.Key(),
        utt2 = wav2_reader.Key();

      // Split the  line by "_" and check number of fileds in each line. 
      // There must be 3 fields name. 
      std::vector<std::string> split_utt1, split_utt2;
      SplitStringToVector(utt1, "_", true, &split_utt1);
      SplitStringToVector(utt2, "_", true, &split_utt2); 
      KALDI_ASSERT(split_utt1.size() == split_utt2.size());
      if (split_utt1.size() != 3) {
        KALDI_WARN << "Invalid utterance id: " << utt1;
        continue;
      }
      std::string combined_utt = split_utt1[0] + "_" + split_utt1[1] + "_" + split_utt2[1];
     
      // Combine two wavefiles with energy factor "energy_factor"
      const WaveData &wav1_data = wav1_reader.Value(),
        &wav2_data = wav2_reader.Value();
      KALDI_ASSERT(wav1_data.SampFreq() == wav2_data.SampFreq());
      int32 num_cols = std::max(wav1_data.Data().NumCols(), wav2_data.Data().NumCols());
      Matrix<BaseFloat> combined_wave_mat(1, num_cols);
      SubVector<BaseFloat> waveform1(wav1_data.Data(), 0), 
        waveform2(wav2_data.Data(), 0);
      CombineWaves(waveform1, waveform2, energy_factor, &combined_wave_mat);
      WaveData combined_wave(wav1_data.SampFreq(), combined_wave_mat);
      wav_writer.Write(combined_utt, combined_wave);
      num_done++;
    }
    KALDI_LOG << "wavefiles combined for " << num_done << "utterances";
    return (num_done != 0 ? 0 : 1);  
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
