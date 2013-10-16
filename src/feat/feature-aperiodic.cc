// feat/feature-aperiodic.cc

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


#include "feat/feature-aperiodic.h"


namespace kaldi {

AperiodicEnergy::AperiodicEnergy(const AperiodicEnergyOptions &opts)
    : opts_(opts), feature_window_function_(opts.frame_opts),
      srfft_(NULL), mel_banks_(NULL) {
  int32 window_size = opts.frame_opts.WindowSize();
  // If the signal window size is N, then the number of points in the FFT
  // computation is the smallest power of 2 that is greater than or equal to 2N
  padded_window_size_ = RoundUpToNearestPowerOfTwo(window_size*2);
  if (padded_window_size_ <=
      static_cast<int32>(opts_.frame_opts.samp_freq/opts_.f0_min)) {
    KALDI_ERR << "Padded window size (" << padded_window_size_ << ") too small "
              << " to capture F0 range.";
  }
  srfft_ = new SplitRadixRealFft<BaseFloat>(padded_window_size_);

  FrameExtractionOptions mel_frame_opts = opts_.frame_opts;
  mel_frame_opts.frame_length_ms =
      (static_cast<BaseFloat>(padded_window_size_)
          * 1000.0) / mel_frame_opts.samp_freq;
  mel_banks_ = new MelBanks(opts_.mel_opts, mel_frame_opts, 1.0);
}

AperiodicEnergy::~AperiodicEnergy() {
  if (srfft_ == NULL) {
    KALDI_WARN << "NULL srfft_ pointer: This should not happen if the class "
               << "was used properly";
    return;
  }
  delete srfft_;
  if (mel_banks_ == NULL) {
    KALDI_WARN << "NULL mel_banks_ pointer: This should not happen if the class "
               << "was used properly";
    return;
  }
  delete mel_banks_;
}

// Note that right now the number of frames in f0 and voicing_prob may be
// different from what Kaldi extracts (due to different windowing in ESPS
// get_f0). There is code to ignore some trailing frames if the difference
// is not large, but those code should be removed once we have a native Kaldi
// F0 extractor.
void AperiodicEnergy::Compute(const VectorBase<BaseFloat> &wave,
                              const VectorBase<BaseFloat> &voicing_prob,
                              const VectorBase<BaseFloat> &f0,
                              Matrix<BaseFloat> *output,
                              Vector<BaseFloat> *wave_remainder) {
  KALDI_ASSERT(output != NULL);
  KALDI_ASSERT(srfft_ != NULL &&
               "srfft_ must not be NULL if class is initialized properly.");
  KALDI_ASSERT(mel_banks_ != NULL &&
               "mel_banks_ must not be NULL if class is initialized properly.");
  int32 frames_out = NumFrames(wave.Dim(), opts_.frame_opts),
      dim_out = opts_.mel_opts.num_bins;
  if (frames_out == 0)
    KALDI_ERR << "No frames fit in file (#samples is " << wave.Dim() << ")";
  //  if (voicing_prob.Dim() != frames_out) {
  // The following line should be replaced with the line above eventually
  if (std::abs(voicing_prob.Dim() - frames_out) > opts_.frame_diff_tolerance) {
    KALDI_ERR << "#frames in probability of voicing vector ("
              << voicing_prob.Dim() << ") doesn't match #frames in data ("
              << frames_out << ").";
  }
  //  if (f0.Dim() != frames_out) {
  // The following line should be replaced with the line above eventually
  if (std::abs(f0.Dim() - frames_out) > opts_.frame_diff_tolerance) {
    KALDI_ERR << "#frames in F0 vector (" << f0.Dim() << ") doesn't match "
              << "#frames in data (" << frames_out << ").";
  }

  frames_out = std::min(frames_out, f0.Dim());  // will be removed eventually
  output->Resize(frames_out, dim_out);
  if (wave_remainder != NULL)
    ExtractWaveformRemainder(wave, opts_.frame_opts, wave_remainder);
  Vector<BaseFloat> wave_window;
  Vector<BaseFloat> padded_window(padded_window_size_, kUndefined);
  Vector<BaseFloat> binned_energies(dim_out);

  for (int32 r = 0; r < frames_out; r++) {  // r is frame index
    SubVector<BaseFloat> this_ap_energy(output->Row(r));
    ExtractWindow(wave, r, opts_.frame_opts, feature_window_function_,
                  &wave_window, NULL);
    int32 window_size = wave_window.Dim();
    padded_window.SetZero();
    padded_window.Range(0, window_size).CopyFromVec(wave_window);
    srfft_->Compute(padded_window.Data(), true);

    Vector<BaseFloat> tmp_spectrum(padded_window);
    ComputePowerSpectrum(&tmp_spectrum);
    SubVector<BaseFloat> power_spectrum(tmp_spectrum, 0,
                                        padded_window_size_/2 + 1);

    if (voicing_prob(r) == 0.0) {  // unvoiced region
      // While the aperiodic energy will not be used during synthesis of the
      // unvoiced regions, it is still modeled as a separate stream in the HMM
      // and so we set it to the log-filterbank values.
      mel_banks_->Compute(power_spectrum, &binned_energies);
      binned_energies.ApplyLog();
      this_ap_energy.CopyFromVec(binned_energies);
      continue;
    }

    std::vector<bool> noise_indices;
    IdentifyNoiseRegions(power_spectrum, f0(r), &noise_indices);

    // padded_window contains the FFT coefficients
    ObtainNoiseSpectrum(padded_window, noise_indices, &tmp_spectrum);
    SubVector<BaseFloat> noise_spectrum(tmp_spectrum, 0,
                                        padded_window_size_/2 + 1);
    mel_banks_->Compute(noise_spectrum, &binned_energies);
    binned_energies.ApplyLog();  // take the log
    this_ap_energy.CopyFromVec(binned_energies);
  }
}


// For voiced regions, we reconstruct the harmonic spectrum from the cepstral
// coefficients around the cepstral peak corresponding to F0, and noise spectrum
// using the rest of the cepstral coefficients. The frequency samples for which
// the noise spectrum has a higher value than the harmonic spectrum are returned.
void AperiodicEnergy::IdentifyNoiseRegions(
    const VectorBase<BaseFloat> &power_spectrum,
    BaseFloat f0,
    std::vector<bool> *noise_indices) {
  KALDI_ASSERT(noise_indices != NULL);
  KALDI_ASSERT(power_spectrum.Dim() == padded_window_size_/2+1 && "Power "
               "spectrum size expected to be half of padded window plus 1.")
  BaseFloat sampling_freq = opts_.frame_opts.samp_freq;
  int32 f0_index = static_cast<int32>(sampling_freq/f0),
      max_f0_index = static_cast<int32>(sampling_freq/opts_.f0_max),
      peak_index;
  noise_indices->resize(padded_window_size_/2+1, false);

  Vector<BaseFloat> noise_spectrum(padded_window_size_, kSetZero);
  noise_spectrum.Range(0, padded_window_size_/2+1).CopyFromVec(power_spectrum);
  PowerSpecToRealCeps(&noise_spectrum);

  // All cepstral coefficients below the one corresponding to maximum F0 are
  // considered to correspond to vocal tract characteristics and ignored from
  // the initial harmonic to noise ratio calculation.
  noise_spectrum.Range(0, max_f0_index-1).SetZero();
  noise_spectrum.Max(&peak_index);
//  peak_index += max_f0_index;
  if (peak_index < f0_index-1 || peak_index > f0_index+1) {
    KALDI_LOG << "Actual cepstral peak (index=" << peak_index << "; value = "
              << noise_spectrum(peak_index) << ") occurs too far from F0 (index="
              << f0_index << "; value = " << noise_spectrum(f0_index) << ").";
//    f0_index = peak_index;  // TODO(arnab): remove this: it is only for testing.
  }

  Vector<BaseFloat> harmonic_spectrum(padded_window_size_, kSetZero);
  // Note that at this point noise_spectrum contains cepstral coeffs
  for (int32 i = f0_index-0; i <= f0_index+0; ++i) {
    harmonic_spectrum(i) = noise_spectrum(i);
    noise_spectrum(i) = 0.0;
  }
  RealCepsToMagnitudeSpec(&harmonic_spectrum, false /* get log spectrum*/);
  RealCepsToMagnitudeSpec(&noise_spectrum, false /* get log spectrum*/);

  for (int32 i = 0; i <= padded_window_size_/2; ++i)
    if (noise_spectrum(i) > harmonic_spectrum(i))
      (*noise_indices)[i] = true;
}

void AperiodicEnergy::ObtainNoiseSpectrum(
    const VectorBase<BaseFloat> &fft_coeffs,
    const std::vector<bool> &noise_indices,
    Vector<BaseFloat> *noise_spectrum) {
  KALDI_ASSERT(noise_spectrum != NULL);
  KALDI_ASSERT(noise_spectrum->Dim() == padded_window_size_);

  noise_spectrum->SetZero();
  Vector<BaseFloat> prev_estimate(padded_window_size_);
  int32 window_size = opts_.frame_opts.WindowSize();
  for (int32 iter = 0; iter < opts_.max_iters; ++iter) {
    // Copy the DFT coefficients for the noise-regions
    for (int32 j = 0; j < padded_window_size_/2; ++j) {
      if (noise_indices[j]) {
        (*noise_spectrum)(j*2) = fft_coeffs(j*2);
        (*noise_spectrum)(j*2+1) = fft_coeffs(j*2+1);
      }
    }

    srfft_->Compute(noise_spectrum->Data(), false /*do IFFT*/);
    BaseFloat ifft_scale = 1.0/padded_window_size_;
    noise_spectrum->Scale(ifft_scale);
    noise_spectrum->Range(window_size,
                          padded_window_size_-window_size).SetZero();
    if (iter > 0) {  // calculate the squared error (in time domain)
      prev_estimate.AddVec(-1.0, *noise_spectrum);
      BaseFloat err = prev_estimate.SumPower(2.0) / ifft_scale;
      KALDI_LOG << "Iteration " << iter
                << ": Aperiodic component squared error = " << err;
      if (err < opts_.min_sq_error) {  // converged
        // noise_spectrum is still in time domain; convert to frequency domain
        srfft_->Compute(noise_spectrum->Data(), true /*do FFT*/);
        break;
      }
    }
    prev_estimate.CopyFromVec(*noise_spectrum);

    srfft_->Compute(noise_spectrum->Data(), true /*do FFT*/);
  }
  ComputePowerSpectrum(noise_spectrum);
}

}  // namespace kaldi
