// featbin/compute-irm-targets.cc

// Copyright 2014 Vimal Manohar

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
#include "matrix/kaldi-matrix.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Compute soft targets for Ideal Ratio Mask used for "
        "TF-Masking \n"
        "The soft function maps the SNR to a sigmoid"
        "d(t,f) = 1/1+exp(-alpha(SNR(t,f) - beta))"
        "Usage: compute-irm-targets [options] (<clean-rspecifier> <noise-rspecifier> <target-wspecifier>) | (<clean-wxfilename> <noise-rxfilename> <target-wxfilename>)\n"
        "e.g.: compute-irm-targets scp:clean_feats.scp scp:noise_feats.scp ark,scp:irm_targets.ark,irm_targets.scp\n";

    ParseOptions po(usage);
    bool binary = true;
    bool compress = false;
    BaseFloat beta = -6;      // in dB
    BaseFloat snr_span = 35;  // in dB
    

    po.Register("binary", &binary, "Binary-mode output (not relevant if writing "
                "to archive)");
    po.Register("compress", &compress, "If true, write output in compressed form"
                "(only currently supported for wxfilename, i.e. archive/script,"
                "output)");
    po.Register("beta", &beta, "Shift the target sigmoid to be centered at beta (in dB)");
    po.Register("snr-span", &snr_span, "The difference between SNR values that correspond to target values of 0.05 and 0.95");
    
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    if (snr_span <= 0) {
      KALDI_ERR << "--snr-span is expected to be > 0. But it is given " << snr_span;
    }

    BaseFloat alpha = 2 * log10(19) / snr_span;

    int32 num_done = 0, num_missing = 0, num_mismatch = 0;
    
    if (ClassifyRspecifier(po.GetArg(1), NULL, NULL) != kNoRspecifier) {
      // Copying tables of features.
      std::string clean_rspecifier = po.GetArg(1);
      std::string noise_rspecifier = po.GetArg(2);
      std::string target_wspecifier = po.GetArg(3);

      if (!compress) {
        BaseFloatMatrixWriter target_writer(target_wspecifier);
        SequentialBaseFloatMatrixReader clean_reader(clean_rspecifier);
        RandomAccessBaseFloatMatrixReader noise_reader(noise_rspecifier);

        for (; !clean_reader.Done(); clean_reader.Next()) {
          std::string key = clean_reader.Key();
          Matrix<BaseFloat> clean_feats = clean_reader.Value();
          if (!noise_reader.HasKey(key)) {
            KALDI_WARN << "Missing noise features for utterance " << key;
            num_missing++;
            continue;
          }
          Matrix<BaseFloat> noise_feats = noise_reader.Value(key);

          int32 num_frames = clean_feats.NumRows();
          int32 dim = clean_feats.NumCols();

          if (num_frames != noise_feats.NumRows()) {
            KALDI_WARN << "Mismatch in number of frames for clean and noise features for utterance " << key << ": \n"
              << num_frames << " vs " << noise_feats.NumRows() << ". " 
              << "Skippking utterance";
            num_mismatch++;
            continue;
          }
          
          if (dim != noise_feats.NumCols()) {
            KALDI_WARN << "Mismatch in feature dimension for clean and noise features for utterance " << key << ": \n"
              << dim << " vs " << noise_feats.NumCols() << ". " 
              << "Skippking utterance";
            num_mismatch++;
            continue;
          }
        
          Matrix<BaseFloat> target_irm(num_frames, dim);

          for (int32 i = 0; i < num_frames; i++) {
            for (int32 j = 0; j < dim; j++) {
              target_irm(i,j) = 1 / (1 + pow(clean_feats(i,j) / noise_feats(i,j), -alpha * exp(1)) * exp(alpha * beta));
            }
          }
          
          target_writer.Write(key, target_irm);
          num_done++;
        }
      } else {
        CompressedMatrixWriter target_writer(target_wspecifier);
        SequentialBaseFloatMatrixReader clean_reader(clean_rspecifier);
        RandomAccessBaseFloatMatrixReader noise_reader(noise_rspecifier);

        for (; !clean_reader.Done(); clean_reader.Next()) {
          std::string key = clean_reader.Key();
          Matrix<BaseFloat> clean_feats = clean_reader.Value();
          if (!noise_reader.HasKey(key)) {
            KALDI_WARN << "Missing noise features for utterance " << key;
            num_missing++;
            continue;
          }
          Matrix<BaseFloat> noise_feats = noise_reader.Value(key);

          int32 num_frames = clean_feats.NumRows();
          int32 dim = clean_feats.NumCols();

          if (num_frames != noise_feats.NumRows()) {
            KALDI_WARN << "Mismatch in number of frames for clean and noise features for utterance " << key << ": \n"
              << num_frames << " vs " << noise_feats.NumRows() << ". " 
              << "Skippking utterance";
            num_mismatch++;
            continue;
          }
          
          if (dim != noise_feats.NumCols()) {
            KALDI_WARN << "Mismatch in feature dimension for clean and noise features for utterance " << key << ": \n"
              << dim << " vs " << noise_feats.NumCols() << ". " 
              << "Skippking utterance";
            num_mismatch++;
            continue;
          }
        
          Matrix<BaseFloat> target_irm(num_frames, dim);

          for (int32 i = 0; i < num_frames; i++) {
            for (int32 j = 0; j < dim; j++) {
              target_irm(i,j) = 1 / (1 + pow(clean_feats(i,j) / noise_feats(i,j), -alpha * exp(1)) * exp(alpha * beta));
            }
          }
          
          target_writer.Write(key, CompressedMatrix(target_irm));
          num_done++;
        }
      }
      KALDI_LOG << "Computed IRM targets for " << num_done << " feature matrices, " 
        << "missing noise features for " << num_missing << " feature matrices, "
        << "mismatch of noise features for " << num_mismatch << " feature matrices.";
      return (num_done > num_missing + num_mismatch ? 0 : 1);
    } else {
      KALDI_ASSERT(!compress && "Compression not yet supported for single files");
      
      std::string clean_rxfilename = po.GetArg(1),
        noise_rxfilename = po.GetArg(2),
        target_wxfilename = po.GetArg(3);

      Matrix<BaseFloat> clean_matrix;
      ReadKaldiObject(clean_rxfilename, &clean_matrix);
      Matrix<BaseFloat> noise_matrix;
      ReadKaldiObject(noise_rxfilename, &noise_matrix);
          
      int32 num_frames = clean_matrix.NumRows();
      int32 dim = clean_matrix.NumCols();

      if (num_frames != noise_matrix.NumRows()) {
        KALDI_ERR << "Mismatch in number of frames for clean and noise feature matrices: \n"
          << num_frames << " vs " << noise_matrix.NumRows() << ".";
      }

      if (dim != noise_matrix.NumCols()) {
        KALDI_ERR << "Mismatch in feature dimension for clean and noise feature matrices: \n"
          << dim << " vs " << noise_matrix.NumCols() << ".";
      }

      Matrix<BaseFloat> target_irm(num_frames, dim);

      for (int32 i = 0; i < num_frames; i++) {
        for (int32 j = 0; j < dim; j++) {
          target_irm(i,j) = 1 / (1 + pow(clean_matrix(i,j) / noise_matrix(i,j), -alpha * exp(1)) * exp(alpha * beta));
        }
      }

      WriteKaldiObject(target_irm, target_wxfilename, binary);
      KALDI_LOG << "Computed IRM target from " 
        << "clean matrix " << clean_rxfilename << " and "
        << "noise matrix " << noise_rxfilename << ". "
        << "Wrote target to " << target_wxfilename;

      return 0;
    }
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


