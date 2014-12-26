// sgmm2bin/sgmm2-compute.cc

// Copyright 2014  Xiaohui Zhang

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
#include "sgmm2/am-sgmm2.h"
#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-device.h"
#include "cudamatrix/cu-value.h"
#include "cudamatrix/cu-vector.h"
#include "hmm/transition-model.h"
#include "hmm/hmm-utils.h"
#include "nnet/nnet-pdf-prior.h"
#include "fstext/fstext-lib.h"
#include "decoder/faster-decoder.h"
#include "decoder/training-graph-compiler.h"
#include "sgmm2/decodable-am-sgmm2.h"
#include "lat/kaldi-lattice.h" // for {Compact}LatticeArc


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "Get log-likelihoods (in a matrix form) of features given SGMM models.\n"
        "Usage: sgmm2-compute [options] model-in graphs-rspecifier "
        "feature-rspecifier prob-matrix-wspecifier\n"
        "e.g.: sgmm2-compute 1.mdl ark:graphs.fsts scp:train.scp ark:1.mat\n";

    ParseOptions po(usage);
    
    bool binary = true;
    BaseFloat transition_scale = 1.0;
    BaseFloat self_loop_scale = 1.0;
    BaseFloat log_prune = 5.0;
    int32 num_pdfs = 0;
    std::string gselect_rspecifier, spkvecs_rspecifier, utt2spk_rspecifier;
    
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("num-pdfs", &num_pdfs, "Number of pdfs in the sgmm system");
    po.Register("log-prune", &log_prune, "Pruning beam used to reduce number "
                "of exp() evaluations.");
    po.Register("spk-vecs", &spkvecs_rspecifier, "Speaker vectors (rspecifier)");
    po.Register("utt2spk", &utt2spk_rspecifier,
                "rspecifier for utterance to speaker map");
    po.Register("transition-scale", &transition_scale, "Scaling factor for "
                "some transition probabilities [see also self-loop-scale].");
    po.Register("self-loop-scale", &self_loop_scale, "Scaling factor for "
                "self-loop versus non-self-loop probability mass [controls "
                "most transition probabilities.]");
    po.Register("gselect", &gselect_rspecifier, "Precomputed Gaussian indices "
                "(rspecifier)");

    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    if (gselect_rspecifier == "")
      KALDI_ERR << "--gselect option is mandatory.";
    
    FasterDecoderOptions decode_opts;

    std::string model_in_filename = po.GetArg(1);
    std::string fst_rspecifier = po.GetArg(2);
    std::string feature_rspecifier = po.GetArg(3);
    std::string post_wspecifier = po.GetArg(4);

    TransitionModel trans_model;
    AmSgmm2 am_sgmm;
    {
      bool binary;
      Input ki(model_in_filename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_sgmm.Read(ki.Stream(), binary);
    }

    SequentialTableReader<fst::VectorFstHolder> fst_reader(fst_rspecifier);
    RandomAccessBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessInt32VectorVectorReader gselect_reader(gselect_rspecifier);

    RandomAccessBaseFloatVectorReaderMapped spkvecs_reader(spkvecs_rspecifier,
                                                           utt2spk_rspecifier);
    BaseFloatMatrixWriter writer(post_wspecifier);

    int num_done = 0, num_err = 0;
    kaldi::int64 frame_count = 0;

    for (; !fst_reader.Done(); fst_reader.Next()) {
      std::string utt = fst_reader.Key();
      if (!feature_reader.HasKey(utt)) num_err++;
      else {
        VectorFst<StdArc> decode_fst(fst_reader.Value());
        // stops copy-on-write of the fst by deleting the fst inside the reader,
        // since we're about to mutate the fst by adding transition probs.
        fst_reader.FreeCurrent();

        const Matrix<BaseFloat> &features = feature_reader.Value(utt);
        if (features.NumRows() == 0) {
          KALDI_WARN << "Zero-length utterance: " << utt;
          num_err++;
          continue;
        }

        Sgmm2PerSpkDerivedVars spk_vars;
        if (spkvecs_reader.IsOpen()) {
          if (spkvecs_reader.HasKey(utt)) {
            spk_vars.SetSpeakerVector(spkvecs_reader.Value(utt));
            am_sgmm.ComputePerSpkDerivedVars(&spk_vars);
          } else {
            KALDI_WARN << "Cannot find speaker vector for " << utt;
            num_err++;
            continue;
          }
        }  // else spk_vars is "empty"

        if (!gselect_reader.HasKey(utt)
            && gselect_reader.Value(utt).size() != features.NumRows()) {
          KALDI_WARN << "No Gaussian-selection info available for utterance "
                     << utt << " (or wrong size)";
          num_err++;
        }
        const std::vector<std::vector<int32> > &gselect =
            gselect_reader.Value(utt);

        if (decode_fst.Start() == fst::kNoStateId) {
          KALDI_WARN << "Empty decoding graph for " << utt;
          num_err++;
          continue;
        }
        
        {  // Add transition-probs to the FST.
          std::vector<int32> disambig_syms;  // empty.
          AddTransitionProbs(trans_model, disambig_syms,
                             transition_scale, self_loop_scale,
                             &decode_fst);
        }

        FasterDecoder decoder(decode_fst, decode_opts);
        
        DecodableAmSgmm2Scaled sgmm_decodable(am_sgmm, trans_model, features, gselect,
                                              log_prune, 1.0, &spk_vars);

        Matrix<BaseFloat> loglike(num_pdfs, features.NumRows());
        for (int32 i = 0; i < features.NumRows(); i++) {
          for (int32 j = 0; j < num_pdfs; j++) {
            loglike(i, j) = sgmm_decodable.LogLikelihoodForPdf(i, j);
          }  
        }      

        writer.Write(utt, loglike);
        frame_count += features.NumRows();
        num_done++;

        if (num_done % 50  == 0) {
            KALDI_LOG << "Processed " << num_done << " utterances. ";
        }
      }
    }

    KALDI_LOG << "Done " << num_done << ", errors on " << num_err;
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


