// bin/add-vec-to-rows.cc

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
#include "matrix/kaldi-matrix.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Add a vector to each row of the input matrices\n "
        "Can be used when adding log priors to SGMM log-likelihoods to get SGMM posteriors.\n"
        "Usage: add-vec-to-rows <vector-rxfilename> <matrix-rspecifier> <matrix-wspecifier>\n";
        "e.g.: add-vec-to-rows prior.vec ark:loglike.mat ark:logpost.mat\n";

    ParseOptions po(usage);
    BaseFloat scale = 1.0;

    po.Register("scale", &scale, "Scaling factor for vectors");
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string vec_rfilename = po.GetArg(1);
    std::string mat_rspecifier = po.GetArg(2);
    std::string mat_wspecifier = po.GetArg(3);

    Vector<BaseFloat> log_prior;

    {
      bool binary;
      Input ki(vec_rfilename, &binary);
      log_prior.Read(ki.Stream(), binary);
    }

    SequentialBaseFloatMatrixReader mat_reader(mat_rspecifier);
    BaseFloatMatrixWriter mat_writer(mat_wspecifier);

    int32 num_done = 0;

    for (; !mat_reader.Done(); mat_reader.Next()) {
      std::string key = mat_reader.Key();
      Matrix<BaseFloat> mat(mat_reader.Value());
      KALDI_ASSERT(mat.NumCols() == log_prior.Dim());
      mat.AddVecToRows(1.0, log_prior);
      // Do the summation in double, to minimize roundoff.
      mat_writer.Write(key, mat);
      num_done++;
    }

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


