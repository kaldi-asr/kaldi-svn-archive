// nnet-dpbin/nnet1-train.cc

// Copyright 2012  Johns Hopkins University (author:  Daniel Povey)

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
#include "nnet-dp/am-nnet1.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Train neural network (of type nnet1).\n"
        "Adaptive training that adjusts learning rates periodically using\n"
        "validation set\n"
        "Usage:  nnet1-train [options] <nnet1-in> <features-rspecifier> <alignments-rspecifier> <validation-utt-list> <nnet1-out>\n"
        "e.g.:\n"
        " nnet1-train exp/nnet1/1.nnet1 'scp:data/train/feats.scp' 'ark:gunzip -c exp/tri1/{?,??}.ali.gz|'  "
        "exp/nnet1/valid.uttlist exp/nnet1/2.nnet1\n";
        
    bool binary_write = true;
    Nnet1InitConfig config;
    
    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    config.Register(&po);
    
    po.Read(argc, argv);

    if (po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }

    std::string nnet1_rxfilename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        alignments_rspecifier = po.GetArg(3),
        validation_utt_list = po.GetArg(4),
        nnet1_wxfilename = po.GetArg(5);
    

    TransitionModel trans_model;
    AmNnet1 am_nnet1;

    {
      bool binary_read;
      Input ki(nnet1_rxfilename, &binary_read);
      trans_model.Read(ko.Stream(), binary_read);
      am_nnet1.Read(ko.Stream(), binary_read);
    }
    

    WriteKaldiObject(am_nnet1, nnet1_wxfilename, binary_write);
        
    KALDI_LOG << "Wrote model to " << nnet1_wxfilename;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


