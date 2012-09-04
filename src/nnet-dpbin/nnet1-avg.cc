// nnet-dpbin/nnet1-avg.cc

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
#include "hmm/transition-model.h"
#include "nnet-dp/am-nnet1.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Average parameters of a set of neural network-based acoustic models\n"
        "Usage:  nnet1-avg [options] <nnet1-in1> <nnet1-int2> ... <nnet1-out>\n"
        "e.g.:\n"
        " nnet1-avg 1.nnet1 2.nnet1 avg.nnet1\n";
    
    bool binary_write = true;
    
    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    
    po.Read(argc, argv);

    if (po.NumArgs() < 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string
        nnet1_rxfilename = po.GetArg(1),
        nnet1_wxfilename = po.GetArg(po.NumArgs());

    TransitionModel trans_model;
    AmNnet1 am_nnet;
    {
      bool binary_read;
      Input ki(nnet1_rxfilename, &binary_read);
      trans_model.Read(ki.Stream(), binary_read);
      am_nnet.Read(ki.Stream(), binary_read);
    }
    
    for (int32 i = 2; i < po.NumArgs(); i++) {
      std::string next_nnet1_rxfilename = po.GetArg(i);
      AmNnet1 am_nnet_tmp;    
      {
        bool binary_read;
        Input ki(next_nnet1_rxfilename, &binary_read);
        trans_model.Read(ki.Stream(), binary_read); // overwrite trans_model.
        am_nnet_tmp.Read(ki.Stream(), binary_read);
      }
      am_nnet.Nnet().CombineWithWeight(am_nnet_tmp.Nnet(), 1.0 / i);
    }
    
    {
      Output ko(nnet1_wxfilename, binary_write);
      trans_model.Write(ko.Stream(), binary_write);
      am_nnet.Write(ko.Stream(), binary_write);
    }
    KALDI_LOG << "Wrote model to " << nnet1_wxfilename;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


