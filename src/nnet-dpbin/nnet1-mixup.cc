// nnet-dpbin/nnet1-mixup.cc

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
        "Mix up neural net of type nnet1, adding Gaussians.\n"
        "Usage:  nnet1-mix-up [options] <nnet1-in> <nnet1-out>\n"
        "e.g.:\n"
        " nnet1-mix-up 1.nnet1 2.nnet2\n";
        
    bool binary_write = true;
    int32 target_neurons = 0;
    BaseFloat power = 0.2;
    BaseFloat perturb_stddev = 0.1;
    
    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("target-neurons", &target_neurons, "Target total number of output "
                "units of softmax layers");
    po.Register("power", &power, "Constant that determines how Gaussians are "
                "allocated among the different output layers (using a power "
                "of occupation count)");
    po.Register("perturb-stddev", &perturb_stddev, "Amount by which to perturb "
                "duplicated rows of the parameter matrix (similar to perturbed "
                "mean in Gaussian mix-up).");
    
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string nnet1_rxfilename = po.GetArg(1),
        nnet1_wxfilename = po.GetArg(2);
    
    TransitionModel trans_model;
    AmNnet1 am_nnet;
    {
      bool binary_read;
      Input ki(nnet1_rxfilename, &binary_read);
      trans_model.Read(ki.Stream(), binary_read);
      am_nnet.Read(ki.Stream(), binary_read);
    }

    am_nnet.Nnet().MixUp(target_neurons, power, perturb_stddev);
    
    {
      Output ko(nnet1_wxfilename, binary_write);
      trans_model.Write(ko.Stream(), binary_write);
      am_nnet.Write(ko.Stream(), binary_write);
    }
    
    KALDI_LOG << "Mixed up model and wrote to " << nnet1_wxfilename;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


