// nnet2bin/nnet-combine.cc

// Copyright 2012  Johns Hopkins University (author:  Daniel Povey)

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
#include "hmm/transition-model.h"
#include "nnet2/nnet-randomize.h"
#include "nnet2/combine-nnet.h"
#include "nnet2/am-nnet.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet2;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Using a validation set, compute an optimal combination of a number of\n"
        "neural nets (the combination weights are separate for each layer and\n"
        "do not have to sum to one).  The optimization is BFGS, which is initialized\n"
        "from the best of the individual input neural nets (or as specified by\n"
        "--initial-model)\n"
        "This version takes 'raw' neural nets as inputs, i.e. just the Nnet object,\n"
        "see nnet-am-combine which takes the .mdl file (including TransitionModel etc.)\n"
        "\n"
        "Usage:  nnet-combine [options] <nnet-in1> <nnet-in2> ... <nnet-inN> <valid-examples-in> <nnet-out>\n"
        "\n"
        "e.g.:\n"
        " nnet-combine 1.1.nnet 1.2.nnet 1.3.nnet ark:valid.egs 2.nnet\n"
        "Caution: the first input neural net must not be a gradient.\n"
        "See also: nnet-am-combine, nnet-average\n";

    bool binary_write = true;
    NnetCombineConfig combine_config;
    
    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    
    combine_config.Register(&po);
    
    po.Read(argc, argv);
    
    if (po.NumArgs() < 3) {
      po.PrintUsage();
      exit(1);
    }
    
    std::string
        nnet1_rxfilename = po.GetArg(1),
        valid_examples_rspecifier = po.GetArg(po.NumArgs() - 1),
        nnet_wxfilename = po.GetArg(po.NumArgs());
    
    Nnet nnet1;

    int32 num_nnets = po.NumArgs() - 2;
    std::vector<Nnet> nnets(num_nnets);

    
    for (int32 n = 0; n < num_nnets; n++)
      ReadKaldiObject(po.GetArg(n + 1), &(nnets[0]));


    std::vector<NnetExample> validation_set; // stores validation frames [may or
                                             // may not really be held out,
                                             // depends on script.]

    { // This block adds samples to "validation_set".
      SequentialNnetExampleReader example_reader(
          valid_examples_rspecifier);
      for (; !example_reader.Done(); example_reader.Next())
        validation_set.push_back(example_reader.Value());
      KALDI_LOG << "Read " << validation_set.size() << " examples from the "
                << "validation set.";
      KALDI_ASSERT(validation_set.size() > 0);
    }

    Nnet nnet_out;
    
    CombineNnets(combine_config,
                 validation_set,
                 nnets,
                 &nnet_out);

    WriteKaldiObject(nnet_out, nnet_wxfilename, binary_write);
    
    KALDI_LOG << "Finished combining [raw] neural nets, wrote model to "
              << nnet_wxfilename;
    return (validation_set.size() == 0 ? 1 : 0);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


