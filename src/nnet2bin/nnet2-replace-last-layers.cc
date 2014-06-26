// nnet2bin/nnet2-replace-last-layers.cc

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
#include "nnet2/am-nnet.h"
#include "nnet2/nnet-functions.h"
#include "hmm/transition-model.h"
#include "tree/context-dep.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet2;
    typedef kaldi::int32 int32;

    const char *usage =
        "This program is for adding new layers to a neural-network acoustic model.\n"
        "It removes the last --remove-layers layers, and adds the layers from the\n"
        "supplied raw-nnet.  The typical use is to remove the last two layers\n"
        "(the softmax, and the affine component before it), and add in replacements\n"
        "for them newly initialized by nnet2-init.  This program is a more flexible\n"
        "way of adding layers than nnet2-insert, but the inserted network needs to\n"
        "contain replacements for the removed layers.  This program by default reads/writes\n"
        "model (.mdl) files, but with the --raw option can also work with 'raw' neural\n"
        "nets.\n"
        "\n"
        "Usage:  nnet2-replace-last-layers [options] <model-in> <nnet-to-insert-in> <model-out>\n"
        "Usage:  nnet2-replace-last-layers --raw [options] <nnet-in> <nnet-to-insert-in> <nnet-out>\n"
        "e.g.:\n"
        " nnet2-replace-last-layers 1.mdl \"nnet2-init hidden_layer.config -|\" 2.mdl\n";

    bool binary_write = true;
    bool raw = false;
    int32 remove_layers = 2;

    ParseOptions po(usage);
    
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("raw", &raw, "If true, this program reads/writes raw "
                "neural nets");
    po.Register("remove-layers", &remove_layers, "Number of final layers "
                "to remove before adding input raw network.");
    
    po.Read(argc, argv);
    
    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string nnet_rxfilename = po.GetArg(1),
        raw_nnet_rxfilename = po.GetArg(2),
        nnet_wxfilename = po.GetArg(3);
    
    TransitionModel trans_model;
    AmNnet am_nnet;
    Nnet nnet;
    if (!raw) {
      bool binary;
      Input ki(nnet_rxfilename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_nnet.Read(ki.Stream(), binary);
    } else {
      ReadKaldiObject(nnet_rxfilename, &nnet);
    }

    Nnet src_nnet; // the one we'll insert.
    ReadKaldiObject(raw_nnet_rxfilename, &src_nnet);

    
    // This function is declared in nnet2-functions.h
    ReplaceLastComponents(src_nnet,
                          remove_layers,
                          (raw ? &nnet : &(am_nnet.GetNnet())));
    KALDI_LOG << "Removed " << remove_layers << " components and added "
              << src_nnet.NumComponents();
    
    if (!raw) {
      Output ko(nnet_wxfilename, binary_write);
      trans_model.Write(ko.Stream(), binary_write);
      am_nnet.Write(ko.Stream(), binary_write);
    } else {
      WriteKaldiObject(nnet, nnet_wxfilename, binary_write);
    }
    KALDI_LOG << "Wrote neural-net acoustic model to " <<  nnet_wxfilename;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
