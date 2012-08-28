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
#include "nnet-dp/nnet1-utils.h"
#include "nnet-dp/train-nnet1.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Train neural network (of type nnet1).\n"
        "Adaptive training that adjusts learning rates periodically using\n"
        "validation set\n"
        "Usage:  nnet1-train [options] <nnet1-in> <features-rspecifier> "
        "<alignments-rspecifier> <validation-utt-list> <nnet1-out>\n"
        "e.g.:\n"
        " nnet1-train exp/nnet1/1.nnet1 'scp:data/train/feats.scp' 'ark:gunzip -c exp/tri1/{?,??}.ali.gz|'  "
        "exp/nnet1/valid.uttlist exp/nnet1/2.nnet1\n";
        
    bool binary_write = true;
    bool zero_occupancy = true;
    Nnet1AdaptiveTrainerConfig adaptive_trainer_config;
    Nnet1BasicTrainerConfig basic_trainer_config;
    
    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("zero-occs", &zero_occupancy, "Set occupation counts stored in "
                "neural net to zero before training");
                
    basic_trainer_config.Register(&po);
    adaptive_trainer_config.Register(&po);
    
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
    AmNnet1 am_nnet;

    {
      bool binary_read;
      Input ki(nnet1_rxfilename, &binary_read);
      trans_model.Read(ki.Stream(), binary_read);
      am_nnet.Read(ki.Stream(), binary_read);
    }


    std::vector<CompressedMatrix> train_feats, validation_feats;
    std::vector<std::vector<int32> > train_ali, validation_ali;
    ReadAlignmentsAndFeatures(feature_rspecifier, alignments_rspecifier,
                              validation_utt_list,
                              &train_feats, &validation_feats,
                              &train_ali, &validation_ali);
    ConvertAlignmentsToPdfs(trans_model, &train_ali);
    ConvertAlignmentsToPdfs(trans_model, &validation_ali);

    if (zero_occupancy)
      am_nnet.Nnet().ZeroOccupancy(); // We zero the stored occupancy counts before
    // each phase of trainingn
        
    Nnet1BasicTrainer basic_trainer(basic_trainer_config, train_feats, train_ali,
                                    &am_nnet);

    Nnet1 nnet_gradient(am_nnet.Nnet()); // Construct a copy of the neural net
    nnet_gradient.SetZeroAndTreatAsGradient();  // which we'll use to store the gradient on the
    // validation set.
    Nnet1ValidationSet validation_set(validation_feats, validation_ali,
                                      am_nnet, &nnet_gradient);
    // This trainer is responsible for training on the training set
    // (using the basic_trainer) and using the validation set gradients
    // to adjust the learning rate on every "phase" of training.
    Nnet1AdaptiveTrainer trainer(adaptive_trainer_config,
                                 &basic_trainer, &validation_set);
    trainer.Train();

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


