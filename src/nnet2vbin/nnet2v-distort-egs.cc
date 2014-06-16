// nnet2vbin/nnet2v-distort-egs.cc

// Copyright 2014  Pegah Ghahremani

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
#include "feat/distortion-functions.h"
#include "hmm/transition-model.h"
#include "nnet2/nnet-randomize.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet2;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "This program generates synthised examples using transformation and it is specialized for vision task;\n"
        "It creates perturbed version of  image using combination of affine displacement, elastic distortion methods\n"
        " and adding background noise to it.\n" 
        "Affine displacement methods generates perturbed image using rotation, scaling and shifting of original image. \n"
        "In elastic distortion method, new image is generated using random displacement field.\n" 
        " Also the background noise can be added to image background;\n"
        "Usage: nnet2v-distort-egs [options] <egs-rspecifier> <egs-wspecifier> \n"
        "e.g.:\n"
        "nnet2v-distort-egs ark:train.egs ark:distorted.egs\n";
    
    ParseOptions po(usage);
    DeformationOptions deform_opts;   
    deform_opts.Register(&po); 
    po.Read(argc, argv);    
    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }
    std::string examples_rspecifier = po.GetArg(1);
    SequentialNnetExampleReader example_reader(examples_rspecifier); 
    std::string examples_wspecifier = po.GetArg(2);
    //NnetExampleWriter* example_writer = new NnetExampleWriter(examples_wspecifier);
    NnetExampleWriter example_writer(examples_wspecifier);
    int32 num_done = 0, num_err = 0; 
    Matrix<BaseFloat> disp_field;
    for (; !example_reader.Done(); example_reader.Next()) {
      const NnetExample &eg = example_reader.Value();
      Matrix<BaseFloat> input_frames(eg.input_frames);
      // Initilize displacement field to apply to images and gernerate new example.
      GetDistortionField(deform_opts, input_frames.NumCols(), input_frames.NumRows(),
                         &disp_field);
      Matrix<BaseFloat> distorted_frames;  
      ApplyDistortionField(deform_opts, disp_field, input_frames, &distorted_frames);  
      NnetExample distorted_eg; 
      distorted_eg.labels = eg.labels;
      distorted_eg.input_frames = distorted_frames;
      distorted_eg.left_context = eg.left_context;
      example_writer.Write(example_reader.Key(), distorted_eg);
      num_done++;
    }
    KALDI_LOG << "Successfully processed " << num_done
              << " examples, " << num_err << " with errors.";
    return (num_done == 0 ? 1 : 0);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
