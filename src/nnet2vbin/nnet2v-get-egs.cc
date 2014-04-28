// nnet2vbin/nnet2v-get-egs.cc

// Copyright 2012-2014  Johns Hopkins University (author: Daniel Povey)

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

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet2;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "This program gets examples of data for neural-network training;\n"
        "this version is specialized for vision tasks.  Note: the time axis\n"
        "(frame index) is identified with the row-index of the input matrix.\n"
        "The --left-context and --right-context variables may be specified,\n"
        "and if so left-context+right-context+1 should equal the number of rows in\n"
        "the input feature matrices;otherwise it works them out from the first example\n"
        "The concept of left and right context is borrowed from speech recognition and\n"
        "isn't very natural for vision, but the framework requires it.\n"
        "Note: all input feature matrices must be the same size; labels-rspecifier\n"
        "should map from example-id (a string) non-negative integers.\n"
        "\n"
        "Usage:  nnet2v-get-egs [options] <features-rspecifier> <labels-rspecifier>"
        " <training-examples-out>\n"
        "e.g.:\n"
        "  nnet2v-get-egs scp:data/train/feats.scp ark:data/train/labels ark:all.egs\n";


    int32 left_context = -1, right_context = -1;
    int32 srand_seed = 0;
    
    ParseOptions po(usage);
    po.Register("left-context", &left_context, "Number of frames of left context "
                "the neural net requires.");
    po.Register("right-context", &right_context, "Number of frames of right context "
                "the neural net requires.");
    po.Register("srand", &srand_seed, "Seed for random number generator "
                "(only relevant if --keep-proportion != 1.0)");
    
    po.Read(argc, argv);

    srand(srand_seed);
    
    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string feature_rspecifier = po.GetArg(1),
        label_rspecifier = po.GetArg(2),
        examples_wspecifier = po.GetArg(3);

    // Read in all the training files.
    SequentialBaseFloatMatrixReader feat_reader(feature_rspecifier);
    RandomAccessInt32Reader label_reader(label_rspecifier);
    NnetExampleWriter example_writer(examples_wspecifier);
    
    int32 num_done = 0, num_err = 0, feature_dim = 0;
    
    for (; !feat_reader.Done(); feat_reader.Next()) {
      std::string key = feat_reader.Key();
      const Matrix<BaseFloat> &feats = feat_reader.Value();
      if (!label_reader.HasKey(key)) {
        KALDI_WARN << "No label for key " << key;
        num_err++;
      } else {
        if (feature_dim == 0) {
          feature_dim = feats.NumCols();
        } else {
          KALDI_ASSERT(feature_dim == feats.NumCols() &&
                       "Inconsistent feature dimensions.");
        }
        if (left_context == -1) {
          left_context = (feats.NumRows() - 1) / 2;
          right_context = feats.NumRows() - 1 - left_context;
        } else {
          KALDI_ASSERT(feats.NumRows() == left_context + right_context + 1 &&
                       "Inconsistent feature dimensions.");
        }

        int32 label = label_reader.Value(key);
        BaseFloat weight = 1.0;
        KALDI_ASSERT(label >= 0);
            
        NnetExample eg;
        eg.labels.push_back(std::make_pair(label, weight));
        eg.input_frames = feats;
        eg.left_context = left_context;  // there is no right_context, it's
                                         // implicit as input_frames.NumRows() -
                                         // 1 - left_context.
        // leave eg.spk_info as empty.
        
        example_writer.Write(key, eg);
        num_done++;
      }
    }

    KALDI_LOG << "Successfully processed " << num_done
              << " examples, " << num_err << " with errors.";
    return (num_done == 0 ? 1 : 0);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
