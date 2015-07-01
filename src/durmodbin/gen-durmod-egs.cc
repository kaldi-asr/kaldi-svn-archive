// bin/gen-phone-dur-egs.cc

// Copyright 2015      Johns Hopkins University (author: Yenda Trmal <jtrmal@gmail.com>)

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
#include "hmm/transition-model.h"
#include "hmm/hmm-utils.h"
#include "util/common-utils.h"
#include "nnet2/nnet-example-functions.h"
#include "tree/build-tree.h"
#include "util/parse-options.h"
#include "durmod/kaldi-durmod.h"

/* static void generate_examples(std::vector<std::pair<int32, int32> > &alignment,
 *                               const std::string &utt_id,
 *                               const int32 left_context,
 *                               const int32 right_context,
 *                               const int32 num_phones,
 *                               const int32 frames_per_eg,
 *                               kaldi::nnet2::NnetExampleWriter *example_writer) {
 *     using namespace kaldi::nnet2;
 *     using namespace kaldi;
 *     NnetExample eg;
 *     // Features will be sequence:of  phones (in 1-of-n encoding) and it's duration
 *     cerr << "Num phones: " << num_phones << std::endl;
 *     cerr << "Left context: " << left_context << std::endl;
 *     cerr << "Right context: " << right_context << std::endl;
 * 
 *     int win_size = left_context + right_context + 1;
 *     cerr << "Win size: " << win_size << std::endl;
 *     int basic_feat_dim = num_phones * win_size + left_context;
 *     cerr << "Feat dim: " << basic_feat_dim << std::endl;
 *     int tot_frames = alignment.size() - left_context - right_context;
 *     cerr << "Num frames: " << tot_frames << std::endl;
 *     int examples_generated = 0;
 * 
 *     if (tot_frames <= 0) {
 *       return;
 *     }
 *     Matrix<BaseFloat> input_frames(tot_frames, frames_per_eg);
 *     eg.left_context = 0;
 *     eg.labels.clear();
 *     eg.labels.resize(tot_frames);
 *     for (int i = 0; i < tot_frames; i++) {
 *       SubVector<BaseFloat> dst(input_frames, i);
 *       int duration_position_offset = 0;
 *       for (int j = 0; j < win_size; j++) {
 *         int32 phone = alignment[i+j].first,
 *               duration= alignment[i+j].second;
 *         // dst(j) = phone;
 *         dst(j * num_phones + phone) = 1;
 * 
 *         if (j == left_context) {
 *           eg.labels[i].resize(1);
 *           eg.labels[i][0] = std::make_pair(duration, 1.0);
 *         } else if ( j < left_context) {
 *           dst(win_size * num_phones + duration_position_offset) = duration;
 *           duration_position_offset++;
 *         }
 *       }
 * 
 *       eg.input_frames = input_frames;
 *       std::ostringstream os;
 *       os << utt_id << "-" << basic_feat_dim;
 * 
 *       std::string key = os.str(); // key is <utt_id>-<frame_id>
 * 
 *       // *num_frames_written += this_num_frames;
 *       // *num_egs_written += 1;
 * 
 *       example_writer->Write(key, eg);
 *     }
 * }
 */

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Generate training examples for the phone duration models\n"
        "Usage:  gen-phone-dur-egs  [options] <model> <alignments-rspecifier> "
        "<egs-wspecifier>\n"
        "e.g.: \n"
        "  gen-phone-dur-egs 1.mdl ark:1.ali ark:1.egs";
    ParseOptions po(usage);
    int left_context = 2;
    int right_context = 2;
    int num_frames = 4;
    BaseFloat frame_shift = 0.01;
    po.Register("frame-shift", &frame_shift,
                "Frame shift for computation of phone duration");
    po.Register("left-context", &left_context, "Number of frames of left "
                "context the neural net requires.");
    po.Register("right-context", &right_context, "Number of frames of right "
                "context the neural net requires.");
    po.Register("num-frames", &num_frames, "Number of frames with labels "
                "that each example contains.");

    po.Read(argc, argv);


    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string roots_filename = po.GetArg(1),
        questions_filename = po.GetArg(2),
        model_filename = po.GetArg(3),
        alignments_rspecifier = po.GetArg(4),
        examples_wspecifier = po.GetArg(5);

    PhoneSets roots;
    {
      Input ki(roots_filename.c_str());
      std::vector<bool> is_shared, is_split;  // dummy variables, won't be used
      ReadRootsFile(ki.Stream(), &roots, &is_shared, &is_split);
    }

    PhoneSets questions;
    ReadIntegerVectorVectorSimple(questions_filename, &questions);

    TransitionModel trans_model;
    ReadKaldiObject(model_filename, &trans_model);

    SequentialInt32VectorReader reader(alignments_rspecifier);
    kaldi::nnet2::NnetExampleWriter example_writer(examples_wspecifier);

    int32 n_done = 0;

    for (; !reader.Done(); reader.Next()) {
      std::string key = reader.Key();
      const std::vector<int32> &alignment = reader.Value();

      std::vector<std::vector<int32> > split;
      SplitToPhones(trans_model, alignment, &split);


      std::vector<std::pair<int32, int32> > pairs;
      for (size_t i = 0; i < split.size(); i++) {
        KALDI_ASSERT(split[i].size() > 0);
        int32 phone = trans_model.TransitionIdToPhone(split[i][0]);
        int32 num_repeats = split[i].size();
        //KALDI_ASSERT(num_repeats!=0);
        pairs.push_back(std::make_pair(phone, num_repeats));
      }
//      generate_examples(pairs, key, left_context, right_context,
//          trans_model.NumPhones(), &example_writer);

    }
    KALDI_LOG << "Done " << n_done << " utterances.";
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


