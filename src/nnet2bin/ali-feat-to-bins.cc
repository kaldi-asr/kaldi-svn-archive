// bin/ali-feat-to-pdf.cc

// Copyright 2014 Pegah Ghahremani

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
#include "nnet2/nnet-example-functions.h"

namespace kaldi {
/*
  This function maps the continuous values of feature for each male
  and female utterance to its discrete bins.
  Male and female utterances have different bins and 1st and 2nd 
  cols of "bins" correspond to bins for male and female utterances 
  respectively.
  To have same col size, bin col correspond to gender with smaller 
  number of bins padded with its largest bin value.
  "gender" is 0 for male and 1 for female. 
*/
static void MapToBins(const MatrixBase<BaseFloat> &feat, 
                      const Matrix<BaseFloat> &bins,
                      int32 gender,
                      std::vector<int32> &alignment) {
  KALDI_ASSERT(bins.NumCols() == 2 && feat.NumCols() == 1);
  KALDI_ASSERT(gender == 0 || gender == 1);
  int32 num_frames = feat.NumRows(), num_bins = bins.NumRows(),
    prev_bin = 0;
  for (int i = 0; i < num_frames; i++) {
    int32 ali = 0;
    prev_bin = 0;
    while (bins(ali, gender) < feat(i,0) && ali < num_bins && 
      bins(ali, gender) != prev_bin) { /* not to exceed bin size */
      prev_bin = bins(ali, gender);
      ali++;
    }
    alignment[i] = ali;
  }
}
} // namespace kaldi

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet2;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;
    const char *usage = 
      "Convert continuous feature values for each frame to discrete bins.\n"
      "It maps feature value for each frame to the bin number it belongs to w.r.t \n"
      "gender and alignment model and get sequence of alignments for each utterance.\n"
      "ali.mdl is matrix containing dicrete bin values and its cols correspond to\n"
      "bins for male and female respectively."
      "\n"
      "Usage: ali-to-bins [options] <model> <features-rspecifier> <utt2gender> <bins-wspecifier>\n"
      "e.g.:\n"
      "ali-to-feats ali.mdl \"$feats\" \\\n"
      " \"ark:select-feats 0 scp:data/train/feats.scp ark:- |\"  ark:utt2gender ark:-";

    ParseOptions po(usage);
    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }
    int32 num_done = 0, num_err = 0;

    std::string model_file_name = po.GetArg(1),
      feature_rspecifier = po.GetArg(2),
      gender_rspecifier = po.GetArg(3),
      alignment_wspecifier = po.GetArg(4);
    SequentialBaseFloatMatrixReader feat_reader(feature_rspecifier);
    RandomAccessInt32Reader gender(gender_rspecifier);
    Int32VectorWriter alignment_writer(alignment_wspecifier); 
    Matrix<BaseFloat> bins;
    ReadKaldiObject(model_file_name, &bins);
    for (; !feat_reader.Done(); feat_reader.Next()) {
      std::string key = feat_reader.Key();
      const Matrix<BaseFloat> &feats = feat_reader.Value();  
      KALDI_ASSERT(feats.NumCols() == 1);
      if (!gender.HasKey(key)) {
        KALDI_ERR << " No gender assigned for utt" << key;
        num_err++;
      } else {
        std::vector<int32> alignment(feats.NumRows(),0);
        MapToBins(feats, bins, gender.Value(key), alignment);
        alignment_writer.Write(key, alignment);
        num_done++;
      }
    }
    KALDI_LOG << "Finished generating alignments, "
              << "successfully processed " << num_done << " alignments,"
              << num_err << " files had errors.";   
    return (num_done == 0 ? 1 : 0); 
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';                     
    return -1;
  }
}
