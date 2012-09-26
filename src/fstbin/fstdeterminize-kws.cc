// fstbin/fstdeterminie-kws.cc

// Copyright 2012  Lucas Ondel

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
#include "fst/fstlib.h"
#include "fstext/fstext-utils.h"
#include "fstext/mapper.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace fst;
    using namespace std;

    const char *usage =
      "Specific determinize tool for keyword spotting. Add dismabiguation to\n"
      "all final arcs before to detemerminize.\n"
      "\n"
      "Usage: fstdeterminize-kws [options] < fst-in > fst-out\n";

    ParseOptions po(usage);
    po.Read(argc, argv);

    if (po.NumArgs() > 0) {
      po.PrintUsage();
      exit(1);
    }

    //Load WFST
    FstReadOptions read_options;
    VectorFst<StdArc> *fst = VectorFst<StdArc>::Read(cin, read_options);

    //Disambig all final arcs
    VectorFst<StdArc> disambig_fst;
    DisambigMapper mapper(fst);
    ArcMap<StdArc, StdArc, DisambigMapper> (*fst, &disambig_fst, &mapper);

    //Determinize WFST
    VectorFst<StdArc> determinized;
    Determinize(disambig_fst, &determinized);

    //Remove disambig symbols
    VectorFst<StdArc> rmdisambig_fst;
    RmDisambigMapper rmapper(&determinized);
    ArcMap<StdArc, StdArc, RmDisambigMapper> (determinized, &rmdisambig_fst, &rmapper);

    //Write on the output
    FstWriteOptions write_options;
    rmdisambig_fst.Write(cout, write_options);

  } catch(std::exception e) {
    e.what();
    return -1;
  }
}
