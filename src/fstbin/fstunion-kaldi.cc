// fstbin/fstpush-union.cc

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

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace fst;
    using namespace std;
    using kaldi::int32;

    const char *usage =
      "Do the union of all WFST passed in the command line."
      "\n"
      "Usage: fstunion-kaldi [options] fst-rspecifier fst-out\n"
      "Usage: fstunion-kaldi [options] 1.fst ... n.fst > out.fst\n";

    ParseOptions po(usage);
    po.Read(argc, argv);

    if (po.NumArgs() < 1) {
      po.PrintUsage();
      exit(1);
    }

    std::string fsts_rspecifier = po.GetArg(1);
    
    if (ClassifyRspecifier(fsts_rspecifier, NULL, NULL) == kNoRspecifier) {
      // Simple file case
      //The final WFST, composed with all input WFSTs
      VectorFst<StdArc> *sumfst = ReadFstKaldi(fsts_rspecifier);

      //Browse all input WFST of the command line
      for(int32 arg_number = 2; arg_number < po.NumArgs(); arg_number++) {
        VectorFst<StdArc> *fst = ReadFstKaldi(po.GetArg(arg_number));

        //Union with the input WFST and the final WFST
        Union(sumfst, 
              *fst);
      }
  
      //Write on the output 
      FstWriteOptions write_options;
      sumfst->Write(cout, write_options);
    } else {
      // Archive case
      SequentialTableReader<VectorFstHolder> fst_reader(fsts_rspecifier);
      
      //The final WFST, composed with all input WFSTs
      VectorFst<StdArc> sumfst;

      for (; !fst_reader.Done(); fst_reader.Next()) {
        VectorFst<StdArc> fst = fst_reader.Value();

        //Union with the input WFST and the final WFST
        Union(&sumfst, 
              fst);
      }
      
      //Write on the output 
      FstWriteOptions write_options;
      sumfst.Write(cout, write_options);
    }
  } catch(const std::exception &e) {
    e.what();
    return -1;
  }
}

