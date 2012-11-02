// fstbin/fstpush-kaldi.cc

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

    bool push_in_log = false;
    bool remove_total_weight = true;
    
    const char *usage =
      "Push weights on a table of lattices."
      "\n"
      "Usage: fstpush-kaldi [options] fst-rspecifier fst-wspecifier\n";

    ParseOptions po(usage);
    po.Register("push-in-log", 
                &push_in_log,
                "Wether or not perform the pushing over the log semiring");
    po.Register("remove-total-weight",
                &remove_total_weight,
                "Remove total weight when pushing weights");
  
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string fsts_rspecifier = po.GetArg(1),
                fsts_wspecifier = po.GetArg(2);
    
    if (ClassifyRspecifier(fsts_rspecifier, NULL, NULL) == kNoRspecifier) {
      // Simple file case
      VectorFst<StdArc> *fst = ReadFstKaldi(fsts_rspecifier);

        if (!push_in_log) {
          //Push over the tropical semiring
          Push<StdArc>(fst, 
                       REWEIGHT_TO_INITIAL, 
                       kDelta, 
                       remove_total_weight);
        } else {
          //Push over the log semiring
          PushInLog<REWEIGHT_TO_INITIAL>(fst, 
                                         kDelta,
                                         remove_total_weight);
        }

        WriteFstKaldi(*fst, fsts_wspecifier);

    } else {
      // Archive case
      SequentialTableReader<VectorFstHolder> fst_reader(fsts_rspecifier);
      TableWriter<VectorFstHolder> fst_writer(fsts_wspecifier);
      
      for (; !fst_reader.Done(); fst_reader.Next()) {
        string key = fst_reader.Key();
        VectorFst<StdArc> fst = fst_reader.Value();

        if (!push_in_log) {
          //Push over the tropical semiring
          Push<StdArc>(&fst, 
                       REWEIGHT_TO_INITIAL, 
                       kDelta, 
                       remove_total_weight);
        } else {
          //Push over the log semiring
          PushInLog<REWEIGHT_TO_INITIAL>(&fst,
                                         kDelta,
                                         remove_total_weight);
        }

        fst_writer.Write(key, fst);
      }
    }
  } catch(const std::exception &e) {
    e.what();
    return -1;
  }
}

