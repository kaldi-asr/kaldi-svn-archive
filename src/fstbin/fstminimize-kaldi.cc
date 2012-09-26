// fstbin/fstminimze-kaldi.cc

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

    const char *usage =
      "Minimize a kaldi table of WFSTs."
      "\n"
      "Usage: fstminimize-kaldi [options] fst-rspecifier fst-wspecifier\n";
    
    bool encode = false;
    bool use_log = false;

    ParseOptions po(usage);

    po.Register("use-log",
                &use_log,
                "Perform the minimization in the log semiring.");

    po.Register("encode",
                &encode,
                "Encode the the WFST in an acceptor before the minimization.");

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
      if (encode) {
        MinimizeEncoded(fst, kDelta);
      } else {
        Minimize<StdArc>(fst);//generates warning
      }
      WriteFstKaldi(*fst, fsts_wspecifier);
    } else {
      // Archive case
      SequentialTableReader<VectorFstHolder> fst_reader(fsts_rspecifier);
      TableWriter<VectorFstHolder> fst_writer(fsts_wspecifier);
      
      for (; !fst_reader.Done(); fst_reader.Next()) {
        string key = fst_reader.Key();
        VectorFst<StdArc> fst = fst_reader.Value();
        VectorFst<StdArc> out;

        if (encode) {
          if (use_log) {
            VectorFst<LogArc> *fst_log = new VectorFst<LogArc>; 
            Cast(fst, fst_log);
            MinimizeEncoded(fst_log, kDelta);
            Cast(*fst_log, &out);
            delete fst_log;
          } else {
            MinimizeEncoded(&fst, kDelta);
            out = fst;
          }
        } else {
          if (use_log) {
            VectorFst<LogArc> *fst_log = new VectorFst<LogArc>; 
            Cast(fst, fst_log);
            Minimize<LogArc>(fst_log);
            Cast(*fst_log, &out);
            delete fst_log;
          } else {
            Minimize<StdArc>(&fst);
            out = fst;
          }
        }
        fst_writer.Write(key, out);
      }
    }
  } catch(const std::exception &e) {
    e.what();
    return -1;
  }
}

