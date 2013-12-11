// bin/compile-ctx-questions.cc

// Copyright 2009-2011  Microsoft Corporation
//           2013-      Arnab Ghoshal
//           2013-      CereProc Ltd.

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
#include "tree/build-tree-questions.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Compile keyed questions only from the text format into the Kaldi\n"
        "Questions class."
        "questions are of the format F ? N N ... where f is a field index and\n"
        "N are possible integer values\n"
        "Usage:  compile-questions [options] <keyed-questions-text-file>\n"
        "<questions-out>\n"
        "e.g.: \n"
        " compile-ctx-questions questions.txt questions.qst\n";

    bool binary = true;

    ParseOptions po(usage);
    po.Register("binary", &binary, "Write output in binary mode");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string keyed_questions_filename = po.GetArg(1),
        questions_out_filename = po.GetArg(2);

    Questions qo;

    // Now read the keyed questions
    if (!keyed_questions_filename.empty()) {
      kaldi::Input ki;
      if (!ki.OpenTextMode(keyed_questions_filename)) {
        KALDI_WARN << "Failed to open keyed questions file: "
                   << keyed_questions_filename;
      } else {
        KALDI_LOG << "Reading keyed questions from: " <<
            keyed_questions_filename;
        ReadKeyedQuestions(ki.Stream(), &qo);
      }
    }

    WriteKaldiObject(qo, questions_out_filename, binary);
    KALDI_LOG << "Wrote questions to " << questions_out_filename;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
