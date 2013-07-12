// idlaktxpbin/convert-to-binary.cc

// Copyright 2013 CereProc Ltd.  (Author: Richard Williams)

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
//

#include <string>
#include <vector>

#include "base/kaldi-common.h"
#include "util/common-utils.h"

int main(int argc, const char* argv[]) {
  using kaldi::Matrix;
  bool binary;
  const char *usage = "Usage: ./convert-to-binary inputFile outputFile";

  try {
    kaldi::ParseOptions po(usage);
    po.Read(argc, argv);
    // Must have inpout and output filenames for XML
    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    // Parse args.
    std::string inputFile = argv[1];
    std::string outputFile = argv[2];
    kaldi::Input ki(inputFile, &binary);
    kaldi::Output kio(outputFile, binary);
    // Put it into kaldi objects.
    Matrix<float> matrix;
    matrix.Read(ki.Stream(), false);

    // Write to std::cout so it can be piped somewhere else.
    matrix.Write(kio.Stream(), false);
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
