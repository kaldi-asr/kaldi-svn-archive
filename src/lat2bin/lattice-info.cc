// lattice-info.cc
// Copyright 2013 Paul R. Dixon

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
#include "fstext/fstext-lib.h"
#include "lat/kaldi-lattice.h"
#include "lat/lattice-functions.h"
#include "exec-stream.h"
#include <fst/script/info-impl.h>

using namespace std;


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace fst;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    using fst::VectorFst;
    using fst::StdArc;
    typedef StdArc::StateId StateId;
    
    const char *usage = "Get OpenFst style information from  a set of lattices\n";
    bool compact = false;
    ParseOptions po(usage);
    po.Register("compact", &compact, "Gather information as a compact lattice");
    po.Read(argc, argv);
    if (po.NumArgs() != 1) {
      po.PrintUsage();
      exit(1);
    }
    std::string lats_rspecifier = po.GetArg(1);
    SequentialCompactLatticeReader clat_reader(lats_rspecifier);
    for (; !clat_reader.Done(); clat_reader.Next()) {
      CompactLattice clat = clat_reader.Value();
      std::string key = clat_reader.Key();
      cout << key << endl;
      if (compact) {
        FstInfo<CompactLatticeArc> info(clat, true);
        PrintFstInfo(info);
      } else {
        Lattice lat;
        ConvertLattice(clat, &lat);
        FstInfo<LatticeArc> info(lat, true);
        PrintFstInfo(info);
      }
      cout <<  endl;
    }
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
