/// far-to-lattice.cc
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// author Paul Dixon (paul.r.dixon@mgmail.com)

#include <fst/script/arg-packs.h>
#include <fst/script/script-impl.h>
#include <fst/extensions/far/far.h>
#include <fst/extensions/far/main.h>
#include <fst/extensions/far/farscript.h>

#include "base/kaldi-common.h"
#include "lat/kaldi-lattice.h"


using namespace std;
using kaldi::LatticeWriter;
using kaldi::LatticeArc;
using kaldi::Lattice;
using kaldi::LatticeWeight;

namespace fst {

template<>
struct WeightConvert<TropicalWeight, LatticeWeight> {
  LatticeWeight operator()(TropicalWeight w) const {
    return w == TropicalWeight::Zero() ? LatticeWeight::Zero() :
      LatticeWeight(w.Value(), 0);
  }
};

namespace script {
typedef args::Package<const vector<string>&, const string&, const string&>
  ToKaldiLatticeArgs;

template<class Arc>
void ToKaldiLattice(ToKaldiLatticeArgs *args) {
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Label Label;
  typedef typename Arc::Weight Weight;
  FarReader<Arc> *reader = FarReader<Arc>::Open(args->arg1);
  if (!reader) return;
  WeightConvertMapper<Arc, LatticeArc> mapper;
  LatticeWriter writer(args->arg3);
  for (; !reader->Done(); reader->Next()) {
    string key = reader->GetKey();
    const Fst<Arc> &fst = reader->GetFst();
    Lattice lattice;
    ArcMap(fst, &lattice, mapper);
    writer.Write(key, lattice);
  }
  delete reader;
}

void ToKaldiLattice(const vector<string> &ifilenames, const string &arc_type, 
    const string &ofilename) { 
  ToKaldiLatticeArgs args(ifilenames, arc_type, ofilename);
  Apply<Operation<ToKaldiLatticeArgs> >("ToKaldiLattice", arc_type, &args);
}

REGISTER_FST_OPERATION(ToKaldiLattice, StdArc, ToKaldiLatticeArgs);
REGISTER_FST_OPERATION(ToKaldiLattice, LogArc, ToKaldiLatticeArgs);
REGISTER_FST_OPERATION(ToKaldiLattice, LatticeArc, ToKaldiLatticeArgs);

}  // namespace script
REGISTER_FST(VectorFst, LatticeArc);
REGISTER_FST(ConstFst, LatticeArc);
}  // namespace fst


using namespace fst;
int main(int argc, char **argv) {
  try {
    namespace s = fst::script;
    using namespace kaldi;
    const char* usage = "Convert OpenFst far files to Kaldi Lattice table.\n"
      "Usage: far-to-lattice far.1 far.2 ... <lattice-wspecifer>\n";
    ParseOptions po(usage);
    po.Read(argc, argv);
    if (po.NumArgs() < 2) {
      po.PrintUsage();
      exit(1);
    }
    vector<string> ifilenames;  
    for (int i = 1; i < argc - 1; ++i)
      ifilenames.push_back(strcmp(argv[i], "") != 0 ? argv[i] : "");
    string ofilename = argv[argc - 1];
    string arc_type = fst::LoadArcTypeFromFar(ifilenames[0]);
    s::ToKaldiLattice(ifilenames, arc_type, ofilename);
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
  return 0;
}
