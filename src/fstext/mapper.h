// fstext/disambig.cc

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

#ifndef DISAMBIG_H_
#define DISAMBIG_H

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fst/fstlib.h"

namespace fst {

  class OutputLabelToEpsilonMapper {
    public:
      // Maps an arc type StdArc to arc type B.
      LogArc operator()(const LogArc &arc);
      // Specifies final action the mapper requires (see above).
      // The mapper will be passed final weights as arcs of the
      // form StdArc(0, 0, weight, kNoStateId).
      MapFinalAction FinalAction() const;
      // Specifies input symbol table action the mapper requires (see above).
      MapSymbolsAction InputSymbolsAction() const;
      // Specifies output symbol table action the mapper requires (see above).
      MapSymbolsAction OutputSymbolsAction() const;
      // This specifies the known properties of an Fst mapped by this
      // mapper. It takes as argument the input Fst's known properties.
      uint64 Properties(uint64 props) const;
  };

  class DisambigMapper {
    public:
      VectorFst<StdArc> *fst;

      DisambigMapper(VectorFst<StdArc> *f);

      // Maps an arc type StdArc to arc type B.
      StdArc operator()(const StdArc &arc);
      // Specifies final action the mapper requires (see above).
      // The mapper will be passed final weights as arcs of the
      // form StdArc(0, 0, weight, kNoStateId).
      MapFinalAction FinalAction() const;
      // Specifies input symbol table action the mapper requires (see above).
      MapSymbolsAction InputSymbolsAction() const;
      // Specifies output symbol table action the mapper requires (see above).
      MapSymbolsAction OutputSymbolsAction() const;
      // This specifies the known properties of an Fst mapped by this
      // mapper. It takes as argument the input Fst's known properties.
      uint64 Properties(uint64 props) const;
  };


  class RmDisambigMapper {
    public:
      VectorFst<StdArc> *fst;

      RmDisambigMapper(VectorFst<StdArc> *f);

      // Maps an arc type StdArc to arc type B.
      StdArc operator()(const StdArc &arc);
      // Specifies final action the mapper requires (see above).
      // The mapper will be passed final weights as arcs of the
      // form StdArc(0, 0, weight, kNoStateId).
      MapFinalAction FinalAction() const;
      // Specifies input symbol table action the mapper requires (see above).
      MapSymbolsAction InputSymbolsAction() const;
      // Specifies output symbol table action the mapper requires (see above).
      MapSymbolsAction OutputSymbolsAction() const;
      // This specifies the known properties of an Fst mapped by this
      // mapper. It takes as argument the input Fst's known properties.
      uint64 Properties(uint64 props) const;
  };

}

#include "mapper-inl.h"
#endif 
