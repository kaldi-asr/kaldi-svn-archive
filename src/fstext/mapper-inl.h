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

#ifndef DISAMBIG_INL_H_
#define DISAMBIG_INL_H_

namespace fst {
  //OutputLabelToEpsilonMapper
  LogArc OutputLabelToEpsilonMapper::operator()(const LogArc &arc) {
    LogArc retval(arc.ilabel, 0, arc.weight, arc.nextstate);
    return retval;
  }

  MapFinalAction OutputLabelToEpsilonMapper::FinalAction() const {
    return MAP_NO_SUPERFINAL;
  }

  MapSymbolsAction OutputLabelToEpsilonMapper::InputSymbolsAction() const {
    return MAP_NOOP_SYMBOLS;
  }

  MapSymbolsAction OutputLabelToEpsilonMapper::OutputSymbolsAction() const {
    return MAP_NOOP_SYMBOLS;
  }

  uint64 OutputLabelToEpsilonMapper::Properties(uint64 props) const {
    return props;
  }
  //DisambigMapper
  DisambigMapper::DisambigMapper(VectorFst<StdArc> *f) {
    fst = f;
  }

  StdArc DisambigMapper::operator()(const StdArc &arc) {
    //Check if the next state is the final state
    if (arc.nextstate >= 0 && 
        fst->Final(arc.nextstate) != TropicalWeight::Zero()) {
      if (arc.olabel != 0) {
        if (arc.ilabel == 0) {
          StdArc retval(arc.olabel, arc.olabel, arc.weight, arc.nextstate);
          return retval;
        } else {
          KALDI_ERR << "input label on final link is not epsilon.";
        }
      } else {
        KALDI_WARN << "Not utterance identifier on final link. arc : " <<
          arc.ilabel << " " << arc.olabel;
      }
    }
    StdArc a(arc.ilabel, arc.olabel, arc.weight, arc.nextstate);
    return a;
  }

  MapFinalAction DisambigMapper::FinalAction() const {
    return MAP_NO_SUPERFINAL;
  }

  MapSymbolsAction DisambigMapper::InputSymbolsAction() const {
    return MAP_NOOP_SYMBOLS;
  }

  MapSymbolsAction DisambigMapper::OutputSymbolsAction() const {
    return MAP_NOOP_SYMBOLS;
  }

  uint64 DisambigMapper::Properties(uint64 props) const {
    return props;
  }

  //RmDisambigMapper
  RmDisambigMapper::RmDisambigMapper(VectorFst<StdArc> *f) {
    fst = f;
  }

  StdArc RmDisambigMapper::operator()(const StdArc &arc) {
    //Check if the next state is the final state
    if (arc.nextstate >= 0 && fst->Final(arc.nextstate) != TropicalWeight::Zero()) {
      StdArc retval(0, arc.olabel, arc.weight, arc.nextstate);
      return retval;
    }
    return arc;
  }

  MapFinalAction RmDisambigMapper::FinalAction() const {
    return MAP_NO_SUPERFINAL;
  }

  MapSymbolsAction RmDisambigMapper::InputSymbolsAction() const {
    return MAP_NOOP_SYMBOLS;
  }

  MapSymbolsAction RmDisambigMapper::OutputSymbolsAction() const {
    return MAP_NOOP_SYMBOLS;
  }

  uint64 RmDisambigMapper::Properties(uint64 props) const {
    return props;
  }
}
#endif

