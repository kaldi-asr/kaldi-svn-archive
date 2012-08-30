// kws/kws-converters.cc

// Copyright 2012   Lucas Ondel

// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License.  You may obtain a copy
// of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.  See the Apache 2 License for the
// specific language governing permissions and limitations under the License.

#include "kws-converters.h"
#include "fst/fstlib.h"
#include "lat/kaldi-lattice.h"
#include <cmath>

namespace kaldi {
  //LatticeLogConverter

  LatticeLogConverter::LatticeLogConverter(BaseFloat ascale) {
    this->acoustic_scale = ascale;
  }

  void *LatticeLogConverter::ConvertWeight(void *oldweight) {
    LatticeWeight *old = (LatticeWeight*)oldweight;
    return new LogWeight(old->Value1() + old->Value2()/this->acoustic_scale);
  }

  void LatticeLogConverter::ModifyArc(void *fst, void *p_arc) {
    LogArc *arc = (LogArc *)p_arc; 
    arc->ilabel = arc->olabel;
    arc->olabel = 0;
  }


  //LogRealConverter

  void *LogRealConverter::ConvertWeight(void *oldweight) {
    LogWeight *old = (LogWeight*)oldweight;
    return new LogWeight(exp(-old->Value()));
  }

  //DisambigFinalArcConverter

  void DisambigFinalArcConverter::ModifyArc(void *p_fst, void *p_arc) {
    VectorFst<LogArc> *fst = (VectorFst<LogArc>*)p_fst;
    LogArc *arc = (LogArc*)p_arc;

    //Check if the arc links the final state
    if (fst->Final(arc->nextstate) != LogWeight::Zero()) {
      if (arc->olabel != 0) {
        if (arc->ilabel == 0) {
          arc->ilabel = arc->olabel;
        } else  {
          KALDI_ERR << "input label on final link is not epsilon.";
        }
      } else {
        KALDI_WARN << "Not utterance identifier on final link. arc : " <<
        arc->ilabel << " " << arc->olabel;
      }
    }
  }

  //RmDisambigFinalArcConverter
  void RmDisambigFinalArcConverter::ModifyArc(void *p_fst, void *p_arc) {
    VectorFst<LogArc> *fst = (VectorFst<LogArc>*)p_fst;
    LogArc *arc = (LogArc*)p_arc;

    //Check if the arc links the final state
    if (fst->Final(arc->nextstate) != LogWeight::Zero()) {
      arc->ilabel = 0;
    }
  }

}

