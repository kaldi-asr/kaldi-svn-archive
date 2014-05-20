// lat/kws-functions2.cc

// Copyright 2012  Johns Hopkins University (Author: Guoguo Chen)

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


#include "lat/kws-functions.h"
#include "fstext/determinize-star.h"
#include "fstext/epsilon-property.h"

// We have divided kws-functions.cc into two pieces to overcome "too many
// sections" and "file too large" problems on Windows.  This is the second part.

namespace kaldi {


class KwsProductFstToKwsLexicographicFstMapper {
 public:
  typedef KwsProductArc FromArc;
  typedef KwsProductWeight FromWeight;
  typedef KwsLexicographicArc ToArc;
  typedef KwsLexicographicWeight ToWeight;

  KwsProductFstToKwsLexicographicFstMapper() {}

  ToArc operator()(const FromArc &arc) const {
    return ToArc(arc.ilabel, 
                 arc.olabel, 
                 (arc.weight == FromWeight::Zero() ?
                  ToWeight::Zero() :
                  ToWeight(arc.weight.Value1().Value(), 
                           StdLStdWeight(arc.weight.Value2().Value1().Value(),
                                         arc.weight.Value2().Value2().Value()))),
                 arc.nextstate);
  }

  fst::MapFinalAction FinalAction() const { return fst::MAP_NO_SUPERFINAL; }

  fst::MapSymbolsAction InputSymbolsAction() const { return fst::MAP_COPY_SYMBOLS; }

  fst::MapSymbolsAction OutputSymbolsAction() const { return fst::MAP_COPY_SYMBOLS;}

  uint64 Properties(uint64 props) const { return props; }
};


void MaybeDoSanityCheck(const KwsProductFst &product_transducer) {
  typedef KwsProductFst::Arc::Label Label;
  if (GetVerboseLevel() < 2) return;
  KwsLexicographicFst index_transducer;
  Map(product_transducer, &index_transducer, KwsProductFstToKwsLexicographicFstMapper());
  MaybeDoSanityCheck(index_transducer);
}


// This function replaces a symbol with epsilon wherever it appears
// (fst must be an acceptor).
template<class Arc>
static void ReplaceSymbolWithEpsilon(typename Arc::Label symbol,
                                     fst::VectorFst<Arc> *fst) {
  typedef typename Arc::StateId StateId;
  for (StateId s = 0; s < fst->NumStates(); s++) {
    for (fst::MutableArcIterator<fst::VectorFst<Arc> > aiter(fst, s);
         !aiter.Done(); aiter.Next()) {
      Arc arc = aiter.Value();
      KALDI_ASSERT(arc.ilabel == arc.olabel);
      if (arc.ilabel == symbol) {
        arc.ilabel = 0;
        arc.olabel = 0;
        aiter.SetValue(arc);
      }
    }
  }
}  


void DoFactorMerging(KwsProductFst *factor_transducer,
                     KwsLexicographicFst *index_transducer) {
  using namespace fst;
  typedef KwsProductFst::Arc::Label Label;

  // Encode the transducer first
  EncodeMapper<KwsProductArc> encoder(kEncodeLabels, ENCODE);
  Encode(factor_transducer, &encoder);


  // We want DeterminizeStar to remove epsilon arcs, so turn whatever it encoded
  // epsilons as, into actual epsilons.
  {
    KwsProductArc epsilon_arc(0, 0, KwsProductWeight::One(), 0);
    Label epsilon_label = encoder(epsilon_arc).ilabel;
    ReplaceSymbolWithEpsilon(epsilon_label, factor_transducer);
  }
    

  MaybeDoSanityCheck(*factor_transducer);

  // Use DeterminizeStar
  KALDI_VLOG(2) << "DoFactorMerging: determinization...";
  KwsProductFst dest_transducer;
  DeterminizeStar(*factor_transducer, &dest_transducer);

  MaybeDoSanityCheck(dest_transducer);

  KALDI_VLOG(2) << "DoFactorMerging: minimization...";
  Minimize(&dest_transducer);

  MaybeDoSanityCheck(dest_transducer);
  
  Decode(&dest_transducer, encoder);

  Map(dest_transducer, index_transducer, KwsProductFstToKwsLexicographicFstMapper());
}


void DoFactorDisambiguation(KwsLexicographicFst *index_transducer) {
  using namespace fst;
  typedef KwsLexicographicArc::StateId StateId;

  StateId ns = index_transducer->NumStates();
  for (StateId s = 0; s < ns; s++) {
    for (MutableArcIterator<KwsLexicographicFst> 
         aiter(index_transducer, s); !aiter.Done(); aiter.Next()) {
      KwsLexicographicArc arc = aiter.Value();
      if (index_transducer->Final(arc.nextstate) != KwsLexicographicWeight::Zero())
        arc.ilabel = s;
      else
        arc.olabel = 0;
      aiter.SetValue(arc);
    }
  }
}


} // end namespace kaldi
