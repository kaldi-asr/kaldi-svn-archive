// kws/kws-converters.h

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

#ifndef KWS_CONVERTERS_H_
#define KWS_CONVERTERS_H_

#include "kws-fstext.h"
#include "base/kaldi-common.h"

namespace kaldi {
 
  /**
    Implementation of a converter from a Kaldi Lattice to the Log semiring. It
    also replaces all output labels by epsilon.
   */
  class LatticeLogConverter : public Converter {

    private:
      BaseFloat acoustic_scale;

    public:

      /**
        Create the converter with the specifed acoustic scale
        @param ascale Acoustic scale use during the weight conversion
       */
      LatticeLogConverter(BaseFloat ascale = 1.0);

      /**
        @copydoc Converter::ConvertWeight(void *)
        As values in the semiring used in Kaldi lattices are equivalent in the
        Log semiring, this method only changes the type of the weight but not
        the value itself.
       */
      void *ConvertWeight(void *oldweight);

      /**
        @copydoc Converter::ModifyArc(void *, void *)
        Replace output label by epsilon.
       */
      void ModifyArc(void *fst, void *p_arc);
  };

  /**
    Implementation of a converter from a the Log semiring to the Real semiring.
    As there is no implementation of the Real semiring in OpenFst (RealWeight
    doesn't exist), this converter converts only weights' value and store it in
    a LogWeight object. Thus, resulting converter should not be used to perform
    operation on the Real semiring. This class is only used for debugging
    purposes.
   */
  class LogRealConverter : public Converter {

    /**
      @copydoc Converter::ConvertWeight(void *weight) 
      Convert the weight by: e^-x, where x is the origingal weight.
      The resulting weight type is still a LogWeight object. 
     */
    void *ConvertWeight(void *oldweight);
  };

  /**
    Perform a disambiguation on each arc linking the final state. This Converter
    expect a FST in the log semiring (LogWeight). This function does not change
    the arcs'weight.
   */
  class DisambigFinalArcConverter : public Converter {

    /**
      @copydoc Converter::ModifyArc(void *,void *)
      If the arc links the final state, copy the output label on the input
      label.
     */
    void ModifyArc(void *p_fst, void *p_arc);
  };


  /**
    Remove the disambiguation done by @ref DisambigFinalArcConverter. This
    Converter expect a FST in the log semiring (LogWeight). This function does
    not change the arcs'weight.
   */
  class RmDisambigFinalArcConverter : public Converter {

    /**
      @copydoc Converter::ModifyArc(void *,void *)
      If the arc links the final state, set the input label to epsilon.
     */
    void ModifyArc(void *p_fst, void *p_arc);
  };
}

#endif 

