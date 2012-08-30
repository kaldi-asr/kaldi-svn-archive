// kws/kws-fstext.h

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


#ifndef KWS_FSTEXT_H_
#define KWS_FSTEXT_H_ 

#include "base/kaldi-common.h"
#include "fst/fstlib.h"
#include <queue>

namespace kaldi {

  using namespace fst;

  /**
    Implement the conversion of a FST. This class could be use to change the
    semiring of the fst as well as to relabel arcs' label. It is a generic class
    and should not be used as is. By default, this class does nothing.
   */
  class Converter {

    public:
      /**
        Convert the old weight into a specific weight depending of the
        conversion semiring. We use an void pointer to avoid mixing polymorphism
        and template (forbidden in C++). If NULL returned (default
        implementation), the old weight is kept. The new weight returned should
        be allocated on the stack and it will be automatically deleted.  
        @param oldweight Pointer to the old weight to convert.  
        @return Pointer to the resulting weight after the conversion. 
       */
      virtual void *ConvertWeight(void *oldweight) {
        return NULL;
      }

      /**
        This method allow you to modify arcs during the conversion. By default
        the method has no effect.  
        @param fst Pointer to the old fst.  
        @param arc Pointer to the arc to modify.
       */
      virtual void ModifyArc(void *fst, 
                             void *arc) {}
  };

  /**
    Convert and FST into another one, depending on the implementation of the
    converter.
    
    @param oldfst The FST to convert.  
    @param newfst Pointer to the result of the conversion.  
    @param converter Implementation of the class @ref Converter.
   */
  template<typename OldWeightT, 
           typename NewWeightT>
    void ConvertFST(VectorFst<ArcTpl<OldWeightT> > oldfst, 
                    VectorFst<ArcTpl<NewWeightT> > *newfst, 
                    Converter *converter) {
      
      //Rename template for an easier reading
      typedef ArcTpl<OldWeightT> OldArc;
      typedef ArcTpl<NewWeightT> NewArc;
      typedef VectorFst<OldArc> OldFst;

      //Creates states 
      for (int32 i=0; i < oldfst.NumStates(); i++) {
        newfst->AddState();
      }

      //Set initial state
      newfst->SetStart(oldfst.Start());

      //Browse states'arcs and copy them
      for (StateIterator<OldFst> siter(oldfst); !siter.Done(); siter.Next()) {
        typename ArcTpl<OldWeightT>::StateId state = siter.Value();
        OldWeightT w_final = oldfst.Final(state);

        // Final state case
        if (w_final != OldWeightT::Zero()) {
          NewWeightT *p_weight = NULL;
          
          NewWeightT *new_weight = 
          (p_weight = (NewWeightT*)converter->ConvertWeight(&w_final)) ?
          p_weight : (NewWeightT*)&w_final;
         
          newfst->SetFinal(state, *new_weight);
          delete p_weight;
        }

        //Arc copy
        for (MutableArcIterator<OldFst> aiter(&oldfst, state); !aiter.Done();
        aiter.Next()) { 
          OldArc arc = aiter.Value(); NewWeightT *p_weight = NULL;
          converter->ModifyArc(&oldfst, &arc); 
          
          NewWeightT *new_weight = 
          (p_weight = (NewWeightT *) converter->ConvertWeight(&arc.weight)) ? 
          p_weight : (NewWeightT*)&arc.weight;
          
          NewArc new_arc(arc.ilabel, arc.olabel, *new_weight, arc.nextstate);
          newfst->AddArc(state, new_arc); delete p_weight;
        }
      }
    }  

  /**
    A filter structure is composed of an application step (when to apply the 
    filter) and the filtering FST.
   */
  struct Filter { 
    /**
      When the filter will be applied.
    */    
    int32 flag;

    /**
      Filtering fst.
    */
    VectorFst<LogArc> *fst;
  };
  
  /**
    Create filters given the formatted string. The format of the string is :
    application1,filename1:application2,filename2:.... All created filters are
    added to the vector.
    @param strfilters Formatted string specifying how to create filters
    @param accepted_string List of the differeant application set accepted
    @param filters Vector where are added all newly created filters
  */
  template<typename ArcT>
  void CreateFilters(string strfilters,
                     vector<string> accepted_strings,
                     vector<Filter>& filters) {
     
    //Filters string format
    //step-application,filename:step-application,filemame:....

    //Parsing filters string
    istringstream iss_filters(strfilters);
    string filter;//Pair of position + filenane 
    while (getline(iss_filters, filter, ':')) {
      //Parsing filter string
      istringstream iss_filter(filter);
      string position;
      string filename;

      getline(iss_filter, position, ',');
      getline(iss_filter, filename, ','); 

      //Check if the filter is correct
      if (position.empty() || filename.empty()) {
        KALDI_LOG << "Hello ";
        KALDI_ERR << "Malformatted filter string\n"
          "format is : \"application1,filename1:application2,filename2\"\n";
      }

      Filter filter;
      filter.flag = -1;

      //Reading the application step
      int32 count = 0;
      for (vector<string>::iterator it = accepted_strings.begin();
          it != accepted_strings.end(); it++) {
        //Check if the string matches
        if (position.compare(*it) != 0) continue;

        //Set the filter flag
        filter.flag=count;

        count++;        
      }

      //Safety check, if the filter has a correct flag
      if (filter.flag == -1) {
        KALDI_ERR << "Unknown filter application step.\n"
          "possible values are: \"input\", \"factor_selection\" and index\"\n";
      }

      //Reading fst
      filter.fst = VectorFst<ArcT>::Read(filename);

      //Add the filter to the list.
      filters.push_back(filter);
    }
  }

  /**
    Apply filters corresponding to the application step provided.
    @param filters The list of filters to apply (only filters corresponding to 
    the application step will be applied)
    @param step The specific application step
    @param input The input fst
    @param Pointer to the returned fst. The returned fst is internally 
    allocated on the heap.
  */
  template<typename  ArcT>
  void ApplyFilters(vector<Filter> filters, 
                    int32 flag,
                    VectorFst<ArcT> input, 
                    VectorFst<ArcT> *& outfst) {
     
    VectorFst<ArcT> *infst = new VectorFst<ArcT>(input);
    outfst = infst; 
      
    //Browse the filter list
    for (vector<Filter>::iterator it = filters.begin();
        it != filters.end(); it++) {
      //Check the filter's application step
      if ((*it).flag != flag) continue;
      
      //Allocate a new temporay output fst
      outfst = new VectorFst<ArcT>();

      //Compose the input and the filter
      Compose<ArcT>(*infst, *(*it).fst, outfst);

      //Update pointers for the next iteration 
      delete infst;
      infst = outfst;
    }
  }

  /**
    Cluster the arcs with the same input labels and overlapping time-spans.
    @param syms The symbol table of the transducer.
    @param fst The transducer to process
  */
  template<typename ArcT>
  void ClusterTimeOverlappedArcs(SymbolTable *syms, VectorFst<ArcT> *fst) {
    
    queue<int32> states;  
  }
}

#endif
