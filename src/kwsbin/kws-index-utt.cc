// kwsbin/kws-index-utt.cc

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


#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fstext/fstext-lib.h"
#include "lat/kaldi-lattice.h"
#include "fst/fstlib.h"
#include "kws/kaldi-kws.h"

#include <sstream>

using namespace kaldi;
using namespace fst;

int main(int argc, char *argv[]) {
  try {
    typedef kaldi::int32 int32;

    BaseFloat acoustic_scale = 1.0;
    vector<string> accepted_strings;
    accepted_strings.push_back("input");
    accepted_strings.push_back("factor_selection");
    accepted_strings.push_back("index");
    vector<Filter> filters;
    string  strfilters = "";

    const char *usage =
      "Create an inverted index of utterance given a set of lattices, "
      "it also creates an input/output symbol table given the original "
      "input symbol table provided. Furthermore, a sequence of filtering "
      "FSTs could be provided via the \"--apply-filters\" parameter. "
      "This argument accepts a string in the format  : "
      "\"application1,fst:application2.fst:...\". Where \"applicationX\" "
      "specify at wich step the filter should be applied (possible values are "
      "\"input\", \"factor_selection\" and \"index\") anf \"fst\" is the "
      "filename of the filtering FST.\n" 
      "Usage: kws-index-utt [options] lattice-rspecifier isyms-rspecificer "
      "iosyms-wspecifier index-wspecifier\n";

    ParseOptions po(usage);
    po.Register("acoustic-scale", 
                &acoustic_scale, 
                "Scaling factor for acoustic likelihoods");
    po.Register("apply-filters", 
                &strfilters, 
                "Apply one or many filters at the specified step of the "
                "algorithm.\n");

    po.Read(argc, argv);

    //Set here the number of minimum required arguments
    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string lats_rspecifier = po.GetArg(1),
      isyms_rspecifier = po.GetArg(2),
      iosyms_wspecifier = po.GetArg(3),
      fst_wspecifier = po.GetArg(4);

    //Create filters
    CreateFilters<LogArc>(strfilters, accepted_strings, filters);
    
    SequentialCompactLatticeReader lattice_reader(lats_rspecifier);
    
    //Load input symbol table
    SymbolTable *itable = SymbolTable::ReadText(isyms_rspecifier);

    //Rename types for an easier reading
    typedef LogArc Arc;
    typedef LogWeight Weight;

    VectorFst<Arc> sumfst;
    VectorFst<Arc> sumdeterminizedfst;
    VectorFst<Arc> disambigfst;
    VectorFst<Arc> finalfst;
    VectorFst<Arc> *filtered_finalfst = NULL;

    //There is no failure mode, barring a crash.
    int32 n_done = 0;
    
    for (; !lattice_reader.Done(); lattice_reader.Next()) {
      //Read lattice
      std::string key = lattice_reader.Key();
      CompactLattice clat = lattice_reader.Value();
      lattice_reader.FreeCurrent();

      Lattice lat;
      VectorFst<Arc> fst;
      VectorFst<Arc> pushedfst;
      VectorFst<Arc> *filtered_pushedfst = NULL;
      VectorFst<Arc> *filtered_fselectionfst = NULL;
      VectorFst<Arc> determinizedfst;
      vector<Weight> distance;
      vector<Weight> rdistance;//reverse distance

      //We don't need the alignments
      RemoveAlignmentsFromCompactLattice(&clat);

      //Convert lattice to non-compact form
      ConvertLattice(clat, &lat);  

      // Convert the lattice in the log semiring
      LatticeLogConverter converter(acoustic_scale);
      ConvertFST(lat,  &pushedfst,(Converter*)&converter); 
       
      //Apply filters for INPUT step 
      ApplyFilters<Arc>(filters, 0, pushedfst,
                   filtered_pushedfst);
      
      //Weight-pushing over the log semiring
      //We remove the total weight as our transducer are not stochastic
      Push<Arc>(filtered_pushedfst, REWEIGHT_TO_INITIAL, kDelta, true);

      //Compute the shortest distance for all the states.
      //It should be done before to create new initial and final states
      ShortestDistance(*filtered_pushedfst, &distance);    

      //Compute the reverse shortest distance for all the states.
      //It should be done before to create new initial and final states
      ShortestDistance(*filtered_pushedfst, &rdistance, true);    

      //Initiating the new graph
      //Create new initial and final states
      Arc::StateId initialstate = filtered_pushedfst->AddState();
      (*filtered_pushedfst).SetStart(initialstate);
      Arc::StateId finalstate = filtered_pushedfst->AddState();
      filtered_pushedfst->SetFinal(finalstate, Weight::One());

      //Create a new reference in the symbol table for the utterance
      int32 utt_symbol = itable->AvailableKey(); 
      itable->AddSymbol(key, utt_symbol);

      //For each state, we create two new transitions 
      //  (initstate, <eps>, <eps>, d[state], state)
      //  (finalstate, utt, n_done+1, f[state], state
      int32 stateid = 0;
      vector<Weight>::iterator itdistance = distance.begin();
      vector<Weight>::iterator itrdistance = rdistance.begin();
      for (; itdistance < distance.end(); itdistance++, itrdistance++) {

        //LatticeArc(input_label, output_label, weight, destination_state)
        //Zero label ID is reserved for epsilon label 
        //(http://www.openfst.org/twiki/bin/view/FST/FstQuickTour)
        filtered_pushedfst->AddArc(initialstate, 
                                   Arc(0, 0, *itdistance, stateid)); 
        filtered_pushedfst->AddArc(stateid, Arc(0, utt_symbol, *itrdistance, 
                                                finalstate));

        //The old final state isn't final any more.
        if (filtered_pushedfst->Final(stateid) != Weight::Zero()) {
          filtered_pushedfst->SetFinal(stateid, Weight::Zero());
        }
          
        stateid++;
      }

      //Apply filters for FACTOR_SELECTION step 
      ApplyFilters<Arc>(filters, 1, *filtered_pushedfst,
                   filtered_fselectionfst);
      
      //Optimization: 
      //Epsilon-Removal, derterminization and minimization 
      //over the log semiring
      RmEpsilon(filtered_fselectionfst);
      Determinize(*filtered_fselectionfst, &determinizedfst);
      Push<Arc>(filtered_fselectionfst, REWEIGHT_TO_INITIAL, kDelta, false);
      MinimizeEncoded(&determinizedfst);

      //Should not be used : push label and remove total weight.
      //The output labels should not be pushed to the beginning
      //and the total weight contains the expected counts. 
      //Generate a warning (control reaches end of non-void function)
      //Minimize(&determinizedfst);

      //Add the resulting fst to the global index
      Union(&sumfst, determinizedfst);

      // We don't need these fsts any more
      delete filtered_pushedfst;
      delete filtered_fselectionfst;

      KALDI_LOG << "Done utterance : " << key;

      //Lattice has been treated
      n_done++;
    }

    //Remove epsilon-transitions added with the union
    RmEpsilon(&sumfst);
    
    //Disambiguation symbols the fst to do the determinization
    DisambigFinalArcConverter disambig_converter;
    ConvertFST(sumfst, &disambigfst, &disambig_converter);  
    
    //Final step of the algorithm, determinize the union of all transducers
    Determinize(disambigfst, &sumdeterminizedfst);
    
    //Remove disambiguation symbols
    RmDisambigFinalArcConverter rmdisambig_converter;
    ConvertFST(sumdeterminizedfst, &finalfst, &rmdisambig_converter);   

    //Apply filters for INDEX step 
    ApplyFilters<Arc>(filters, 2, finalfst,
        filtered_finalfst);

    //Write the resulting transducer
    filtered_finalfst->Write(fst_wspecifier);

    //Write the input/output symbol table
    itable->WriteText(iosyms_wspecifier);

    KALDI_LOG << "Done indexing " << n_done << " lattices.";
    return (n_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
