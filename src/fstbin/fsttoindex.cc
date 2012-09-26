// fstbin/fsttoindex.cc

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

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fst/fstlib.h"
#include "fstext/fstext-utils.h"
#include "fstext/mapper.h"

#define kEpsilon 0

int main(int argc, char *argv[]) {
  try {
    using namespace fst;
    using namespace kaldi;
    using namespace std;
    using kaldi::int32; //disambig

    const char *usage = 
      "Create an inverted index of an WFST."
      "\n"
      "fsttoindex [options] table-symbol-rspecifier fsts-rspecifier "
      "fsts-wspecifier.\n";
    
    ParseOptions po(usage);
    po.Read(argc, argv);
    
    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }
    
    string isyms_rspecifier = po.GetArg(1),
           fsts_rspecifier = po.GetArg(2),
           fsts_wspecifier = po.GetArg(3);
    
    SequentialTableReader<VectorFstHolder> fst_reader(fsts_rspecifier);
    TableWriter<VectorFstHolder> fst_writer(fsts_wspecifier);

    //Load input symbol table
    SymbolTable *itable = SymbolTable::ReadText(isyms_rspecifier);
   
    //For checking that everything is ok
    int32 n_done = 0;

    //Browse all FSTs
    for (; !fst_reader.Done(); fst_reader.Next()) {
      //Load lattice
      string key = fst_reader.Key();
      VectorFst<StdArc> lattice = fst_reader.Value();
      
      //We cast the lattice in a log lattice (i.e. WFST) for the following of
      //the algorithm 
      VectorFst<LogArc> *lattice_log = new VectorFst<LogArc>;
      Cast(lattice, lattice_log);
      
      //Compute the shortest distance for all the states.
      vector<LogWeight> distance;
      ShortestDistance(*lattice_log, 
                       &distance);    

      //Compute the reverse shortest distance for all the states.
      vector<LogWeight> rdistance;
      ShortestDistance(*lattice_log, 
                       &rdistance, true);   
    
      //Map all output symbols to <eps>
      OutputLabelToEpsilonMapper mapper;
      ArcMap<LogArc, OutputLabelToEpsilonMapper> (lattice_log, &mapper);

      //Creating the new graph
      //Create new initial and final states
      LogArc::StateId initialstate = lattice_log->AddState();
      lattice_log->SetStart(initialstate);
      LogArc::StateId finalstate = lattice_log->AddState();
      lattice_log->SetFinal(finalstate, 
                            LogWeight::One());
      
      //Get the symbol for the corresponding utterance
      int32 utt_symbol = itable->Find(key);

      //For each state, we create two new transitions 
      //  (initstate, <eps>, <eps>, d[state], state)
      //  (finalstate, <eps>, utt, f[state], state
      LogArc::StateId stateid = 0;
      vector<LogWeight>::iterator itdistance = distance.begin();
      vector<LogWeight>::iterator itrdistance = rdistance.begin();
      for (; itdistance < distance.end(); itdistance++, itrdistance++) {

        //LatticeArc(input_label, output_label, weight, destination_state)
        //Zero label ID is reserved for epsilon label 
        //(http://www.openfst.org/twiki/bin/view/FST/FstQuickTour)
        lattice_log->AddArc(initialstate, 
                            LogArc(kEpsilon, 
                                   kEpsilon,
                                   *itdistance,
                                   stateid)); 
        lattice_log->AddArc(stateid, 
                           LogArc(kEpsilon, 
                                 utt_symbol, 
                                 *itrdistance, 
                                 finalstate));

        //The old final state isn't final any more.
        if (lattice_log->Final(stateid) != LogWeight::Zero()) {
          lattice_log->SetFinal(stateid,
                                LogWeight::Zero());
        }
          
        stateid++;
      }
        
    //Cast it to a Tropical WFST before to write it 
    VectorFst<StdArc> tmp;
    Cast(*lattice_log, 
         &tmp);
    fst_writer.Write(key,
                     tmp);
  
    n_done++;
    }

    KALDI_LOG << n_done << " WFSTs has been indexed";
    
  } catch (std::exception e) {
    e.what();
    return -1;
  }
}
