// sgmmbin/sgmm-est.cc

// Copyright 2012  Karlsruhe Institue Technology
// Author:  Thang Vu

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


#include <iostream>
#include <fstream>
#include "base/kaldi-common.h"
#include "util/common-utils.h"

#include "sgmm/am-sgmm.h"
#include "hmm/transition-model.h"
#include "sgmm/estimate-am-sgmm.h"

namespace kaldi {

std::vector<std::vector<int32> > ReadList(const std::string &rxfilename ) {
     std::vector<std::vector<int32> > vector;
     Input ki(rxfilename, false); // false == text mode.
     std::string line;
     while (std::getline(ki.Stream(), line)) {
           std::vector<int32> list; 
           if (!SplitStringToIntegers(line, " \t\n\r", true, &list))
		           KALDI_EXIT << "Bad line " << line << " in this file.";
           vector.push_back(list);
     }
     return vector;
 }

std::vector<Vector<double> > ReadDoubleList(const std::string &rxfilename) {
     std::vector<Vector<double> > vector;
     Input ki(rxfilename, false); // false == text mode.
     std::string line;
     while (std::getline(ki.Stream(), line)) {
           std::vector<std::string> list; 
           SplitStringToVector(line, " \t\n\r", &list, true);
	   double value;
	   Vector<double> dList(list.size());
           for (int32 l = 0; l < list.size(); l++) {
		   ConvertStringToReal(list[l], &value );
		   dList(l) = value;
	   }
           vector.push_back(dList);
     }
     return vector;
 }
}


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    const char *usage =
        "Estimate SGMM model parameters from accumulated stats.\n"
        "Usage: sgmm-est [options] <model-in> <stats-in> <model-out> <states> <questions> \n";

    bool binary_write = false;
    std::string update_flags_str = "vMNwcS";
    kaldi::TransitionUpdateConfig tcfg;
    kaldi::MleAmSgmmOptions sgmm_opts;
    int32 split_substates = 0;
    int32 number_subsets = 0;
    int32 increase_phn_dim = 0;
    int32 increase_spk_dim = 0;
    int32 N = 3; // for triphones case
    bool remove_speaker_space = false;
    BaseFloat perturb_factor = 0.01;
    BaseFloat power = 0.2;
    BaseFloat max_cond = 100;
    std::string occs_out_filename;

    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("split-substates", &split_substates, "Increase number of "
        "substates to this overall target.");
    po.Register("increase-subsets", &number_subsets, "Increase number of "
        "sub sets to this overall target."); 
    po.Register("increase-phn-dim", &increase_phn_dim, "Increase phone-space "
        "dimension to this overall target.");
    po.Register("increase-spk-dim", &increase_spk_dim, "Increase speaker-space "
        "dimension to this overall target.");
    po.Register("remove-speaker-space", &remove_speaker_space, "Remove speaker-specific "
                "projections N");
    po.Register("power", &power, "Exponent for substate occupancies used while"
        "splitting substates.");
    po.Register("perturb-factor", &perturb_factor, "Perturbation factor for "
        "state vectors while splitting substates.");
    po.Register("max-cond-split", &max_cond, "Max condition number of smoothing "
        "matrix used in substate splitting.");
    po.Register("write-occs", &occs_out_filename, "File to write state "
                "occupancies to.");
    po.Register("update-flags", &update_flags_str, "Which SGMM parameters to "
                "update: subset of vMNwcS.");
    tcfg.Register(&po);
    sgmm_opts.Register(&po);

    po.Read(argc, argv);
    if (po.NumArgs() != 7) {
      po.PrintUsage();
      exit(1);
    }
    std::string model_in_filename = po.GetArg(1),
        stats_filename = po.GetArg(2),
        model_out_filename = po.GetArg(3),
        states_file = po.GetArg(4),
	questions_file = po.GetArg(5),
	subset_outfile = po.GetArg(6),
	vectors_outfile = po.GetArg(7);

    kaldi::SgmmUpdateFlagsType acc_flags = StringToSgmmUpdateFlags(update_flags_str);

    AmSgmm am_sgmm;
    TransitionModel trans_model;
    {
      bool binary;
      Input is(model_in_filename, &binary);
      trans_model.Read(is.Stream(), binary);
      am_sgmm.Read(is.Stream(), binary);
    }

    Vector<double> transition_accs;
    MleAmSgmmAccs sgmm_accs;
    {
      bool binary;
      Input is(stats_filename, &binary);
      transition_accs.Read(is.Stream(), binary);
      sgmm_accs.Read(is.Stream(), binary, true);  // true == add; doesn't matter here.
    }

    {  // Update transition model.
      BaseFloat objf_impr, count;
      trans_model.Update(transition_accs, tcfg, &objf_impr, &count);
      KALDI_LOG << "Transition model update: average " << (objf_impr/count)
                << " log-like improvement per frame over " << (count)
                << " frames.";
    }

    sgmm_accs.Check(am_sgmm, true); // Will check consistency and print some diagnostics.
    
    std::vector<std::vector<int32> > states = ReadList(states_file);
    KALDI_LOG << "There are in total " << states.size() << " contexts.";

    std::vector<std::vector<std::vector<int32> > > questions;
    std::vector<std::vector<int32> > questions_per_state = ReadList(questions_file);
    std::vector<std::vector<int32> > questions_position(N);
    KALDI_LOG << "There are " << questions_per_state.size() << " question for each context position";

    //assume that we have tri phone, for left, central and right phone we use the same questions
    for (int32 n = 0; n < N; ++n) {
	questions_position[n].push_back(n);
    	questions.push_back(questions_per_state);
    }
    questions.push_back(questions_position);

    KALDI_LOG << "There are questions for " << questions.size(); 
    
    std::vector<std::vector<int32> > subset(number_subsets);
    std::vector<Vector<double> > vectors(number_subsets);
    if (access(subset_outfile.c_str(), F_OK) == -1) {  
    	subset.erase(subset.begin(), subset.end());
    	vectors.erase(vectors.begin(), vectors.end());
    } else {
	subset = ReadList(subset_outfile);
	vectors = ReadDoubleList(vectors_outfile);
    } 
   
    {  // Update SGMM. 
      kaldi::MleAmSgmmUpdater sgmm_updater(sgmm_opts);
      if (number_subsets == 0) 
	 sgmm_updater.Update(sgmm_accs, &am_sgmm, states, questions, &subset, &vectors, -1, acc_flags);
      sgmm_updater.Update(sgmm_accs, &am_sgmm, states, questions, &subset, &vectors, number_subsets, acc_flags);  
    }

    if (split_substates != 0 || !occs_out_filename.empty()) {  // get state occs
      Vector<BaseFloat> state_occs;
      sgmm_accs.GetStateOccupancies(&state_occs);

      if (split_substates != 0) {
        am_sgmm.SplitSubstates(state_occs, split_substates, perturb_factor,
                               power, max_cond);
        am_sgmm.ComputeDerivedVars();  // recompute normalizers...
      }

      if (!occs_out_filename.empty()) {
        kaldi::Output os(occs_out_filename, binary_write);
        state_occs.Write(os.Stream(), binary_write);
      }
    }

    if (increase_phn_dim != 0 || increase_spk_dim != 0) {
      // Feature normalizing transform matrix used to initialize the new columns
      // of the phonetic- or speaker-space projection matrices.
      kaldi::Matrix<BaseFloat> norm_xform;
      ComputeFeatureNormalizer(am_sgmm.full_ubm(), &norm_xform);
      if (increase_phn_dim != 0)
        am_sgmm.IncreasePhoneSpaceDim(increase_phn_dim, norm_xform);
      if (increase_spk_dim != 0)
        am_sgmm.IncreaseSpkSpaceDim(increase_spk_dim, norm_xform);
    }
    if (remove_speaker_space) {
      KALDI_LOG << "Removing speaker space (projections N_)";
      am_sgmm.RemoveSpeakerSpace();
    }

    {
      Output os(model_out_filename, binary_write);
      trans_model.Write(os.Stream(), binary_write);
      am_sgmm.Write(os.Stream(), binary_write, kSgmmWriteAll);
    }
    
    {  
      std::ofstream file;
      file.open(subset_outfile.c_str());
      for (int32 s=0; s < number_subsets; s++) {
	      for (int32 c=0; c < subset[s].size(); c++)
		      file << subset[s][c] << " ";
	      file << "\n";
      }
      file.close();
    } 

    {
      std::ofstream file;
      file.open(vectors_outfile.c_str());
      for (int32 s=0; s < number_subsets; s++) {
	      for (int32 c=0; c < (int)vectors[s].Dim(); c++)
		      file << vectors[s](c) << " ";
	      file << "\n";
      }
      file.close();
    }
    KALDI_LOG << "Written model to " << model_out_filename;
    return 0;
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}


