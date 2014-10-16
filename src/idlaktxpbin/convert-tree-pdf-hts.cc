// idlaktxp/convert-tree-pdf-hts.cc

// Copyright 2014 CereProc Ltd.  (Author: Matthew Aylett)

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
//

#include <pugixml.hpp>
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "idlaktxp/idlaktxp.h"
#include "hmm/hmm-topology.h"
#include "tree/context-dep.h"
#include "tree/event-map.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "matrix/matrix-lib.h"
///#include "idlaktxp/convert-tree-pdf-hts.h"
namespace kaldi {
void _output_tree(const std::string &tab, const EventMap * emap, LookupMap * lkp);
void _readqsets(std::istream &is, std::istream &htsis, LookupMap * lkp);
}
// This takes tree and gmm based model output from the idlak build system
// together with HTS question set definitions and writes pdfs and trees
// which are compatible with hts engine.

int main(int argc, char *argv[]) {
  using namespace kaldi;
  LookupMap kaldi2htsqset;
  const char *usage =
      "Convert kaldi tree and gmm based model to an HTS format\n"
      "Usage:   [options] convert-tree-pdf-hts tree model\n";
  // input output variables
  std::string tree_in_filename;
  std::string model_in_filename;
  std::string kaldiqset_filename;
  std::string htsqset_filename;
  std::string tab;
  ContextDependency ctx_dep;
  TransitionModel trans_model;
  AmDiagGmm am_gmm;
  const DiagGmm * pdf;
  kaldi::int32 i, j;
  kaldi::Vector<BaseFloat> weights;
  kaldi::Matrix<BaseFloat> means;
  kaldi::Matrix<BaseFloat> vars;
  try {
    kaldi::TxpParseOptions po(usage);
    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }
    kaldiqset_filename = po.GetArg(1);
    htsqset_filename = po.GetArg(2);
    tree_in_filename = po.GetArg(3);
    model_in_filename = po.GetArg(4);
    ReadKaldiObject(tree_in_filename, &ctx_dep);
    bool binary_read;
    Input ki(model_in_filename, &binary_read);
    trans_model.Read(ki.Stream(), binary_read);
    am_gmm.Read(ki.Stream(), binary_read);
    for (i = 0; i < am_gmm.NumPdfs(); i++) {
      pdf = &am_gmm.GetPdf(i);
      // only one guassian allowed
      if (pdf->NumGauss() != 1) {
        KALDI_ERR << "Model as more than 1 Guassian (" << pdf->NumGauss() << ")";
      }
      weights = pdf->weights();
      // output weights
      //std::cout << weights(0) << " ";
      pdf->GetMeans(&means);
      pdf->GetVars(&vars);
      // output means
      for (j = 0; j < pdf->Dim(); j++) {
        //std::cout << means(0, j) << " ";
      }
      // output variances
      for (j = 0; j < pdf->Dim(); j++) {
        //std::cout << vars(0, j);
        //if (j < pdf->Dim() - 1) std::cout << " ";
        //else std::cout << "\n";
      }
    }    
    Input htsqset;
    htsqset.OpenTextMode(htsqset_filename);
    Input kaldiqset;
    kaldiqset.OpenTextMode(kaldiqset_filename);
    _readqsets(kaldiqset.Stream(), htsqset.Stream(), &kaldi2htsqset);
    _output_tree(tab, &ctx_dep.ToPdfMap(), &kaldi2htsqset);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

namespace kaldi {
//TODO
// For tree

// get event maps

// recursive print val
// or key vals
// yes
// no
// Then replace with HTS questions and vals
void _output_tree(const std::string &tab, const EventMap * emap, LookupMap * lkp) {
  std::string newtab = tab + "  ";
  std::string kaldiq;
  std::vector<EventMap*> out;
  const ConstIntegerSet<EventValueType> * yesset;
  LookupMap::const_iterator lookup;
  emap->GetChildren(&out);
  if (out.size()) {
    yesset = emap->YesSet();
    kaldiq.clear();
    std::ostringstream oss(kaldiq);
    oss << emap->EventKey() << " ?";
    std::cout << newtab << "Y:" << emap->EventKey() << "\n" << newtab << "[";
    for (ConstIntegerSet<EventValueType>::iterator iter = yesset->begin();
         iter != yesset->end();
         ++iter) {
      oss << " " << *iter;
      std::cout << *iter << " ";
    }
    std::cout << "]\n";
    std::cout << newtab << oss.str() << "\n";
    lookup = lkp->find(oss.str());
    if (lookup != lkp->end()){
      std::cout << newtab << lookup->second  << "\n";
    }
    _output_tree(newtab, out[0], lkp);
    _output_tree(newtab, out[1], lkp);
  }
  else {
    std::cout  << newtab << "A:" << static_cast<const ConstantEventMap * >(emap)->GetAnswer() << "\n";
  }
}

void _readqsets(std::istream &is, std::istream &htsis, LookupMap * lkp) {
  std::string line;
  std::string htsline;
  std::vector<std::string> stringvector;
  while (std::getline(is, line)) {
    if (!std::getline(htsis, htsline)) {
      KALDI_ERR << "Fewer hts questions than kaldi in cex";
    }
    //std::cout  << line;
    //std::cout  << htsline;
    std::istringstream iss(htsline);
    SplitStringToVector(iss.str(), " \t\r", true, &stringvector);
    std::cout << line << "\n";
    std::cout << stringvector[1] << "\n";
    lkp->insert(LookupItem(line, stringvector[1]));
  }
  if (std::getline(htsis, htsline)) {
    KALDI_ERR << "Fewer kaldi questions than hts in cex";
  }
}



}
