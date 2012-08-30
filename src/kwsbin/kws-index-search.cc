// kwsbin/kws-index-search.cc

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

using namespace kaldi;
using namespace fst;

int main(int argc, char *argv[]) {
  try {
    typedef kaldi::int32 int32;
    
    vector<Filter> filters;
    vector<string> accepted_strings;
    accepted_strings.push_back("index");
    accepted_strings.push_back("query");
    string  strfilters = "";
    
    const char *usage =
      "Search for a token in an inverted index.\n" 
      "Usage: kws-index-search [options] "
      "fstindex-rspecifier fstquery-rspecifier fstresult-wspecifier\n";

    ParseOptions po(usage);
    po.Register("apply-filters", 
                &strfilters, 
                "Apply one or many filters at the specified step of the "
                "algorithm.\n");
    
    po.Read(argc, argv);

    //Set here the number of minimum required arguments
    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }
 
    std::string fstindex_rspecifier = po.GetArg(1),
                fstquery_rspecifier = po.GetArg(2),
                fstresult_wspecifier = po.GetArg(3);

    
    //Create filters
    CreateFilters<LogArc>(strfilters, accepted_strings, filters);
   
   typedef LogArc Arc;
   
    //Load index and query transducers
    VectorFst<Arc> indexfst = *(VectorFst<Arc>::Read(fstindex_rspecifier));
    VectorFst<Arc> *filtered_indexfst = NULL;
    
    //Apply filters for query step
    ApplyFilters<Arc>(filters, 0, indexfst,
        filtered_indexfst);
    
    VectorFst<Arc> queryfst = *(VectorFst<Arc>::Read(fstquery_rspecifier)); 
    VectorFst<Arc> *filtered_queryfst = NULL;
       
    //Apply filters for query step
    ApplyFilters<Arc>(filters, 1, queryfst,
        filtered_queryfst);
  
    VectorFst<Arc> composedfst;
    VectorFst<Arc> determinizedfst;

    //Compose the index and the query to compute the result
    Compose<Arc>(queryfst, indexfst, &composedfst);
    composedfst.Write("results/composed.fst");

    //Projection on the output
    Project(&composedfst, PROJECT_OUTPUT);

    //Remove epsilon transition
    RmEpsilon(&composedfst);

    //Determinization
    Determinize(composedfst, &determinizedfst);

    //Write the resulting FST
    determinizedfst.Write(fstresult_wspecifier);
    
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
