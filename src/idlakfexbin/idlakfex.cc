// idlakfexbin/idlakfex.cc

// Copyright 2013 CereProc Ltd.  (Author: Matthew Aylett)

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
#include "idlakfex/fex.h"

/// Example program that carries out feature extraction on a syllabified XML structure
/// You need a text processing database (tpdb) to run this. An example is in
/// ../../idlak-data/arctic-bdl/tpdb
int main(int argc, char *argv[]) {
  const char *usage =
      "Extract feature set from syllabified XML input\n"
      "Usage:  idlakfex [options] xml_input feature_output\n"
      "e.g.: ./idlakfex --tpdb=../../idlak-data/arctic-bdl/tpdb ../idlaktxp/mod-syllabify-out002.xml output.dat\n" //NOLINT
      "e.g.: cat  ../idlaktxp/mod-syllabify-out002.xml | idlakfex --tpdb=../../idlak-data/arctic-bdl/tpdb - - > output.dat\n"; //NOLINT
  // input output variables 
  std::string filein;
  std::string fileout;
  std::string tpdb;
  std::string configf;
  std::string input;
  std::ostream *out;
  std::ofstream fout;

  try {
    kaldi::ParseOptions po(usage);
    po.Register("tpdb", &tpdb,
                "Text processing database (directory XML language/speaker files)"); //NOLINT
    po.Read(argc, argv);
    // Must have inpout and output filenames for XML
    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }
    filein = po.GetArg(1);
    fileout = po.GetArg(2);
    // Allow piping
    if (fileout == "-") {
      out = &(std::cout);
    } else {
      fout.open(fileout.c_str());
      out = &fout;
    }
    // Set up input/output streams
    bool binary;
    kaldi::Input ki(filein, &binary);
    kaldi::Output kio(fileout, binary);
    // Setup feature extraction
    kaldi::Fex fex("default");
    fex.Parse(tpdb.c_str());
    // Use pujiXMl to read input file
    pugi::xml_document doc;
    pugi::xml_parse_result r = doc.load(ki.Stream(), pugi::encoding_utf8);
    if (!r) {
      KALDI_ERR << "PugiXML Parse Error in Input Stream" << r.description()
                << "Error offset: " << r.offset;
    }
    // Output result for each phone and break tag
    pugi::xpath_node_set tks =
        doc.document_element().select_nodes("//phon|//break");
    tks.sort();
    int i = 0;
    for (pugi::xpath_node_set::const_iterator it = tks.begin();
         it != tks.end();
	 ++it, i++) {
      pugi::xml_node tk = (*it).node();
      if (!strcmp(tk.name(), "break")) {
        std::cout << tk.attribute("type").value() << "\n";
      } else if (!strcmp(tk.name(), "phon")) {
        std::cout << tk.attribute("val").value() << "\n";
      }
    }
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
