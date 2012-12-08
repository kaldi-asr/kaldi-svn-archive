// idlaktxp/mod-phrasing.cc

// Copyright 2012 CereProc Ltd.  (Author: Matthew Aylett)

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

#include "./txpmodule.h"

namespace kaldi {

static bool _copy_until_break(pugi::xml_node *parent,
                              pugi::xml_node *phrasenode,
                              pugi::xml_node *firstbreak,
                              pugi::xml_node *lastbreak);

static bool _is_utt_final(const pugi::xml_node &spt);

TxpPhrasing::TxpPhrasing(const std::string &tpdb, const std::string &configf)
    : TxpModule("phrasing", tpdb, configf) {
}

TxpPhrasing::~TxpPhrasing() {
}

bool TxpPhrasing::Process(pugi::xml_document * input) {
  bool final = false;
  pugi::xml_node uttnode, phrasenode, rootnode, lastbreak, firstbreak;
  rootnode = input->document_element();
  uttnode = rootnode.append_child("utt");
  lastbreak = rootnode.select_nodes("//break[last()]").first().node();
  firstbreak = rootnode.select_nodes("//break[1]").first().node();
  // firstbreak.print(std::cout);
  // lastbreak.print(std::cout);
  while (!final) {
    phrasenode = uttnode.append_child("spt");
    final = _copy_until_break(&rootnode, &phrasenode, &firstbreak, &lastbreak);
    // phrasenode.print(std::cout);
    lastbreak = rootnode.select_nodes("//break[last()]").first().node();
    firstbreak = rootnode.select_nodes("//break[1]").first().node();
    if (_is_utt_final(phrasenode))
       uttnode = rootnode.append_child("utt");
  }
  return true;
}

static bool _copy_until_break(pugi::xml_node *parent,
                              pugi::xml_node *phrasenode,
                              pugi::xml_node *firstbreak,
                              pugi::xml_node *lastbreak) {
  pugi::xml_node child, nextchild, childcopy;

  for (child = parent->first_child(); child; child = nextchild) {
    // get next child before current child is deleted
    nextchild = child.next_sibling();
    if (child.type() != pugi::node_element) {
      phrasenode->append_copy(child);
      parent->remove_child(child);
    } else {
      if (!strcmp(child.name(), "utt")) {
        continue;
      } else if (!strcmp(child.name(), "break")) {
        if (child == *firstbreak) {
          childcopy = phrasenode->append_copy(child);
          parent->remove_child(child);
        } else {
          childcopy = phrasenode->append_copy(child);
          // Split the break
          if (child != *lastbreak) {
            childcopy.attribute("time").set_value(
                child.attribute("time").as_float() / 2.0f);
            child.attribute("time").set_value(
                childcopy.attribute("time").as_float());
            return false;
          } else {
            parent->remove_child(child);
          }
        }
      } else if (!strcmp(child.name(), "tk") || !strcmp(child.name(), "ws")) {
        // if (!child.attribute("norm").empty())
        //  std::cout << child.attribute("norm").value() << "\n";
        childcopy = phrasenode->append_copy(child);
        parent->remove_child(child);
      } else {
        childcopy = phrasenode->append_child(child.name());
        for (pugi::xml_attribute a = child.first_attribute();
            a;
            a = a.next_attribute()) {
          childcopy.append_attribute(a.name()).set_value(a.value());
        }
        if (_copy_until_break(&child, &childcopy, firstbreak, lastbreak)) {
          parent->remove_child(child);
        } else {
          return false;
        }
      }
    }
  }
  return true;
}

static bool _is_utt_final(const pugi::xml_node &spt) {
  pugi::xml_node lastbreak;
  pugi::xpath_node_set breaks;
  breaks = spt.select_nodes("//break");
  lastbreak = breaks[breaks.size() - 1].node();
  // lastbreak.print(std::cout);
  //std::cout << "!" << lastbreak.attribute("type").value() << "\n";
  if (lastbreak.attribute("type").as_int(0) == 4) return true;
  return false;
}

}  // namespace kaldi
