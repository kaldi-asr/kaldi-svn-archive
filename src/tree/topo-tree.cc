// tree/topo-tree.cc

// Copyright 2013  Korbinian Riedhammer

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

#include <algorithm>
#include <utility>
#include <list>

#include "tree/topo-tree.h"

namespace kaldi {

std::string EventTypeToString(const EventType &e, int32 P) {
  std::stringstream ss;
  ss << "[ ";
  for (int32 i = 1; i < e.size(); ++i) {
    if (e[i].second == 0) {
      ss << "- ";
      continue;
    }

    if (i == P+1) {
      ss << "/" << e[i].second << " ";
      if (e[0].second == -1)
        ss << "-";
      else
        ss << e[0].second;
      ss << "/ ";
    }
    else
      ss << e[i].second << " ";
  }
  ss << "] ctx=" << EventTypeContextSize(e, P) << " bal=" << EventTypeBalance(e, P);

  if (e[0].second == kNoPdf)
    ss << " PdfClass=kNoPdf";
  else
    ss << " PdfClass=" << e[0].second;

  return ss.str();
}

int32 TopoNode::TraverseSpecializations(std::vector<TopoNode *> &node_list) {
  std::list<TopoNode *> agenda;
  agenda.push_back(this);

  while (agenda.size() > 0) {
    TopoNode *n = agenda.front();
    agenda.pop_front();

    for (std::vector<TopoNode *>::iterator it = n->specializations_.begin();
        it != n->specializations_.end(); ++it) {
      node_list.push_back(*it);
      agenda.push_back(*it);
    }
  }

  return node_list.size();
}


void TopoNode::Clear() {
  if (specializations_.size() > 0) {
    for (std::vector<TopoNode *>::iterator it = specializations_.begin();
        it != specializations_.end(); ++it) {
      (*it)->Clear();
      delete *it;
    }

    specializations_.clear();
  }
}


void TopoNode::Read(std::istream &is, bool binary) {
  // make sure this node is empty
  ClearPointers();

  ReadEventType(is, binary, &event_type_);
  ReadBasicType(is, binary, &pdf_id_);
  int32 n;
  ReadBasicType(is, binary, &n);
  for (int32 i = 0; i < n; ++i) {
    TopoNode *node = new TopoNode(this);
    node->Read(is, binary);
    specializations_.push_back(node);
  }
}


void TopoNode::Write(std::ostream &os, bool binary) const {
  WriteEventType(os, binary, event_type_);
  WriteBasicType(os, binary, pdf_id_);
  WriteBasicType(os, binary, specializations_.size());
  for (int32 i = 0; i < specializations_.size(); ++i)
    specializations_[i]->Write(os, binary);
}

TopoNode *TopoTree::FindSpecialization(const TopoNode *node, const EventType &event_type) const {
  KALDI_ASSERT(node != NULL);
  KALDI_ASSERT(event_type[0].first == kPdfClass);

  for (int32 i = 0; i < node->specializations_.size(); ++i) {
    if (EventTypeComparison(event_type, node->specializations_[i]->event_type_, P_).Fits())
      return node->specializations_[i];
  }

  return NULL;
}


TopoNode *TopoTree::Compute(const EventType &event_type) const {
  KALDI_ASSERT(event_type.size() == N_ + 1);

  EventValueType phone = event_type[P_ + 1].second;

  // we are unable to map this phone
  if (roots_.find(phone) == roots_.end())
    return NULL;

  // get the root node, search from there
  // should be TopoNode *boot = roots_[phone]; but that doesn't compile. (???)
   return Compute(roots_.find(phone)->second, event_type);
}

TopoNode *TopoTree::Compute(TopoNode *root, const EventType &event_type) const {
  KALDI_ASSERT(event_type.size() == N_ + 1);

  TopoNode *cand = root;
  TopoNode *next = NULL;

  // descend down the tree
  while (EventTypeComparison(event_type, cand->event_type_, P_).Fits()) {

    // we reached a leaf;  no more descent
    if (cand->IsLeaf())
      break;

    // find the specialization that fits the EventType
    next = FindSpecialization(cand, event_type);

    // no further specialization possible, use current candidate.
    if (next == NULL)
      break;

    cand = next;
  }

  return cand;
}


bool TopoTree::Compute(const std::vector<int32> &phoneseq, int32 pdf_class,
                       int32 *pdf_id) const {
  KALDI_ASSERT(static_cast<int32>(phoneseq.size()) == N_);

  // construct event type
  EventType event_type;

  event_type.push_back(std::pair<EventKeyType,EventValueType>(kPdfClass, pdf_class));
  for (int32 i = 0; i < N_; ++i)
    event_type.push_back(std::pair<EventKeyType,EventValueType>(i, phoneseq[i]));

  TopoNode *n = Compute(event_type);

  if (n == NULL)
    return false;

  // if we have a virtual node, step up the tree
  while (n->pdf_id_ == kNoPdf && n != NULL)
    n = n->generalization_;

  // whoops, this should never happen.
  if (n == NULL)
    return false;

  (*pdf_id) = n->pdf_id_;
  return true;
}


bool TopoTree::Insert(const EventType &event_type) {
  KALDI_ASSERT(event_type[0].first == kPdfClass);

  EventValueType phone = event_type[P_ + 1].second;

  // see if the respective root node exists, create it otherwise
  if (roots_.find(phone) == roots_.end()) {
    EventType root_event;
    RootEventType(event_type, &root_event, P_);
    roots_.insert(std::pair<EventValueType, TopoNode *>(phone, new TopoNode(root_event)));
  }

  // query the topology for te best insert position
  TopoNode *ins = Compute(event_type);

  // there must be a hit, initially it will be the RootEventType
  KALDI_ASSERT(ins != NULL);

  // we already have a node for this EventType
  if (event_type == ins->event_type_)
    return false;

  return Insert(ins, new TopoNode(event_type));
}


bool TopoTree::Insert(TopoNode *target, TopoNode *node) {
  KALDI_ASSERT(target != NULL);
  KALDI_ASSERT(node != NULL);

  // do the actual insert.
  node->generalization_ = target;

  // simplest case:  first specification to add to.
  if (target->IsLeaf()) {
    target->specializations_.push_back(node);
    return true;
  }

  // regular case:  check if there is any existing specialization that we need
  // to re-position.
  std::vector<TopoNode *> reinsert;
  for (std::vector<TopoNode *>::iterator it = target->specializations_.begin();
      it != target->specializations_.end(); /* no-op */) {
    bool erase = false;

    EventTypeComparison comp(node->event_type_, (*it)->event_type_, P_);
    if (comp.IsSpecialization()) {
      // the iterated node is a specialization of the newly inserted, add it here.
      Insert(node, *it);
      erase = true;
    } else if (
        (comp.IsPartialSpecialization() == -1 && comp.IsPartialGeneralization() ==  1) ||
        (comp.IsPartialSpecialization() ==  1 && comp.IsPartialGeneralization() == -1)) {
      // e.g., we inserted x/a/x, and on the same level now is  xx/a/
      // e.g., we inserted x/a/x, and on the same level now is    /a/xx
      // traverse all its specializations and add them to re-insertion list
      (*it)->TraverseSpecializations(reinsert);
      (*it)->specializations_.clear();
    }

    // see if we had to erase the node.
    if (erase)
      target->specializations_.erase(it);
    else
      it++;
  }

  // sort the node list so that the most balanced one is front.
  target->specializations_.push_back(node);
  std::sort(target->specializations_.begin(), target->specializations_.end(), TopoNodeComparison(P_));

  // re-insert each of the nodes to make sure the tree is in consistent shape.
  if (reinsert.size() > 0) {
    std::sort(reinsert.begin(), reinsert.end(), TopoNodeComparison(P_));
    for (std::vector<TopoNode *>::iterator it = reinsert.begin();
        it != reinsert.end(); it++) {

      // make sure to clear all pointer before inserting
      (*it)->ClearPointers();

      // find the right node to insert
      Insert(Compute(target, (*it)->event_type_), *it);
    }
  }

  return true;
}


bool TopoTree::Remove(const EventType &event_type) {
  TopoNode *n = Compute(event_type);

  // see if node exists, can't delete root node
  if (n == NULL || n->IsRoot())
    return false;

  TopoNode *g = n->generalization_;

  {
    std::vector<TopoNode *>::iterator it = std::find(g->specializations_.begin(), g->specializations_.end(), n);

    // the pointer must be found.
    KALDI_ASSERT(it != g->specializations_.end());

    g->specializations_.erase(it);
  }

  // if this node had specializations, traverse them and insert them at g
  if (n->specializations_.size() > 0) {
    std::vector<TopoNode *> leaves;
    n->TraverseSpecializations(leaves);

    for (std::vector<TopoNode *>::iterator it = leaves.begin();
        it != leaves.end(); it++) {
      // make sure there are no pointers set
      (*it)->ClearPointers();

      // insert, but start search from g
      Insert(Compute(g, (*it)->event_type_), *it);
    }
  }

  return true;
}


bool TopoTree::Virtualize(const EventType &event_type) {
  TopoNode *n = Compute(event_type);

  // see if node exists
  if (n == NULL)
    return false;

  n->pdf_id_ = kNoPdf;

  return true;
}


int32 TopoTree::Populate() {
  num_pdfs_ = 0;

  for (std::map<EventValueType, TopoNode *>::iterator mit = roots_.begin();
      mit != roots_.end(); mit++) {

    // the root node is virtual
    mit->second->pdf_id_ = kNoPdf;

    std::vector<TopoNode *> leaves;
    mit->second->TraverseSpecializations(leaves);

    for (std::vector<TopoNode *>::iterator vit = leaves.begin();
        vit != leaves.end(); vit++) {
      if (!(*vit)->IsVirtual())
        (*vit)->pdf_id_ = num_pdfs_++;
    }
  }

  return num_pdfs_;
}

void TopoTree::Fill() {
  for (std::map<EventValueType, TopoNode *>::iterator it = roots_.begin();
      it != roots_.end(); it++) {
    std::vector<TopoNode *> leaves;
    it->second->TraverseSpecializations(leaves);

    std::vector<TopoNode *>::iterator it = leaves.begin();
    while (it != leaves.end()) {
      EventType event_type = (*it)->event_type_, gen;

      // generate all generalizations, cache them to insert them in the inverse
      // order
      std::list<EventType> events;
      while (GeneralizeEventType(event_type, &gen, P_, EventTypeBalance(event_type, P_) < 0)) {
        events.push_front(gen);
        event_type = gen;
      }

      // insert all generalizations in the reverse order (less tree-reorderings)
      for (std::list<EventType>::iterator ei = events.begin();
          ei != events.end(); ei++) {
        Insert(*ei);
      }

      leaves.erase(it);
    }
  }
}

void TopoTree::Read(std::istream &is, bool binary) {
  // make sure the tree is empty
  if (roots_.size() > 0)
    Clear();

  ReadBasicType(is, binary, &N_);
  ReadBasicType(is, binary, &P_);
  ReadBasicType(is, binary, &num_pdfs_);

  int32 n;
  ReadBasicType(is, binary, &n);

  for (int32 i = 0; i < n; ++i) {
    EventValueType phone;
    ReadBasicType(is, binary, &phone);

    TopoNode *node = new TopoNode(NULL);
    node->Read(is, binary);

    roots_.insert(std::make_pair(phone, node));
  }
}


void TopoTree::Write(std::ostream &os, bool binary) const {
  WriteBasicType(os, binary, N_);
  WriteBasicType(os, binary, P_);
  WriteBasicType(os, binary, num_pdfs_);

  WriteBasicType(os, binary, roots_.size());
  for (std::map<EventValueType, TopoNode *>::const_iterator it = roots_.begin();
    it != roots_.end(); it++) {
    WriteBasicType(os, binary, it->first);
    it->second->Write(os, binary);
  }
}


void TopoTree::Print(std::ostream &out) {
  out << N_ << std::endl;
  out << P_ << std::endl;

  std::list<std::pair<int32, TopoNode *> > agenda;

  for (std::map<EventValueType, TopoNode *>::iterator it = roots_.begin();
      it != roots_.end(); it++) {
    agenda.push_back(std::make_pair(0, it->second));
  }

  // depth first search
  while (agenda.size() > 0) {
    std::pair<int32, TopoNode *> pair = agenda.front();  agenda.pop_front();

    // print current
    for (int i = 0; i < pair.first; ++i)
      std::cout << "\t";

    std::cout << EventTypeToString(pair.second->event_type_, P_) << " PdfId=" << pair.second->pdf_id_ << std::endl;

    // add the children
    for (std::vector<TopoNode *>::iterator it = pair.second->specializations_.begin();
        it != pair.second->specializations_.end(); it++)
      agenda.push_front(std::make_pair(pair.first + 1, *it));
  }
}


void EventTypeComparison::compare(const EventType &ref, const EventType &chk, int32 P) {
  KALDI_ASSERT(ref[0].first == kPdfClass);
  KALDI_ASSERT(chk[0].first == kPdfClass);

  phone_ref_ = ref[P + 1].second;
  phone_chk_ = chk[P + 1].second;

  // if the phones don't match, we're done already.
  if (phone_ref_ != phone_chk_) {
    num_incompatibilities_ = 1;

    left_generalization_ = right_generalization_ = false;
    left_specialization_ = right_specialization_ = false;

    return;
  }

  pdf_class_ref_ = ref[0].second;
  pdf_class_chk_ = chk[0].second;

  num_generalizations_ = 0;
  num_specializations_ = 0;
  num_incompatibilities_ = 0;

  // compute the "edit" distance between the remainder (we'll keep track of
  // both left and right side)
  // the pdf class is done separately after this loop.
  int32 left_s, left_g, left_i;
  for (int32 i = 1; i < ref.size(); ++i) {
    // make sure both are sorted right
    KALDI_ASSERT(ref[i].first == chk[i].first);

    // see if we have a left general-/specialization, remember state for right
    // side comparison.
    if (i == P + 1) {
       left_specialization_ = IsSpecialization();
       left_generalization_ = IsGeneralization();

       left_s = num_specializations_;
       left_g = num_generalizations_;
       left_i = num_incompatibilities_;

       // reset counts for right hand side
       num_specializations_ = num_generalizations_ = num_incompatibilities_ = 0;

       // this is center phone, nothing more to do
    }

    EventValueType r = ref[i].second;
    EventValueType c = chk[i].second;

    if (r == c) {
      continue;
    } else if (r == 0) {
      num_specializations_++;
    } else if (c == 0) {
      num_generalizations_++;
    } else {
      num_incompatibilities_++;
    }
  }

  // see if there was a right general-/specialization
  right_specialization_ = IsSpecialization();
  right_generalization_ = IsGeneralization();

  // restore the full counts
  num_specializations_ += left_s;
  num_generalizations_ += left_g;
  num_incompatibilities_ += left_i;

  // treat the pdf class separately.
  if (pdf_class_ref_ == pdf_class_chk_)
    ; // nothing to do.
  else if (pdf_class_ref_ == kNoPdf && pdf_class_chk_ >= 0) {
    num_specializations_++;
    left_generalization_ = right_generalization_ = false;
  } else if (pdf_class_chk_ == kNoPdf && pdf_class_ref_ >= 0) {
    num_generalizations_++;
    left_specialization_ = right_specialization_ = false;
  } else {
    num_incompatibilities_++;
    left_generalization_ = right_generalization_ = false;
    left_specialization_ = right_specialization_ = false;
  }

  return;
}


void RootEventType(const EventType &event_type_in, EventType *event_type_out, int32 P) {
  KALDI_ASSERT(event_type_out != NULL);
  KALDI_ASSERT(event_type_in[0].first == kPdfClass);

  event_type_out->clear();
  event_type_out->insert(event_type_out->begin(), event_type_in.begin(), event_type_in.end());

  // set the pdf class to kNoPdf
  (*event_type_out)[0].second = kNoPdf;

  // set all non-center phones to zero (boundary)
  for (int32 i = 1; i < event_type_in.size(); ++i) {
    if (i == P + 1)
      continue;

    (*event_type_out)[i].second = 0;
  }

  return;
}


bool GeneralizeEventType(const EventType &event_type_in, EventType *event_type_out, int32 P, bool left) {
  KALDI_ASSERT(event_type_out != NULL);

  event_type_out->clear();
  event_type_out->insert(event_type_out->begin(), event_type_in.begin(), event_type_in.end());

  if (left) {
    for (int32 i = 1; i < P + 1; ++i) {
      if ((*event_type_out)[i].second > 0) {
        (*event_type_out)[i].second = 0;
        return true;
      }
    }
  } else {
    for (int32 i = event_type_in.size() - 1; i > P + 1; --i) {
      if ((*event_type_out)[i].second > 0) {
        (*event_type_out)[i].second = 0;
        return true;
      }
    }
  }

  // if we get to here, the event type is fully generalized;  see if we can still
  // go to the root event type
  if (event_type_in[0].second != kNoPdf) {
    (*event_type_out)[0].second = kNoPdf;
    return true;
  }

  return false;
}


int32 EventTypeBalance(const EventType &event_type, int32 P) {
  KALDI_ASSERT(event_type.size() > 0);
  KALDI_ASSERT(event_type[0].first == kPdfClass);

  int32 bal = 0;
  for (int32 i = 1; i < P + 1; ++i) {
    if (event_type[i].second != 0)
      bal -= 1;
  }

  for (int32 i = P + 2; i < event_type.size(); ++i) {
    if (event_type[i].second != 0)
      bal += 1;
  }

  return bal;
}


int32 EventTypeContextSize(const EventType &event_type, int32 P) {
  KALDI_ASSERT(event_type.size() > 0);
  KALDI_ASSERT(event_type[0].first == kPdfClass);

  int32 ctx_size = 0;
  for (int32 i = 1; i < event_type.size(); ++i) {
    if (event_type[i].second > 0)
      ctx_size += 1;
  }

  return ctx_size;
}


bool TopoNodeComparison::operator() (const TopoNode *a, const TopoNode *b) const {
  KALDI_ASSERT(a != NULL);
  KALDI_ASSERT(b != NULL);

  int32 bal1 = EventTypeBalance(a->event_type_, P_);
  int32 bal2 = EventTypeBalance(b->event_type_, P_);

  // multiply by two, subtract one if it was < 0
  // this is a cheap implementation of a two-way sort, where we want the
  // resulting sort order to be ascending in terms of abs(bal), but if there
  // are two values with the same abs(bal), we want the negative one to go
  // first (i.e. more left context goes first)
  bal1 = 2 * abs(bal1) - (bal1 < 0 ? 1 : 0);
  bal2 = 2 * abs(bal2) - (bal2 < 0 ? 1 : 0);

  if (bal1 == bal2)
    return EventTypeContextSize(a->event_type_, P_) < EventTypeContextSize(b->event_type_, P_);
  else
    return bal1 < bal2;
}


} // end namespace kaldi.

