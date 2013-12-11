// pythonlib/idlakapi.h

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

// This is a header file for wrapping kaldi functionality for the idlak
// voice building process

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "tree/context-dep.h"
#include "tree/event-map.h"
#include "util/const-integer-set.h"
#include <string>
#include <iostream>
#include <sstream>
#include "idlakapi.h"

std::string * IDLAK_string_new() {
  return new std::string;
}

void IDLAK_string_delete(std::string * s) {
  if (s) delete s;
}

const char * IDLAK_string_val(std::string * s) {
  return s->c_str();
}

kaldi::ContextDependency *
IDLAK_read_contextdependency_tree(const char * fname) {
  kaldi::ContextDependency * ctx_dep;
  std::string s = fname;
  ctx_dep = new kaldi::ContextDependency();
  ReadKaldiObject(s, ctx_dep);
  return ctx_dep;
}

const kaldi::EventMap *
IDLAK_contextdependency_tree_root(kaldi::ContextDependency * ctx_dep) {
  return &(ctx_dep->ToPdfMap());
}

void IDLAK_contextdependency_tree_delete(kaldi::ContextDependency * ctx_dep) {
  delete ctx_dep;
}

int IDLAK_contextdependency_tree_contextwidth(kaldi::ContextDependency * ctx_dep) {
  return ctx_dep->ContextWidth();
}

int IDLAK_contextdependency_tree_centralposition(kaldi::ContextDependency * ctx_dep) {
  return ctx_dep->CentralPosition();
}

EventMapVector * IDLAK_eventmapvector_new() {
  EventMapVector * eventmapvector = new EventMapVector;
  return eventmapvector;
}

void  IDLAK_eventmapvector_delete(EventMapVector * eventmapvector) {
  if (eventmapvector) delete eventmapvector;
}

void IDLAK_eventmap_getchildren(kaldi::EventMap * eventmap,
                                EventMapVector * eventmapvector) {
  eventmap->GetChildren(eventmapvector);
}

int IDLAK_eventmapvector_size(EventMapVector * eventmapvector) {
  return eventmapvector->size();
}

const kaldi::EventMap *
IDLAK_eventmapvector_at(EventMapVector * eventmapvector, int index) {
  return (*eventmapvector)[index];
}

int IDLAK_eventmap_key(const kaldi::EventMap * eventmap) {
  return (int) eventmap->EventKey();
}

// Returns answer only for constant event type
int IDLAK_eventmap_answer(const kaldi::EventMap * eventmap) {
  kaldi::EventType const empty_event;
  kaldi::EventAnswerType answer;
  eventmap->Map(empty_event, &answer);
  return answer;
}

// fills the string with the integer values in the yes set
void IDLAK_eventmap_yesset(const kaldi::EventMap * eventmap, std::string * s) {
  std::ostringstream oss;
  s->clear();
  const kaldi::ConstIntegerSet<kaldi::EventValueType> * yes_set;
  kaldi::ConstIntegerSet<kaldi::EventValueType>::iterator iter;
  yes_set = eventmap->YesSet();
  if (yes_set) {
    for(iter = yes_set->begin(); iter != yes_set->end(); iter++) {
      oss << *iter << " ";
    }
    *s += oss.str();
  }
}
