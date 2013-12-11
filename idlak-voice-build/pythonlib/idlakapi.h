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

// Exposing context dependency functions to allow conversion to HTK style
// decision tree


typedef std::vector<kaldi::EventMap*> EventMapVector;

// method to manage sting output memory

std::string * IDLAK_string_new();

void IDLAK_string_delete(std::string * s);

const char * IDLAK_string_val(std::string * s);

// context dependency tree

kaldi::ContextDependency *
IDLAK_read_contextdependency_tree(const char * fname);

const kaldi::EventMap *
IDLAK_contextdependency_tree_root(kaldi::ContextDependency * ctx_dep);

void IDLAK_contextdependency_tree_delete(kaldi::ContextDependency * ctx_dep);

int IDLAK_contextdependency_tree_contextwidth(kaldi::ContextDependency * ctx_dep);

int IDLAK_contextdependency_tree_centralposition(kaldi::ContextDependency * ctx_dep);

// event map vector

EventMapVector * IDLAK_eventmapvector_new();

void  IDLAK_eventmapvector_delete(EventMapVector * eventmapvector);

// functions to access event map

void IDLAK_eventmap_getchildren(kaldi::EventMap * eventmap,
                                EventMapVector * eventmapvector);

int IDLAK_eventmapvector_size(EventMapVector * eventmapvector);

const kaldi::EventMap *
IDLAK_eventmapvector_at(EventMapVector * eventmapvector, int index);

int IDLAK_eventmap_key(const kaldi::EventMap * eventmap);

int IDLAK_eventmap_answer(const kaldi::EventMap * eventmap);

void IDLAK_eventmap_yesset(const kaldi::EventMap * eventmap, std::string * s);
