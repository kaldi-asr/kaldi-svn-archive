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

#include "util/kaldi-io.h"
#include "tree/topo-tree.h"

using namespace std;
using namespace kaldi;

EventType make_event(int32 pdf_class, int32 N, EventValueType *phoneseq) {
  EventType e;
  e.push_back(make_pair(kPdfClass, pdf_class));

  for (int32 i = 0; i < N; ++i)
    e.push_back(make_pair(i, phoneseq[i]));

  return e;
}

void TestRootEventType() {
  cout << "TestRootEventType:" << endl;

  int32 P = 2;

  EventValueType phoneseq[] = { 0, 1, 2, 3, 4 };
  EventType e1 = make_event(0, 5, phoneseq), e2;

  RootEventType(e1, &e2, 2);

  cout << "orig: " << EventTypeToString(e1, P) << endl;
  cout << "root: " << EventTypeToString(e2, P) << endl;

  EventTypeComparison cmp1(e1, e2, P);
  EventTypeComparison cmp2(e2, e1, P);

  cout << "Compare(orig, root).IsGeneralization(): " << cmp1.IsGeneralization() << endl;
  cout << "Compare(orig, root).IsSpecialization(): " << cmp1.IsSpecialization() << endl;
  cout << "Compare(root, orig).IsGeneralization(): " << cmp2.IsGeneralization() << endl;
  cout << "Compare(root, orig).IsSpecialization(): " << cmp2.IsSpecialization() << endl;

  KALDI_ASSERT(e2[0].second == kNoPdf);
}

void TestCompression() {
  cout << "TestCompression:" << endl;

  EventValueType phoneseqs[][5] = {
      { 0, 0, 2, 0, 0 }, //    /a/x
      { 0, 1, 2, 0, 0 }, //  xx/a/
      { 0, 0, 2, 1, 0 }, //   x/a/
      { 0, 1, 2, 1, 0 }, //   x/a/x
      { 1, 1, 2, 1, 1 }
  };

  int32 N = 5;

  EventType ev[] = {
      make_event(0, N, phoneseqs[0]),
      make_event(0, N, phoneseqs[1]),
      make_event(0, N, phoneseqs[2]),
      make_event(0, N, phoneseqs[3]),
      make_event(kNoPdf, N, phoneseqs[4])
  };

  for (int i = 0; i < 5; ++i) {
    EventType comp = CompressEventType(ev[i]);
    EventType infl = InflateEventType(comp, 5);

    cout << "orig: " << EventTypeToString(ev[i], 2) << endl;
    cout << "comp: " << EventTypeToString(comp) << endl;
    cout << "infl: " << EventTypeToString(infl, 2) << endl << endl;
  }
}

void TestGeneralization() {
  cout << "TestGeneralization:" << endl;

  int32 P = 2;

  EventValueType phoneseq[] = { 0, 1, 2, 3, 4 };
  EventType e1 = make_event(0, 5, phoneseq), e2, e3;
  e2 = e1;

  cout << "e1: " << EventTypeToString(e1, P) << endl;

  cout << "bal(e1): " << EventTypeBalance(e1, P) << endl;
  cout << "ctx(e1): " << EventTypeContextSize(e1, P) << endl;


  for (int32 i = 0; i < 3; ++i) {
    cout << "generalize-left: " << i;
    bool ret = GeneralizeEventType(e2, &e3, P);
    cout << (ret ? " ...ok" : "...fail") << " e3: " << EventTypeToString(e3, P) << endl;
    cout << "bal(e3): " << EventTypeBalance(e3, P) << endl;
    cout << "ctx(e3): " << EventTypeContextSize(e3, P) << endl;

    EventTypeComparison cmp1(e1, e3, P);
    EventTypeComparison cmp2(e3, e1, P);

    cout << "Compare(e1, e3).IsGeneralization(): " << cmp1.IsGeneralization() << endl;
    cout << "Compare(e1, e3).IsSpecialization(): " << cmp1.IsSpecialization() << endl;
    cout << "Compare(e3, e1).IsGeneralization(): " << cmp2.IsGeneralization() << endl;
    cout << "Compare(e3, e1).IsSpecialization(): " << cmp2.IsSpecialization() << endl;

    // test if they fit
    cout << "Compare(e1, e3).Fits(): " << cmp1.Fits() << endl;
    cout << "Compare(e3, e1).Fits(): " << cmp2.Fits() << endl;

    if (i == 2) {
      cout << cmp1.NumGeneralizations() << " " << cmp1.NumSpecializations() << " " << cmp1.NumIncompatibilities() << endl;
      cout << cmp2.NumGeneralizations() << " " << cmp2.NumSpecializations() << " " << cmp2.NumIncompatibilities() << endl;
    }

    e2 = e3;
  }

  EventType e4 = e2;

  e2 = e1;
  for (int32 i = 0; i < 3; ++i) {
    cout << "generalize-right: " << i;
    bool ret = GeneralizeEventType(e2, &e3, P, false);
    cout << (ret ? " ...ok" : "...fail") << " e3: " << EventTypeToString(e3, P) << endl;
    cout << "bal(e3): " << EventTypeBalance(e3, P) << endl;
    cout << "ctx(e3): " << EventTypeContextSize(e3, P) << endl;

    EventTypeComparison cmp1(e1, e3, P);
    EventTypeComparison cmp2(e3, e1, P);

    cout << "Compare(e1, e3).IsGeneralization(): " << cmp1.IsGeneralization() << endl;
    cout << "Compare(e1, e3).IsSpecialization(): " << cmp1.IsSpecialization() << endl;
    cout << "Compare(e3, e1).IsGeneralization(): " << cmp2.IsGeneralization() << endl;
    cout << "Compare(e3, e1).IsSpecialization(): " << cmp2.IsSpecialization() << endl;

    // test if they fit (should work as long as no root event type is involved
    cout << "Compare(e1, e3).Fits(): " << cmp1.Fits() << endl;
    cout << "Compare(e3, e1).Fits(): " << cmp2.Fits() << endl;

    e2 = e3;
  }

  cout << "Compare:" << endl;
  cout << "e2: " << EventTypeToString(e2, P) << endl;
  cout << "e4: " << EventTypeToString(e4, P) << endl;

  EventTypeComparison cmp1(e2, e4, P);
  EventTypeComparison cmp2(e4, e2, P);

  cout << "Compare(e2, e4).IsGeneralization(): " << cmp1.IsGeneralization() << endl;
  cout << "Compare(e2, e4).IsSpecialization(): " << cmp1.IsSpecialization() << endl;
  cout << "Compare(e4, e2).IsGeneralization(): " << cmp2.IsGeneralization() << endl;
  cout << "Compare(e4, e2).IsSpecialization(): " << cmp2.IsSpecialization() << endl;

  cout << "Compare(e2, e4).IsPartialGeneralization(): " << cmp1.IsPartialGeneralization() << endl;
  cout << "Compare(e2, e4).IsPartialSpecialization(): " << cmp1.IsPartialSpecialization() << endl;
  cout << "Compare(e4, e2).IsPartialGeneralization(): " << cmp2.IsPartialGeneralization() << endl;
  cout << "Compare(e4, e2).IsPartialSpecialization(): " << cmp2.IsPartialSpecialization() << endl;

}

void TestTree() {
  cout << "TestTree" << endl;

  int32 N = 5;
  int32 P = 2;

  EventValueType phoneseqs[][5] = {
      { 0, 0, 2, 3, 0 }, //    /a/x
      { 1, 1, 2, 0, 0 }, //  xx/a/
      { 0, 1, 2, 0, 0 }, //   x/a/
      { 0, 1, 2, 3, 0 }, //   x/a/x
      { 1, 1, 2, 3, 0 }
  };

  EventType ev[] = {
      make_event(0, N, phoneseqs[0]),
      make_event(0, N, phoneseqs[1]),
      make_event(0, N, phoneseqs[2]),
      make_event(0, N, phoneseqs[3]),
      make_event(0, N, phoneseqs[4])
  };

  TopoTree tree(N, P);
  tree.Print(cout);

  for (int i = 0; i < 5; ++i) {
    cout << endl << "Insert: " << EventTypeToString(ev[i], P) << endl;
    tree.Insert(ev[i]);
    tree.Print(cout);
  }

  cout << "Filling..." << endl;
  tree.Fill();
  tree.Print(cout);

  cout << "Removing: " << EventTypeToString(ev[2], P) << endl;
  tree.Remove(ev[2]);
  tree.Print(cout);

  cout << "Manifesting states..." << endl;
  for (int i = 0; i < 5; ++i)
    tree.Compute(ev[i])->SetPdfId(0);

  cout << "Populating" << endl;
  cout << "num_pdfs = " << tree.Populate() << endl;
  tree.Print(cout);

  vector<int32> phones; phones.push_back(0);
  vector<int32> pdf_classes; pdf_classes.push_back(0);

  vector<vector<pair<EventKeyType, EventValueType> > > pdf_info;
  tree.GetPdfInfo(phones, pdf_classes, &pdf_info);

  int32 i = 0;
  for (vector<vector<pair<EventKeyType, EventValueType> > >::iterator it = pdf_info.begin();
      it != pdf_info.end(); it++) {
    cout << i << ": ";
    for (vector<pair<EventKeyType, EventValueType> >::iterator ci = (*it).begin();
        ci != (*it).end(); ci++)
      cout << (*ci).first << " " << (*ci).second << " ";
    cout << endl;
    i++;
  }


  // Test IO
  cout << "Writing (ascii)..." << endl;
  WriteKaldiObject(tree, "test.tree", false);

  TopoTree loaded;
  cout << "Reading (ascii)..." << endl;
  ReadKaldiObject("test.tree", &loaded);

  loaded.Print(cout);

}

int main(int argc, char **argv) {

  TestRootEventType();
  TestCompression();
  TestGeneralization();
  TestTree();


  return 0;
}
