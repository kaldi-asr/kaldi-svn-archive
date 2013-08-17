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


#include "tree/topo-tree.h"

using namespace std;
using namespace kaldi;


int main(int argc, char **argv) {
  EventType e1, e2, e3, e4;

  e1.push_back(pair<EventKeyType,EventValueType>(kPdfClass, 0));
  e1.push_back(pair<EventKeyType,EventValueType>(0, 1));
  e1.push_back(pair<EventKeyType,EventValueType>(1, 2));
  e1.push_back(pair<EventKeyType,EventValueType>(2, 3));
  e1.push_back(pair<EventKeyType,EventValueType>(3, 4));
  e1.push_back(pair<EventKeyType,EventValueType>(4, 5));

  int32 P = 2;

  cout << "Original EventType:" << endl;
  WriteEventType(cout, false, e1);

  // test root event type, comparisons
  {
    cout << "RootEventType:" << endl;
    RootEventType(e1, &e2, P);
    WriteEventType(cout, false, e2);

    EventTypeComparison cmp1(e1, e2, P);
    EventTypeComparison cmp2(e2, e1, P);

    cout << "Compare(orig, root).IsGeneralization(): " << cmp1.IsGeneralization() << endl;
    cout << "Compare(orig, root).IsSpecialization(): " << cmp1.IsSpecialization() << endl;
    cout << "Compare(root, orig).IsGeneralization(): " << cmp2.IsGeneralization() << endl;
    cout << "Compare(root, orig).IsSpecialization(): " << cmp2.IsSpecialization() << endl;
  }

  // test generalisations (left, right), comparison
  {
    e2 = e1;
    for (int32 i = 0; i < 3; ++i) {
      cout << "generalize-left: " << i;
      bool ret = GeneralizeEventType(e2, &e3, P);
      cout << (ret ? " ...ok" : "...fail") << endl;

      WriteEventType(cout, false, e3);

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

    e4 = e2;

    e2 = e1;
    for (int32 i = 0; i < 3; ++i) {
      cout << "generalize-right: " << i;
      bool ret = GeneralizeEventType(e2, &e3, P, false);
      cout << (ret ? " ...ok" : "...fail") << endl;

      WriteEventType(cout, false, e3);

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

  }

  {
    cout << "Compare:" << endl;
    WriteEventType(cout, false, e2);
    WriteEventType(cout, false, e4);

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


  {
    cout << "Testing Tree functions" << endl;
    TopoTree tree(5, 2);
    tree.Print(cout);

    cout << "Inserting e1: " << EventTypeToString(e1, P) << endl;
    tree.Insert(new TopoNode(e1));
    tree.Print(cout);

    GeneralizeEventType(e1, &e2, P, true);
    cout << "Inserting e2: " << EventTypeToString(e2, P) << endl;
    tree.Insert(new TopoNode(e2));
    tree.Print(cout);
  }

  return 0;
}
