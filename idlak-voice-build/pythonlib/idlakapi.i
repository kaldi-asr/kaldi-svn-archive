// file: idlakapi.i
%module idlakapi
%{
// list here headers which api requires to wrap
// these includes are added to the wrap.cxx file
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "tree/context-dep.h"
#include <string>
#include <vector>
#include "idlakapi.h"
%}
// list here headers containing functions you wish to wrap
%include "idlakapi.h"
