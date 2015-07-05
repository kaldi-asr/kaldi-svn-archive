// nnet3/nnet-optimize.cc

// Copyright      2015  Johns Hopkins University (author: Daniel Povey)

// See ../../COPYING for clarification regarding multiple authors
//
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

#include "nnet3/nnet-optimize.h"

namespace kaldi {
namespace nnet3 {


void IdentifySubmatrixArgs(NnetComputation::Command *c,
                           std::vector<int32*> *submatrix_args) {
  submatrix_args->clear();
  switch (c->command_type) {
      case NnetComputation::kResizeMatrixZeroed:
      case NnetComputation::kResizeMatrixUndefined:
      case NnetComputation::kResizeMatrixEmpty:
        break;
    case NnetComputation::kPropagate:
      submatrix_args->push_back(&c->arg3);
      submatrix_args->push_back(&c->arg4);
      break;
    case NnetComputation::kStoreStats:
      submatrix_args->push_back(&c->arg2);
      break;
    case NnetComputation::kBackprop:
      submatrix_args->push_back(&c->arg4);
      submatrix_args->push_back(&c->arg5);
      submatrix_args->push_back(&c->arg6);
      submatrix_args->push_back(&c->arg7);      
      break;
    case NnetComputation::kMatrixCopy:
    case NnetComputation::kMatrixAdd:
    case NnetComputation::kAddRows:
    case NnetComputation::kCopyRows:
    case NnetComputation::kAddRowRanges:
      submatrix_args->push_back(&c->arg1);
      submatrix_args->push_back(&c->arg2);
      break;
    case NnetComputation::kAddRowsMulti:
    case NnetComputation::kCopyRowsMulti:
    case NnetComputation::kAddToRowsMulti:
    case NnetComputation::kCopyToRowsMulti:
      submatrix_args->push_back(&c->arg1);
      break;
    case NnetComputation::kNoOperation:
    case NnetComputation::kNoOperationMarker:
      break;
    default:
      KALDI_ERR << "Unknown command type.";
  }
}
    
void IdentifyMatrixArgs(NnetComputation::Command *c,
                        std::vector<int32*> *matrix_args) {
  matrix_args->clear();
  switch (c->command_type) {
    case NnetComputation::kResizeMatrixZeroed:
    case NnetComputation::kResizeMatrixUndefined:
    case NnetComputation::kResizeMatrixEmpty:
      matrix_args->push_back(&c->arg1);
      break;
    default:
      break;
  }
}

// We declare this class in the .cc file, we don't need to export it.
// It's used inside RemoveSomeMatrices.  matrices_to_remove must be
// sorted and uniq.
class ComputationRenumberer {
 public:
  ComputationRenumberer(NnetComputation *computation):
      computation_(computation) { }
  
  void Renumber() {
    SetUpMappings();
    RenumberCommands();
    RenumberMatrices();
    RenumberSubmatrices();
    RenumberIndexesMulti();
    RenumberDebugInfo();
    RenumberIo();
  }
 private:
  void SetUpMappings();
  void RenumberCommands();
  void RenumberMatrices();
  void RenumberSubmatrices();
  void RenumberIndexesMulti();
  void RenumberDebugInfo();
  void RenumberIo();

  struct SubMatrixHasher {
    SubMatrixHasher() { }
    size_t operator () (const NnetComputation::SubMatrixInfo &submat) const {
      // these numbers are arbitrarily chosen primes.
      return submat.matrix_index +
          19553 * submat.row_offset +
          29297 * submat.num_rows +
          42209 * submat.col_offset +
          56527 * submat.num_cols;
    }
  };
  
  /// creates a renumbering that removes the elements in "to_remove",
  /// e.g. if old_num_elements = 3 and to_remove = [1], would output
  /// the vector [ 0, -1, 1 ].
  static void CreateRenumbering(int32 old_num_elements,
                                const std::vector<int32> &to_remove,
                                std::vector<int32> *renumbering);

  std::vector<int32> matrices_to_remove_;
  NnetComputation *computation_;

  int32 num_matrices_orig_;
  int32 num_submatrices_orig_;
  int32 num_matrices_new_;
  int32 num_submatrices_new_;
  std::vector<int32> old_to_new_matrix_; // numbered by orig-matrix-index, gives
                                         // new-matrix-index.  -1 for removed
                                         // ones.
  std::vector<int32> old_to_new_submatrix_; // numbered by orig-submatrix-index,
                                            // gives new-submatrix-index.  -1
                                            // for removed ones.

};

//static
void ComputationRenumberer::CreateRenumbering(
    int32 old_num_elements,
    const std::vector<int32> &to_remove,
    std::vector<int32> *renumbering) {
  KALDI_ASSERT(IsSortedAndUniq(to_remove) && old_num_elements > 0);
  renumbering->clear();
  renumbering->resize(old_num_elements, 0);
  int32 num_remove = to_remove.size();
  for (int32 r = 0; r < num_remove; r++) {
    int32 this_remove = to_remove[r];
    // the "> 0" would be ">= 0" in a more generic context, but zero is
    // not valid in this particular application.
    KALDI_ASSERT(this_remove > 0 && this_remove < old_num_elements);
    (*renumbering)[this_remove] = -1;
  }
  int32 cur_number = 0;
  for (int32 i = 0; i < old_num_elements; i++) {
    if ((*renumbering)[i] != -1)
      (*renumbering)[i] = cur_number++;
  }
  KALDI_ASSERT(cur_number == old_num_elements -
               static_cast<int32>(to_remove.size()));
}


void ComputationRenumberer::SetUpMappings() {
  KALDI_ASSERT(matrices_to_remove_.empty());
  num_matrices_orig_ = computation_->matrices.size();
  num_submatrices_orig_ = computation_->submatrices.size();

  // list of submats per matrix.  
  std::vector<std::vector<int32> > submatrix_lists; 
  ComputeSubmatLists(*computation_, &submatrix_lists);

  for (int32 m = 1; m < num_matrices_orig_; m++)
    if (submatrix_lists[m].empty())
      matrices_to_remove_.push_back(m);

  CreateRenumbering(num_matrices_orig_, matrices_to_remove_,
                    &old_to_new_matrix_);
  
  unordered_map<NnetComputation::SubMatrixInfo, int32,
                SubMatrixHasher> submat_map;
  int32 cur_index = 1;
  // the old_to_new_submatrix_ map will remove duplicates.
  old_to_new_submatrix_.resize(num_submatrices_orig_);
  old_to_new_submatrix_[0] = 0;
  for (int32 s = 1; s < num_submatrices_orig_; s++) {
    const NnetComputation::SubMatrixInfo &info =
        computation_->submatrices[s];
    if (submat_map.count(info) > 0)
      old_to_new_submatrix_[s] = submat_map[info];
    else
      old_to_new_submatrix_[s] = (submat_map[info] = cur_index++);
  }        
  num_submatrices_new_ = cur_index;
}

void ComputationRenumberer::RenumberCommands() {
  // renumbers matrices and submatrices in commands.
  const int32 num_matrices_old = num_matrices_orig_,
      num_submatrices_old = num_submatrices_orig_;
  int32 num_commands = computation_->commands.size();
  for (int32 command_index = 0; command_index < num_commands; command_index++) {
    NnetComputation::Command &c = computation_->commands[command_index];
    {
      std::vector<int32*> submatrix_args;
      IdentifySubmatrixArgs(&c, &submatrix_args);
      std::vector<int32*>::const_iterator iter = submatrix_args.begin(),
          end = submatrix_args.end();
      for (; iter != end; ++iter) {
        int32 *submatrix_arg = *iter;
        int32 submatrix_index = *submatrix_arg,
            new_submatrix_index = old_to_new_submatrix_[submatrix_index];
        KALDI_ASSERT(submatrix_index >= 0 &&
                     submatrix_index < num_submatrices_old);
        // renumber the argument of the command.
        *submatrix_arg = new_submatrix_index;
      }
    }
    {
      std::vector<int32*> matrix_args;
      IdentifyMatrixArgs(&c, &matrix_args);
      std::vector<int32*>::const_iterator iter = matrix_args.begin(),
          end = matrix_args.end();
      for (; iter != end; ++iter) {
        int32 *matrix_arg = *iter;
        int32 matrix_index = *matrix_arg,
            new_matrix_index = old_to_new_matrix_[matrix_index];
        KALDI_ASSERT(matrix_index >= 0 && matrix_index < num_matrices_old &&
                     new_matrix_index >= 0);
        // renumber the argument of the command.
        *matrix_arg = new_matrix_index;
      }
    }
  }
}

void ComputationRenumberer::RenumberMatrices() {
  std::vector<NnetComputation::MatrixInfo> new_matrices(num_matrices_new_);
  for (int32 m = 0; m < num_matrices_orig_; m++) {
    int32 m_new = old_to_new_matrix_[m];
    if (m_new != -1)
      new_matrices[m_new] = computation_->matrices[m];
  }
  computation_->matrices = new_matrices;
}



void ComputationRenumberer::RenumberSubmatrices() {
  std::vector<NnetComputation::SubMatrixInfo> new_submatrices(
      num_submatrices_new_);
  for (int32 s = 0; s < num_submatrices_orig_; s++) {
    int32 s_new = old_to_new_submatrix_[s];
    if (s_new != -1) {
      NnetComputation::SubMatrixInfo &dest = new_submatrices[s_new];
      dest = computation_->submatrices[s];
      dest.matrix_index = old_to_new_matrix_[dest.matrix_index];
      KALDI_ASSERT(dest.matrix_index >= 0);
    }
  }
  computation_->submatrices = new_submatrices;
}

void ComputationRenumberer::RenumberIndexesMulti() {
  std::vector<std::vector<std::pair<int32,int32> > >::iterator
      iter = computation_->indexes_multi.begin(),
      end = computation_->indexes_multi.end();
  for (; iter != end; ++iter) {
    std::vector<std::pair<int32,int32> >::iterator
        iter2 = iter->begin(), end2 = iter->end();
    for (; iter2 != end2; ++iter2) {
      int32 &submatrix_index = iter2->first;
      if (submatrix_index > 0) {
        KALDI_ASSERT(submatrix_index < num_submatrices_orig_);
        submatrix_index = old_to_new_submatrix_[submatrix_index];
      }
    }
  }    
}

void ComputationRenumberer::RenumberDebugInfo() {
  if (computation_->matrix_debug_info.empty())
    return;
  KALDI_ASSERT(static_cast<int32>(computation_->matrix_debug_info.size()) ==
               num_matrices_orig_);
  // we arbitrarily keep the matrix debug info from the earliest numbered matrix
  // when constructing the new debug info.  The info may sometimes differ and
  // we'll just choose to identify the matrix with one or other of the nodes.
  // this information is only consumed by human readers anyway, while debugging.
  std::vector<NnetComputation::MatrixDebugInfo> matrix_debug_info(
      num_matrices_new_);
  for (int32 m = 0; m < num_matrices_orig_; m++) {
    int32 m_new = old_to_new_matrix_[m];
    if (m_new != -1 && matrix_debug_info[m_new].indexes.empty())
      matrix_debug_info[m_new] = computation_->matrix_debug_info[m];
  }
  computation_->matrix_debug_info = matrix_debug_info;
}

void ComputationRenumberer::RenumberIo() {
  unordered_map<int32, std::pair<int32, int32> >::iterator
      iter = computation_->input_output_info.begin(),
      end = computation_->input_output_info.end();
  for (; iter != end; ++iter) {
    int32 &value_matrix_index = iter->second.first,
        &deriv_matrix_index = iter->second.second;
    KALDI_ASSERT(value_matrix_index > 0 && value_matrix_index <
                 num_matrices_orig_);
    value_matrix_index = old_to_new_matrix_[value_matrix_index];
    KALDI_ASSERT(value_matrix_index != -1);
    if (deriv_matrix_index != -1) {
      KALDI_ASSERT(deriv_matrix_index > 0 && deriv_matrix_index <
                   num_matrices_orig_);
      deriv_matrix_index = old_to_new_matrix_[deriv_matrix_index];
      KALDI_ASSERT(deriv_matrix_index != -1);
    }
  }
}


/// This function detects matrices that have no submatrices corresponding to them (due,
/// to changes made in other optimization code), and removes them from the computation.
/// It also renumbers the submatrix indexes to remove duplicates.
void RemoveOrphanMatrices(NnetComputation *computation) {
  ComputationRenumberer renumberer(computation);
  renumberer.Renumber();
}

/// Wherever matrix orig_matrix_index appears in the output of the network
/// (i.e. in computation->input_output_info), replaces it with new_matrix_index.
/// Returns true if it did replace it.
bool ReplaceInOutput(
    const Nnet &nnet,
    int32 orig_matrix_index, int32 new_matrix_index,
    NnetComputation *computation) {
  bool ans = false;
  int32 num_matrices = computation->matrices.size();
  KALDI_ASSERT(orig_matrix_index > 0 && orig_matrix_index < num_matrices &&
               new_matrix_index > 0 && new_matrix_index < num_matrices);
  unordered_map<int32, std::pair<int32, int32> >::iterator
      iter = computation->input_output_info.begin(),
      end = computation->input_output_info.end();
  for (; iter != end; ++iter) {
    int32 network_node = iter->first,
        &value_matrix_index = iter->second.first,
        &deriv_matrix_index = iter->second.second;
    if (nnet.IsOutputNode(network_node)) {
      // value_matrix_index would be an output of the computation.
      if (value_matrix_index == orig_matrix_index) {
        value_matrix_index = new_matrix_index;
        ans = true;
      }
    } else {
      // deriv_matrix_index would be an output of the computation.
      if (deriv_matrix_index == orig_matrix_index) {
        deriv_matrix_index = new_matrix_index;
        ans = true;
      }
    }
  }
  return ans;
}

VariableMergingOptimizer::VariableMergingOptimizer(
    const NnetOptimizeConfig &config,
    const Nnet &nnet,
    const ComputationRequest &request,
    NnetComputation *computation):
    config_(config), nnet_(nnet), request_(request),
    computation_(computation), variables_(*computation) { }

bool VariableMergingOptimizer::MergeVariables() {
  bool merged = false;
  Initialize();
  int32 num_commands = computation_->commands.size();
  for (int32 command_index = 0; command_index < num_commands;
       command_index++) {
    const NnetComputation::Command &c =
        computation_->commands[command_index];
    int32 s1 = -1, s2 = -1;
    if (c.command_type == NnetComputation::kMatrixCopy) {
      s2 = c.arg1;  // s2 is the written-to matrix.
      s1 = c.arg2;
    } else if (c.command_type == NnetComputation::kPropagate) {
      const Component *component = nnet_.GetComponent(c.arg1);
      if (component->Properties() & kPropagateInPlace) {
        s1 = c.arg3;
        s2 = c.arg4;  // s2 is the written-to matrix.
      }
    } else if (c.command_type == NnetComputation::kBackprop) {
      const Component *component = nnet_.GetComponent(c.arg2);
      if (component->Properties() & kBackpropInPlace) {
        s1 = c.arg6;
        s2 = c.arg7;  // s2 is the written-to matrix.
        if (s1 == c.arg4 || s2 == c.arg4 || s1 == c.arg5 || s2 == c.arg5) {
          // we don't think this should ever happen, but just out of an
          // abundance of caution: if either of these submatrix indexes are the
          // input-value or output-value args to Backprop, don't do the optimization.
          s1 = -1;
          s2 = -1;
        }
      }
    }
    if (s1 != -1 && IsCandidate(command_index, s1, s2)) {
      merged = true;
      DoMerge(command_index, s1, s2);
    }
  }
  if (merged) {
    RemoveOrphanMatrices(computation_);
  }
  return merged;
}

void VariableMergingOptimizer::DoMerge(int32 command_index,
                                       int32 s1, int32 s2) {
  NnetComputation::Command &c = computation_->commands[command_index];
  int32 m1 = computation_->submatrices[s1].matrix_index,
      m2 = computation_->submatrices[s2].matrix_index;
  KALDI_ASSERT(m1 != m2 && m1 > 0 && m2 > 0);
  { // renumber submatrices for m2 to refer to m1 instead.
    std::vector<int32>::const_iterator iter = submatrix_lists_[m2].begin(),
        end = submatrix_lists_[m2].end();
    for (; iter != end; ++iter) {
      int32 submatrix_index = *iter;
      KALDI_ASSERT(computation_->submatrices[submatrix_index].matrix_index==m2);
      computation_->submatrices[submatrix_index].matrix_index = m1;
    }
  }
  //  - If m2 was an output, replace it as an output with m1.  
  bool replaced = ReplaceInOutput(nnet_, m2, m1, computation_);
  KALDI_ASSERT(replaced == matrix_accesses_[m2].is_output);
  //  - If it was case (a), replace the assignment command with a no-op.  
  if (c.command_type == NnetComputation::kMatrixCopy) {
    // remove the command.
    c.command_type = NnetComputation::kNoOperation;
    c.arg1 = -1;
    c.arg2 = -1;
  }
  //  - Modify the command that deallocates m2 (if it exists) to make it
  //    deallocate m1 instead.
  if (matrix_accesses_[m2].destroy_command != -1) {
    NnetComputation::Command &destroy_command = computation_->commands[
        matrix_accesses_[m2].destroy_command];
    KALDI_ASSERT(destroy_command.command_type ==
                 NnetComputation::kResizeMatrixEmpty &&
                 destroy_command.arg1 == m2);
    destroy_command.arg1 = m1;
  }
  // Remove the original command that deallocated m1 (which should exist).
  KALDI_ASSERT(matrix_accesses_[m1].destroy_command != -1);
  {
    NnetComputation::Command &destroy_command = computation_->commands[
        matrix_accesses_[m1].destroy_command];
    KALDI_ASSERT(destroy_command.command_type ==
                 NnetComputation::kResizeMatrixEmpty &&
                 destroy_command.arg1 == m1);
    destroy_command.command_type = NnetComputation::kNoOperation;
    destroy_command.arg1 = -1;
  }
  // Prevent further optimizations touching m1 or m2 (we can
  // try again in a later round of optimization, with a new
  // instance of this class).
  matrix_already_optimized_[m1] = true;
  matrix_already_optimized_[m2] = true;  
}

// see comment by declaration of this function in nnet-optimize.h.
bool VariableMergingOptimizer::IsCandidate(int32 command_index,
                                           int32 s1, int32 s2) const {
  if (s1 == s2) return false;
  if (!computation_->IsWholeMatrix(s1) ||
      !computation_->IsWholeMatrix(s2)) return false;
  int32 m1 = computation_->submatrices[s1].matrix_index,
      m2 = computation_->submatrices[s2].matrix_index;
  if (matrix_already_optimized_[m1] || matrix_already_optimized_[m2])
    return false;
  const MatrixAccesses &m1_access = matrix_accesses_[m1],
      &m2_access = matrix_accesses_[m2];
  if (m1_access.is_output) return false;
  if (m2_access.is_input) return false;
  // the following check would probably indicate a coding error- this
  // function should never be called if those things are empty.
  if (m1_access.access_commands.empty() || m2_access.access_commands.empty())
    KALDI_ERR << "Matrices never accessed [confusing].";
  // m1 is accessed after command "command_index".
  if (m1_access.access_commands.back() > command_index)
    return false;
  // m2 is accessed before command "command_index" (but not counting
  // zeroing in initialization.)

  int32 m2_first_command = m2_access.access_commands.front();
  if (m2_first_command != m2_access.initialize_command &&
      m2_first_command < command_index) {
    // m2 accessed before that command.
    return false;
  }
  if (m2_first_command == m2_access.initialize_command) {
    // first access just initializes it ->must consider second
    KALDI_ASSERT(m2_access.access_commands.size() > 1);
    int32 m2_second_command = m2_access.access_commands[1];
    if (m2_second_command < command_index) {
      // m2 accessed before command_index.
      return false;
    }
  }
  return true;
}


void VariableMergingOptimizer::Initialize() {
  KALDI_ASSERT(matrix_already_optimized_.empty() &&
               "You cannot call Merge twice on the same object.");
  ComputeCommandAttributes(nnet_, *computation_, variables_,
                           &attributes_);
  ComputeVariableAccesses(attributes_, &variable_accesses_);
  ComputeMatrixAccesses(nnet_, *computation_, variables_,
                        attributes_, &matrix_accesses_);
  ComputeSubmatLists(*computation_, &submatrix_lists_);
  matrix_already_optimized_.resize(computation_->matrices.size(), false);
}


} // namespace nnet3
} // namespace kaldi
