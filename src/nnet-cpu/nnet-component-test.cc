// nnet-cpu/net-component-test.cc

// Copyright 2012  Johns Hopkins University (author:  Daniel Povey)

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

#include "nnet-cpu/nnet-component.h"
#include "util/common-utils.h"

namespace kaldi {


void UnitTestGenericComponentInternal(const Component &component) {
  int32 input_dim = component.InputDim(),
       output_dim = component.OutputDim();

  Vector<BaseFloat> objf_vec(output_dim); // objective function is linear function of output.
  objf_vec.SetRandn(); // set to Gaussian noise.
  
  int32 num_egs = 10 + rand() % 5;
  Matrix<BaseFloat> input(num_egs, input_dim),
      output(num_egs, output_dim);
  input.SetRandn();
  
  component.Propagate(input, &output);
  {
    bool binary = (rand() % 2 == 0);
    Output ko("tmpf", binary);
    component.Write(ko.Stream(), binary);
  }
  Component *component_copy;
  {
    bool binary_in;
    Input ki("tmpf", &binary_in);
    component_copy = Component::ReadNew(ki.Stream(), binary_in);
  }
  
  { // Test backward derivative is correct.
    Vector<BaseFloat> output_objfs(num_egs);
    output_objfs.AddMatVec(1.0, output, kNoTrans, objf_vec);
    BaseFloat objf = output_objfs.Sum();

    Matrix<BaseFloat> output_deriv(output.NumRows(), output.NumCols());
    for (int32 i = 0; i < output_deriv.NumRows(); i++)
      output_deriv.Row(i).CopyFromVec(objf_vec);

    Matrix<BaseFloat> input_deriv(input.NumRows(), input.NumCols());
    
    component_copy->Backprop(input, output, output_deriv, NULL, &input_deriv);

    int32 num_ok = 0, num_bad = 0, num_tries = 7;
    KALDI_LOG << "Comparing feature gradients " << num_tries << " times.";
    for (int32 i = 0; i < num_tries; i++) {
      Matrix<BaseFloat> perturbed_input(input.NumRows(), input.NumCols());
      perturbed_input.SetRandn();
      perturbed_input.Scale(1.0e-04); // scale by a small amount so it's like a delta.
      BaseFloat predicted_difference = TraceMatMat(perturbed_input,
                                                   input_deriv, kTrans);
      perturbed_input.AddMat(1.0, input); // now it's the input + a delta.
      { // Compute objf with perturbed input and make sure it matches prediction.
        Matrix<BaseFloat> perturbed_output(output.NumRows(), output.NumCols());
        component.Propagate(perturbed_input, &perturbed_output);
        Vector<BaseFloat> perturbed_output_objfs(num_egs);
        perturbed_output_objfs.AddMatVec(1.0, perturbed_output, kNoTrans,
                                         objf_vec);
        BaseFloat perturbed_objf = perturbed_output_objfs.Sum(),
             observed_difference = perturbed_objf - objf;
        KALDI_LOG << "Input gradients: comparing " << predicted_difference
                  << " and " << observed_difference;
        if (fabs(predicted_difference - observed_difference) >
            0.1 * fabs((predicted_difference + observed_difference)/2)) {
          KALDI_WARN << "Bad difference!";
          num_bad++;
        } else {
          num_ok++;
        }
      }
    }
    KALDI_LOG << "Succeeded for " << num_ok << " out of " << num_tries
              << " tries.";
    KALDI_ASSERT(num_ok > num_bad);
  }

  UpdatableComponent *ucomponent =
      dynamic_cast<UpdatableComponent*>(component_copy);

  if (ucomponent != NULL) { // Test parameter derivative is correct.

    int32 num_ok = 0, num_bad = 0, num_tries = 5;
    KALDI_LOG << "Comparing model gradients " << num_tries << " times.";
    for (int32 i = 0; i < num_tries; i++) {    
      UpdatableComponent *perturbed_ucomponent =
          dynamic_cast<UpdatableComponent*>(ucomponent->Copy()),
          *gradient_ucomponent =
          dynamic_cast<UpdatableComponent*>(ucomponent->Copy());
      KALDI_ASSERT(perturbed_ucomponent != NULL);
      gradient_ucomponent->SetZero(true); // set params to zero and treat as gradient.
      BaseFloat perturb_stddev = 1.0e-05;
      perturbed_ucomponent->PerturbParams(perturb_stddev);

      Vector<BaseFloat> output_objfs(num_egs);
      output_objfs.AddMatVec(1.0, output, kNoTrans, objf_vec, 0.0);
      BaseFloat objf = output_objfs.Sum();

      Matrix<BaseFloat> output_deriv(output.NumRows(), output.NumCols());
      for (int32 i = 0; i < output_deriv.NumRows(); i++)
        output_deriv.Row(i).CopyFromVec(objf_vec);
      Matrix<BaseFloat> input_deriv(input.NumRows(), input.NumCols());

      // This will compute the parameter gradient.
      ucomponent->Backprop(input, output, output_deriv, gradient_ucomponent, &input_deriv);

      // Now compute the perturbed objf.
      BaseFloat objf_perturbed;
      {
        Matrix<BaseFloat> output_perturbed(num_egs, output_dim);
        perturbed_ucomponent->Propagate(input, &output_perturbed);
        Vector<BaseFloat> output_objfs_perturbed(num_egs);
        output_objfs_perturbed.AddMatVec(1.0, output_perturbed,
                                         kNoTrans, objf_vec, 0.0);
        objf_perturbed = output_objfs_perturbed.Sum();
      }

      BaseFloat delta_objf_observed = objf_perturbed - objf,
          delta_objf_predicted = (perturbed_ucomponent->DotProduct(*gradient_ucomponent) -
                                  ucomponent->DotProduct(*gradient_ucomponent));

      KALDI_LOG << "Model gradients: comparing " << delta_objf_observed
                << " and " << delta_objf_predicted;
      if (fabs(delta_objf_predicted - delta_objf_observed) >
          0.1 * fabs((delta_objf_predicted + delta_objf_observed)/2)) {
        KALDI_WARN << "Bad difference!";
        num_bad++;
      } else {
        num_ok++;
      }
      delete perturbed_ucomponent;
      delete gradient_ucomponent;
    }
    KALDI_ASSERT(num_ok > num_bad);
  }
  delete component_copy; // No longer needed.
}


void UnitTestSigmoidComponent() {
  // We're testing that the gradients are computed correctly:
  // the input gradients and the model gradients.
  
  int32 input_dim = 10 + rand() % 50;
  {
    SigmoidComponent sigmoid_component(input_dim);
    UnitTestGenericComponentInternal(sigmoid_component);
  }
  {
    SigmoidComponent sigmoid_component;
    sigmoid_component.InitFromString("15");
    UnitTestGenericComponentInternal(sigmoid_component);
  }
}

template<class T>
void UnitTestGenericComponent() { // works if it has an initializer from int,
  // e.g. tanh, sigmoid.
  
  // We're testing that the gradients are computed correctly:
  // the input gradients and the model gradients.
  
  int32 input_dim = 10 + rand() % 50;
  {
    T component(input_dim);
    UnitTestGenericComponentInternal(component);
  }
  {
    T component;
    component.InitFromString("15");
    UnitTestGenericComponentInternal(component);
  }
}

void UnitTestAffineComponent() {
  BaseFloat learning_rate = 0.01, l2_penalty = 0.001,
             param_stddev = 0.1;
  int32 input_dim = 5 + rand() % 10, output_dim = 5 + rand() % 10;

  {
    AffineComponent component;
    component.Init(learning_rate, l2_penalty, input_dim, output_dim,
                   param_stddev);
    UnitTestGenericComponentInternal(component);
  }
  {
    const char *str = "0.01 0.001 10 15 0.1";
    AffineComponent component;
    component.InitFromString(str);
    UnitTestGenericComponentInternal(component);
  }
}

void UnitTestBlockAffineComponent() {
  BaseFloat learning_rate = 0.01, l2_penalty = 0.001,
             param_stddev = 0.1;
  int32 num_blocks = 1 + rand() % 3,
         input_dim = num_blocks * (2 + rand() % 4),
        output_dim = num_blocks * (2 + rand() % 4);
  
  {
    BlockAffineComponent component;
    component.Init(learning_rate, l2_penalty, input_dim, output_dim,
                   param_stddev, num_blocks);
    UnitTestGenericComponentInternal(component);
  }
  {
    const char *str = "0.01 0.001 10 15 0.1 5";
    BlockAffineComponent component;
    component.InitFromString(str);
    UnitTestGenericComponentInternal(component);
  }
}

void UnitTestMixtureProbComponent() {
  BaseFloat learning_rate = 0.01, l2_penalty = 0.001,
      diag_element = 0.8;
  std::vector<int32> sizes;
  int32 num_sizes = 1 + rand() % 5; // allow 
  for (int32 i = 0; i < num_sizes; i++)
    sizes.push_back(1 + rand() % 5);
  
  
  {
    MixtureProbComponent component;
    component.Init(learning_rate, l2_penalty, diag_element, sizes);
    UnitTestGenericComponentInternal(component);
  }
  {
    const char *str = "0.01 0.001 0.9 3 4 5";
    MixtureProbComponent component;
    component.InitFromString(str);
    UnitTestGenericComponentInternal(component);
  }
}


/*
void UnitTestGenericComponentInternal(
    ComponentGenericLayer &test_layer,
                                  GenericLayer &gradient,
                                  int32 input_dim,
                                  int32 output_dim) {
  Vector<BaseFloat> objf_vec(output_dim); // objective function is linear function of output.
  objf_vec.SetRandn(); // set to Gaussian noise.
  
  int32 num_egs = 10 + rand() % 5;
  Matrix<BaseFloat> input(num_egs, input_dim),
      output(num_egs, output_dim);
  input.SetRandn();
  
  test_layer.Forward(input, &output);
  { // Test backward derivative and model derivatives are correct.
    Vector<BaseFloat> output_objfs(num_egs);
    output_objfs.AddMatVec(1.0, output, kNoTrans, objf_vec);
    BaseFloat objf = output_objfs.Sum();

    Matrix<BaseFloat> output_deriv(output.NumRows(), output.NumCols());
    for (int32 i = 0; i < output_deriv.NumRows(); i++)
      output_deriv.Row(i).CopyFromVec(objf_vec);

    Matrix<BaseFloat> input_deriv(input.NumRows(), input.NumCols());
    
    test_layer.Backward(input, output, output_deriv, &input_deriv, &gradient);

    {
      Matrix<BaseFloat> perturbed_input(input.NumRows(), input.NumCols());
      perturbed_input.SetRandn();
      perturbed_input.Scale(1.0e-05); // scale by a small amount so it's like a delta.
      BaseFloat predicted_difference = TraceMatMat(perturbed_input,
                                                   input_deriv, kTrans);
      perturbed_input.AddMat(1.0, input);
      { // Compute objf with perturbed input and make sure it matches prediction.
        Matrix<BaseFloat> perturbed_output(output.NumRows(), output.NumCols());
        test_layer.Forward(perturbed_input, &perturbed_output);
        Vector<BaseFloat> perturbed_output_objfs(num_egs);
        perturbed_output_objfs.AddMatVec(1.0, perturbed_output, kNoTrans,
                                         objf_vec);
        BaseFloat perturbed_objf = perturbed_output_objfs.Sum(),
            observed_difference = perturbed_objf - objf;
        KALDI_LOG << "Input gradients: comparing " << predicted_difference
                  << " and " << observed_difference;
        KALDI_ASSERT (fabs(predicted_difference - observed_difference) <
                      0.1 * fabs((predicted_difference + observed_difference)/2));
      }
    }
    {
      Matrix<BaseFloat> perturbed_params(output_dim, input_dim);
      perturbed_params.SetRandn();
      perturbed_params.Scale(1.0e-06);
      BaseFloat predicted_difference = TraceMatMat(gradient.Params(),
                                                   perturbed_params,
                                                   kTrans);
      perturbed_params.AddMat(1.0, test_layer.Params());
      test_layer.SetParams(perturbed_params);
      { // Compute objf with perturbed params and make sure it matches prediction.
        Matrix<BaseFloat> perturbed_output(output.NumRows(), output.NumCols());
        test_layer.Forward(input, &perturbed_output);
        Vector<BaseFloat> perturbed_output_objfs(num_egs);
        perturbed_output_objfs.AddMatVec(1.0, perturbed_output, kNoTrans,
                                         objf_vec);
        BaseFloat perturbed_objf = perturbed_output_objfs.Sum(),
            observed_difference = perturbed_objf - objf;
        KALDI_LOG << "Param gradients: comparing " << predicted_difference
                  << " and " << observed_difference;
        KALDI_ASSERT (fabs(predicted_difference - observed_difference) <
                      0.1 * fabs((predicted_difference + observed_difference)/2));
      }
    }
  }
}



void UnitTestSoftmaxLayer() {
  // We're testing that the gradients are computed correctly:
  // the input gradients and the model gradients.
  
  int32 input_dim = 10 + rand() % 50, output_dim = 10 + rand() % 50;
  BaseFloat learning_rate = 0.2; // arbitrary.
  
  SoftmaxLayer test_layer(input_dim, output_dim, learning_rate);
  {
    Matrix<BaseFloat> temp(output_dim, input_dim);
    temp.SetRandn();
    temp.Scale(0.1);
    test_layer.SetParams(temp);
  }  

  SoftmaxLayer gradient(test_layer);
  gradient.SetZero();
  gradient.SetLearningRate(1.0);
  UnitTestGenericLayerInternal(test_layer, gradient, input_dim, output_dim);
}
void UnitTestLinearLayer() {
  // We're testing that the gradients are computed correctly:
  // the input gradients and the model gradients.
  
  int32 input_dim = 10 + rand() % 50, output_dim = input_dim;
  BaseFloat learning_rate = 0.2; // arbitrary.
  BaseFloat diag_element = 0.9;
  
  LinearLayer test_layer(input_dim, diag_element, learning_rate);

  {
    Matrix<BaseFloat> temp(output_dim, input_dim);
    temp.SetRandn();
    temp.Scale(0.1);
    test_layer.SetParams(temp);
  }  

  LinearLayer gradient(test_layer);
  gradient.SetZero();
  gradient.SetLearningRate(1.0);
  UnitTestGenericLayerInternal(test_layer, gradient, input_dim, output_dim);
}

*/

} // namespace kaldi

int main() {
  using namespace kaldi;
  for (int32 i = 0; i < 5; i++) {
    UnitTestGenericComponent<SigmoidComponent>();
    UnitTestGenericComponent<TanhComponent>();
    UnitTestGenericComponent<PermuteComponent>();
    UnitTestGenericComponent<SoftmaxComponent>();
  }
  UnitTestAffineComponent();
  UnitTestBlockAffineComponent();
  UnitTestMixtureProbComponent();
}
