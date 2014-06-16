// feat/distortion-function-test.cc 
 
 // Copyright    2014  Pegah Ghahremani 
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
#include <iostream>
#include "feat/distortion-functions.h"

using namespace kaldi;

static void UnitTestSimpleDistortion() {
  // shift the image upward
  DeformationOptions deform_opts;
  deform_opts.shift_stddev = 5;
  deform_opts.rot_angle_stddev = 0;
  Matrix<BaseFloat> input_image, output_image;
  std::string image_file = "test/test_image.txt";
  std::ifstream is(image_file.c_str());
  input_image.Read(is, false);
  int image_height = input_image.NumRows();
  int image_width = input_image.NumCols();
  Matrix<BaseFloat> disp_field(2,image_height * image_width);
  GetDistortionField(deform_opts, image_width, image_height, &disp_field);
  ApplyDistortionField(deform_opts, disp_field, input_image, &output_image);
  std::string outfile = "test/shifted_image.txt"; 
  std::ofstream os(outfile.c_str());
  output_image.Write(os, false);
  
  // rotate the image
  deform_opts.shift_stddev = 0;
  deform_opts.rot_angle_stddev = 10;
  GetDistortionField(deform_opts, image_width, image_height, &disp_field);
  ApplyDistortionField(deform_opts, disp_field, input_image, &output_image); 
  std::string outfile2 = "test/rotated_image.txt"; 
  std::ofstream os2(outfile2.c_str());
  output_image.Write(os2, false);

  // scale the image
  deform_opts.rot_angle_stddev = 0;
  deform_opts.scale_stddev = 0.2;
  GetDistortionField(deform_opts, image_width, image_height, &disp_field);
  ApplyDistortionField(deform_opts, disp_field, input_image, &output_image);   
  std::string outfile3 = "test/scaled_image.txt"; 
  std::ofstream os3(outfile3.c_str());
  output_image.Write(os3, false);

  // rotate and scale the image
  deform_opts.rot_angle_stddev = 15;
  deform_opts.scale_stddev = 0.2;
  GetDistortionField(deform_opts, image_width, image_height, &disp_field);
  ApplyDistortionField(deform_opts, disp_field, input_image, &output_image);   
  std::string disp = "test/rot_scaled_dist.txt";
  std::ofstream os4(disp.c_str());
  disp_field.Write(os4, false);
  std::string outfile4 = "test/rot_scaled_image.txt"; 
  std::ofstream os5(outfile3.c_str());
  output_image.Write(os5, false);
}

static void UnitTestElasticDeformation() {
  DeformationOptions deform_opts;
  deform_opts.apply_affine_trans = false; 
  deform_opts.apply_elastic_deform = true;
  deform_opts.elastic_stddev = 3;
  deform_opts.elastic_scale_factor = 32;
  Matrix<BaseFloat> input_image, output_image;
  std::string image_file = "test/test_image.txt";
  std::ifstream is(image_file.c_str());
  input_image.Read(is, false);
  int image_height = input_image.NumRows();
  int image_width = input_image.NumCols(); 
  Matrix<BaseFloat> disp_field(2, image_height * image_width);
  GetDistortionField(deform_opts, image_width, image_height, &disp_field);
  ApplyDistortionField(deform_opts, disp_field, input_image, &output_image);
  std::string disp = "test/dist.txt";
  std::ofstream os1(disp.c_str());
  disp_field.Write(os1, false);
  std::string outfile = "test/elastic_deformed_image.txt"; 
  std::ofstream os2(outfile.c_str());
  output_image.Write(os2, false);
}

static void UnitTestSynthetizeImage() {
  UnitTestSimpleDistortion();
  UnitTestElasticDeformation();
}

int main() {
  try {
    UnitTestSynthetizeImage();
    KALDI_LOG << "Tests succeeded.\n";
    return 0;
  } catch(const std::exception &e) {
    KALDI_ERR << e.what();
    return 1;
  }
}
