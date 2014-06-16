// feat/distortion-functions.h

// Copyright 2014 Pegah Ghahremani

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

#ifndef KALDI_FEAT_DISTORTION_FUNCTIONS_H_
#define KALDI_FEAT_DISTORTION_FUNCTIONS_H_

#include "base/kaldi-common.h"
#include "util/common-utils.h"

namespace kaldi {
/// @addtogroup  feat 
/// @{

struct DeformationOptions {
  // DeformationOptions deform_opts;
  BaseFloat rot_angle_stddev; // Standard deviation for rotation angle to rotate image w.r.t its 
                              // center by rot_angle degree. image rotates clockwise with positive angle.
  BaseFloat shift_stddev;     // Standard deviation for shift distribution to shift image up and down or left and right.
  BaseFloat scale_stddev;     // Standard deviation for scale distribution to scale image in x and y direction
  BaseFloat elastic_stddev;   // the variance of gaussian, which convolve with random
                              // displacement field in elastic distortion
  BaseFloat elastic_scale_factor;  // The displacement field is multiplied by scaling factor to control the the deformation
  bool first_rot;             // if true, it first rotate the image and then scale it.
  bool apply_affine_trans;    // If true, displacement field such as simple rotation, scaling, or 
                              // shifting applies to image and all pixels of image rotate, shift or scale with same values.
  bool apply_elastic_deform;  // If true, the image deformations created by generating
                              // random displacement fields.
  bool apply_back_noise;      // If true, a random background inserted in the digit image.
                              // Each pixel value of the background generated uniformly.
  explicit DeformationOptions() :
    rot_angle_stddev(15),
    shift_stddev(0.2),
    scale_stddev(0.2),
    elastic_stddev(3),
    elastic_scale_factor(20),
    first_rot(true),
    apply_affine_trans(true),
    apply_elastic_deform(true),
    apply_back_noise(false)
    {}
  void Register(OptionsItf *po) {
    po->Register("rot-angle-stddev", &rot_angle_stddev,
                 "The standard deviation of rotation angle to generate rot_angle to rotate image  w.r.t. its center pixel," 
                  "it rotates clockwise with positive angle");
    po->Register("shift-stddev", &shift_stddev, 
                 "The standard deviation of shift distribution to shift image in 2 directions,"
                 " it moves upward with positive x_shift");
    po->Register("scale-stddev", &scale_stddev,
                 "The standard deviation of scale distribution to scale image by x_scale in x-direction");
    po->Register("elastic-stddev", &elastic_stddev,
                 " the standard deviation for gaussian, which convolve with random"
                 " displacement field in elastic distortion");
    po->Register("elastic-scale-factor", &elastic_scale_factor,
                 "The displacement field is multiplied by scaling factor"
                 " to control the the deformation");
    po->Register("first-rot", &first_rot, 
                 "if true, it first rotates the image and then scale it.");
    po->Register("apply-affine-trans", &apply_affine_trans,
                 "If true, displacement field such as simple rotation, scaling, or shifting"
                 " applies to image.");
    po->Register("apply-elastic-deform", &apply_elastic_deform,
                 "If true, the image deformations created by generating" 
                 "random displacement fields.");
    po->Register("apply-back-noise", &apply_back_noise,
                 " If true, a random background inserted in the digit image."
                 "Each pixel value of the background generated uniformly");
  }
};
/*
  Data variation using transformation has shown effectiveness in 
  speech and image recognition. Generating plausible transformation of 
  input data helps to model different variation of data and improve 
  recognition task. Data augmentation would be helpful when we have 
  limited amount of training data. Synthetized data are mostly generated
  using Affine distortion, elastic deformation , or adding noise 
  to image background.
*/
/*
  This function is responsible for genarating displacement field that
  used to generate new examples for vision task.
  Simple distortions includes  translations, rotations, and skewing, 
  that is generated by applying affine displacement fields to pixels 
  of image. 
  The elastic deformation is a random displacement field for each 
  pixel of image that convolves with Gaussians with variance sigma.
  
  This function is genrating new target location correspond to each 
  pixel of images using affine displacement field such as shifting,
  scaling and rotating of original image or elastic deformation and 
  random background.
  This computes new target location for every pixel w.r.t its original
  location. We can model image shift, and scaling and rotation of image
  using transformation matrix. Sythetised image is rotation and 
  scaling of original image w.r.t its center, and transformation 
  matrix is product of rotation and scaling matrix. 
  The rotation matrix for angle a is [cos(a) -sin(a); sin(a) cos(a)],
  and scaling matrix for scaling factor x_scale and y_scale is 
  [1/x_scale 0; 0 1/y_scale]. 
  Different order of ratation and scaling results in different 
  transformation matrix.
  we need to add a bias vector to transformation matrix to model image shift. 
  Transtormation matrix disp_mat is a lin. transformation and a bias-term. 
  For each coordinate that maps 
    x -> a11 * x + a12 * y + a13 , y -> a21 * x + a22 * y + a23,
    and disp_mat = [a11 a12 a13; a21 a22 a23]
  a13 and a23 are bias terms which models image shift in x and 
  y direction respectively.
  e.g if disp_mat = [1 0 1;0 1 0], new image is a shift of 
  original pixels by 1 to upward.
  if disp_mat = [cos(rot) -sin(rot) 0; sin(rot) cos(rot) 0], 
  it would rotate the image clockwise by rot deg. angle. 
  The order of scaling and rotation results in different trans matrix.
  The target location is computed as 
    (x_target, y_target) = disp_mat * (x,y,1) 
  Column of disp_field contain the target location of pixels of original 
  image. e.g. (i * image_width + j) col of disp_field is the new location for pixel(i,j) 
*/
void GetDistortionField(DeformationOptions deform_opts, 
                        int image_width, int image_height,
                        Matrix<BaseFloat> *disp_field);
/*
  This function computes the new graylevel for every pixels to gerenate 
  synthetized image. Since all transforamtion coeffs are non-integer value,
  we need to interpolate graylevel at new location. The new grayscale value 
  of every pixel corresponds to the grayscale level of its new target 
  location w.r.t original image. The new gray level is computed using 
  bilinear interpolation among the gray level of the vertices of  the 
  square that the new location belongs to it.
  That can be computed as a horizental interpolation followed by 
  vertical one.
  Assume the new target location for pixel (0,0) is (3.2,4.5). 
  The new gray level for pixel(0,0) is average of 
  linear horizental interpolation 
    (greylevel(3,4) + 0.2 * (greylevel(4,4)-greylevel(3,4)) ) and 
  linear vertical interpolation 
    ( greylevel(3,4) + 0.5 * (greylevel(3,5)-greylevel(3,4)) ).
*/
void ApplyDistortionField(DeformationOptions deform,
                    const Matrix<BaseFloat> disp_field,
                    const Matrix<BaseFloat> input_feats,
                    Matrix<BaseFloat> *output_feats);

/// @} End of "addtogroup feat" 
} // namespace kaldi

#endif  // KALDI_FEAT_DISTORTION_FUNCTIONS_H_
