// feat/distortion-functions.cc

// Copyright 2014  Pegah Ghahremani

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
#include <utility>
#include <vector>
#include "feat/distortion-functions.h"

namespace kaldi {

void GenerateGaussian(BaseFloat gauss_stddev, Tensor<BaseFloat> *gauss_tensor) {
  KALDI_ASSERT(gauss_tensor->NumIndexes() == 2);
  int32 width = gauss_tensor->Dim(0);
  int32 height = gauss_tensor->Dim(1);
  BaseFloat x_center = height / 2, y_center = width / 2;
  std::vector<int32> index(2);
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < height; j++) {
      index[0] = i;
      index[1] = j;
      BaseFloat distance = (pow(i + 1 - x_center, 2) + pow(j + 1 - y_center, 2)) / pow(gauss_stddev, 2);
      (*gauss_tensor)(index) = 1.0 / (2.0 * M_PI *  pow(gauss_stddev, 2)) * exp(-0.5 * distance);
    }
  }
}

void GetDistortionField(DeformationOptions deform_opts, 
                        int image_width, int image_height,
                        Matrix<BaseFloat> *disp_field) {
  disp_field->Resize(2, image_height * image_width);
  BaseFloat x_scale, y_scale, rot_angle, x_shift, y_shift, rot;
  if (deform_opts.apply_affine_trans == true) {
    // in affine displacement, all pixel of image is rotated, scaled and shifted with same values.
    // rotation, scaling and shifts are generated based on their mean and variance distribution.
    rot_angle = (RandGauss() * deform_opts.rot_angle_stddev);
    x_scale = 1 + (RandGauss() * deform_opts.scale_stddev);
    y_scale = 1 + (RandGauss() * deform_opts.scale_stddev);
    x_shift = (RandGauss() * deform_opts.shift_stddev);
    y_shift = (RandGauss() * deform_opts.shift_stddev);
    rot = M_2PI * rot_angle / 360.0;   
    Matrix<BaseFloat> disp_mat(2, 3), coord_mat(image_width * image_height, 3);
    disp_field->Resize(2, image_width * image_height);
    BaseFloat x_center = image_width / 2.0;
    BaseFloat y_center = image_height / 2.0;
    BaseFloat x_offset, y_offset;
    // since sythetised image is rotation and scaling of original 
    // image w.r.t its center, trans. matrix is product of rotation 
    // and scaling matrix. rotation matrix for angle rot is 
    // [cos(rot) -sin(rot); sin(rot) cos(rot)], and
    // scaling matrix for scaling factor x_scale and y_scale is 
    // [1/x_scale 0; 0 1/y_scale] and the product depend on product order.
    // The new coordinate for pixel (x,y) is 
    // (x_new, y_new) = (x_center, y_center) + transformation-matrix * (x-x_center, y-y_center)
    // So (x_new, y_new) = transformation-matrix * (x,y) + 
    //                    (I - transformation-matrix)*(x_center,y_center), 
    // where I is 2X2 identity matrix, and (x_center, y_center) is center of image.
    // Second term of (x_new, y_new) is called (x_offset, y_ofsset) 
    // which is the offset w.r.t transformation around center and  depends on image center.
    disp_mat(0, 0) = 1.0 / x_scale * cos(rot);
    disp_mat(0, 2) = x_shift;
    disp_mat(1, 1) = 1.0 / y_scale * cos(rot);
    disp_mat(1, 2) = y_shift;
      
    if (deform_opts.first_rot == true) {  // rotation(scale(x,y))
      disp_mat(0, 1) = -1.0 / x_scale * sin(rot);
      disp_mat(1, 0) = 1.0 / y_scale * sin(rot);

      x_offset = (1 - 1.0 /x_scale * cos(rot)) * x_center + 1.0 / x_scale * sin(rot) * y_center;
      y_offset = -1.0 / y_scale * sin(rot) * x_center + (1 - 1.0 / y_scale * cos(rot)) * y_center;
    } else {  // scale(rotation(x,y))
      disp_mat(0, 1) = -1.0 / y_scale * sin(rot);
      disp_mat(1, 0) = 1.0 / x_scale * sin(rot);

      x_offset = (1 - 1.0 /x_scale * cos(rot)) * x_center + 1.0 / y_scale * sin(rot) * y_center;
      y_offset = -1.0 / x_scale * sin(rot) * x_center + (1 - 1.0 / y_scale * cos(rot)) * y_center;
    }
    // To compute new coordinate correspond to each pixel, 
    // we should multiply transform matrix with inital coordinate.
    // coord_mat contains coordinate of all initial pixels in image, 
    // and each rows of coord_mat  contains coordinate of one pixel 
    // in image and bias term 1 as [x y 1].
    for (int i = 0; i < image_height; i++) {
      for (int j = 0; j < image_width; j++) { 
        coord_mat(i * image_width + j, 0) = i; 
        coord_mat(i * image_width + j, 1) = j; 
        coord_mat(i * image_width + j, 2) = 1.0;
      }
    }
    if ( std::abs(x_offset) < 10e-5) x_offset = 0;
    if ( std::abs(y_offset) < 10e-5) y_offset = 0;
    disp_field->AddMatMat(1.0, disp_mat, kNoTrans, coord_mat, kTrans, 0.0);
    disp_field->Row(0).Add(x_offset);
    disp_field->Row(1).Add(y_offset);
  } 
  if (deform_opts.apply_elastic_deform == true) {
    // In deforming image using elastic deformation, Gaussian random 
    // displacement field generated for each pixel of image. 
    // displacement in x and y direction is random number generated with 
    // a Gaussian distribution. Elastic deformation for pixel(i,j) is 
    // stored in col (i * image_width + j) of conv_elastic_disp. 
    typedef std::pair<int32, int32> DimsStrides;
    int32 gauss_dim = int(3 * deform_opts.elastic_stddev),
          x_field_dim = image_width + gauss_dim - 1,
          y_field_dim = image_height + gauss_dim - 1;
    Matrix<BaseFloat> x_elastic_field(1, x_field_dim * y_field_dim),
      y_elastic_field(1, x_field_dim * y_field_dim),
      gauss_field_mat(1, gauss_dim * gauss_dim),
      x_gauss_field_mat(1, image_width * image_height),
      y_gauss_field_mat(1, image_width * image_height),
      conv_elastic_disp(2, image_width * image_height);
    x_elastic_field.SetRandn();
    y_elastic_field.SetRandn();
    std::vector<DimsStrides> dims_stride, gauss_dims_stride,
       gauss_field_dims_stride;
    dims_stride.push_back(DimsStrides(x_field_dim, 1));
    dims_stride.push_back(DimsStrides(y_field_dim, x_field_dim));
    Tensor<BaseFloat> tx_elastic_field(dims_stride, x_elastic_field),
      ty_elastic_field(dims_stride, y_elastic_field);
    
    gauss_dims_stride.push_back(DimsStrides(gauss_dim, 1));
    gauss_dims_stride.push_back(DimsStrides(gauss_dim, gauss_dim));
    Matrix<BaseFloat> gauss_mat(1, gauss_dim * gauss_dim);
    Tensor<BaseFloat> gauss_tensor(gauss_dims_stride, gauss_mat);
  
    // generates Gaussian with standard deviation elastic_stddev
    GenerateGaussian(deform_opts.elastic_stddev, &gauss_tensor);
    gauss_field_dims_stride.push_back(DimsStrides(image_width, 1));
    gauss_field_dims_stride.push_back(DimsStrides(image_height, image_width));
    Tensor<BaseFloat> tx_gauss_field(gauss_field_dims_stride, x_gauss_field_mat),
      ty_gauss_field(gauss_field_dims_stride, y_gauss_field_mat);
  
    // convolve it with Gaussian (Gaussian of standard deviation sigma) to smooth displacement 
    tx_gauss_field.ConvTensorTensor(1.0, tx_elastic_field, gauss_tensor);
    ty_gauss_field.ConvTensorTensor(1.0, ty_elastic_field, gauss_tensor);
    
    // Normalize the random displacement field in x and y direction to have unit norm
    // sum elements of tx_gauss_field
    BaseFloat tx_norm2 = tx_gauss_field.FrobeniusNorm(),
      ty_norm2 = ty_gauss_field.FrobeniusNorm();
    BaseFloat x_scale = deform_opts.elastic_scale_factor / tx_norm2,
      y_scale = deform_opts.elastic_scale_factor / ty_norm2;

    /* tx_gauss_field and ty_gauss_field are displacement field for all pixels, 
       in x and y direction. disp_field contain final location of pixels which 
       is sum of displacement field and their inital coordinate.
       if apply_affine_trans=false, we add initial coordinates to disp_field.
       The normalized displacement field is multiplied by scaling factor 
       elastic_scale_factor, to control intensity of the deformation
    */
    tx_gauss_field.Scale(x_scale);
    ty_gauss_field.Scale(y_scale);
    std::vector<int32> indexes(2);
    if (deform_opts.apply_affine_trans == false) {
      for (int i = 0; i < image_height; i++) {
        for (int j = 0; j < image_width; j++) { 
          indexes[0] = i;
          indexes[1] = j;
          conv_elastic_disp(0, i * image_width + j) = i + tx_gauss_field(indexes); 
          conv_elastic_disp(1, i * image_width + j) = j + ty_gauss_field(indexes);
        }
      }
    } else {
      for (int i = 0; i < image_height; i++) {
        for (int j = 0; j < image_width; j++) {
          indexes[0] = i; 
          indexes[1] = j;
          conv_elastic_disp(0, i * image_width + j) = tx_gauss_field(indexes); 
          conv_elastic_disp(1, i * image_width + j) = ty_gauss_field(indexes);
        }
      }
    }
  disp_field->AddMat(1.0, conv_elastic_disp);
  }
}

void ApplyDistortionField(DeformationOptions deform,
                    const Matrix<BaseFloat> disp_field,
                    const Matrix<BaseFloat> input_feats,
                    Matrix<BaseFloat> *output_feats) {
  BaseFloat x_val, y_val;
  int x_new, y_new;
  int image_height = input_feats.NumRows();
  int image_width = input_feats.NumCols();
  KALDI_ASSERT(disp_field.NumCols() == image_width * image_height && 
               disp_field.NumRows() == 2);
  output_feats->Resize(image_height, image_width);
  // In order to compute the new grey level value for each pixel, 
  // we first innterpolating the value horizontally,
  // and interpolating the value vertically.
  // Then we use the average of horizental and vertical interpolation as
  // new gray level.
  for (int32 i = 0; i < image_height; i++) {
    for (int32 j = 0; j < image_width; j++) {
      int coord = i * image_width + j;
      x_new = static_cast<int>(disp_field(0, coord));
      y_new = static_cast<int>(disp_field(1, coord));

      // If a displacement ends up outside the image, a background value 0,
      // is assumed for all pixel locations outside the given image
      if (x_new > image_height-2 || y_new > image_width-2 || x_new < 0 || y_new < 0) {
        (*output_feats)(i, j) = 0.0;
      } else {
        // Linear interpolation in x and y direction
        x_val = input_feats(x_new, y_new) +
                (disp_field(0, coord)- x_new)* (input_feats(x_new + 1, y_new) 
              - input_feats(x_new, y_new));
        y_val = input_feats(x_new, y_new) +
                (disp_field(1, coord)- y_new) * (input_feats(x_new , y_new + 1) 
                - input_feats(x_new , y_new));
        (*output_feats)(i, j) = (x_val + y_val)/2;
      }
    }
  }
  // a random background inserted in the digit image. 
  // Each pixel value of the background was generated uniformly.  
  if (deform.apply_back_noise == true) {
    Matrix<BaseFloat> rand_background(image_height, image_width);
    rand_background.SetRandUniform();
    output_feats->AddMat(1.0, rand_background);
  }
}
}  // namespace kaldi
