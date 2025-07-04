/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef __LIDAR_POSTPROCESS_HPP__
#define __LIDAR_POSTPROCESS_HPP__

#include <memory>
#include <vector>
#include "common/dtype.hpp"

namespace pointpillar {
namespace lidar {

struct BoundingBox {
    float x;
    float y;
    float z;
    float w;
    float l;
    float h;
    float rt;
    int id;
    float score;
    BoundingBox(){};
    BoundingBox(float x_, float y_, float z_, float w_, float l_, float h_, float rt_, int id_, float score_)
        : x(x_), y(y_), z(z_), w(w_), l(l_), h(h_), rt(rt_), id(id_), score(score_) {}
};

struct PostProcessParameter {
    nvtype::Float3 min_range;
    nvtype::Float3 max_range;
    nvtype::Int2 feature_size;
    // kitti
    // int num_classes = 3;
    // int num_anchors = 6;
    // int len_per_anchor = 4;

    // kl 14 类
    // int num_classes = 14;
    // int num_anchors = 28;
    // int len_per_anchor = 4;

    // kl 15 类
    int num_classes = 15;
    int num_anchors = 30;
    int len_per_anchor = 4;

    // kitti
    // float anchors[24] = {
    //         3.9,1.6,1.56,0.0,
    //         3.9,1.6,1.56,1.57,
    //         0.8,0.6,1.73,0.0,
    //         0.8,0.6,1.73,1.57,
    //         1.76,0.6,1.73,0.0,
    //         1.76,0.6,1.73,1.57,
    //     };

    // kl 14 类
    // float anchors[112] = {
    //     // Pedestrian
    //     0.8, 0.6, 1.73, 0.0,    0.8, 0.6, 1.73, 1.57,
    
    //     // Car
    //     3.9, 1.6, 1.56, 0.0,    3.9, 1.6, 1.56, 1.57,
    
    //     // IGV-Full
    //     10.5, 2.94, 3.47, 0.0,  10.5, 2.94, 3.47, 1.57,
    
    //     // Truck
    //     6.93, 2.51, 2.84, 0.0,  6.93, 2.51, 2.84, 1.57,
    
    //     // Trailer-Empty
    //     12.29, 2.90, 3.87, 0.0, 12.29, 2.90, 3.87, 1.57,
    
    //     // Trailer-Full
    //     12.29, 2.90, 3.87, 0.0, 12.29, 2.90, 3.87, 1.57,
    
    //     // IGV-Empty
    //     10.5, 2.94, 3.47, 0.0,  10.5, 2.94, 3.47, 1.57,
    
    //     // Crane
    //     10.5, 2.94, 3.47, 0.0,  10.5, 2.94, 3.47, 1.57,
    
    //     // OtherVehicle
    //     4.0, 1.8, 1.5, 0.0,     4.0, 1.8, 1.5, 1.57,
    
    //     // Cone
    //     0.5, 0.5, 1.0, 0.0,     0.5, 0.5, 1.0, 1.57,
    
    //     // ContainerForklift
    //     5.0, 2.0, 5.0, 0.0,     5.0, 2.0, 5.0, 1.57,
    
    //     // Forklift
    //     1.76, 1.0, 1.73, 0.0,   1.76, 1.0, 1.73, 1.57,
    
    //     // Lorry
    //     6.93, 2.51, 2.84, 0.0,  6.93, 2.51, 2.84, 1.57,
    
    //     // ConstructionVehicle
    //     6.93, 2.51, 2.84, 0.0,  6.93, 2.51, 2.84, 1.57
    // };

    // kl 15 类
    float anchors[120] = {
        // Pedestrian
        0.70, 0.72, 1.73, 0.0,    0.70, 0.72, 1.73, 1.57,
    
        // Car
        4.63, 1.98, 1.64, 0.0,    4.63, 1.98, 1.64, 1.57,
    
        // IGV-Full
        12.96, 2.87, 4.21, 0.0,  12.96, 2.87, 4.21, 1.57,
    
        // Truck
        6.16, 2.83, 3.79, 0.0,  6.16, 2.83, 3.79, 1.57,
    
        // Trailer-Empty
        12.13, 2.78, 2.07, 0.0, 12.13, 2.78, 2.07, 1.57,
    
        // Trailer-Full
        12.13, 2.78, 4.43, 0.0, 12.13, 2.78, 4.43, 1.57,
    
        // IGV-Empty
        14.59, 3.25, 2.23, 0.0,  14.59, 3.25, 2.23, 1.57,
    
        // Crane
        8.97, 4.66, 5.10, 0.0,  8.97, 4.66, 5.10, 1.57,
    
        // OtherVehicle
        4.52, 1.82, 1.5, 0.0,   4.52, 1.82, 1.5, 1.57,
    
        // Cone
        0.38, 0.38, 0.8, 0.0,     0.38, 0.38, 0.8, 1.57,
    
        // ContainerForklift
        7.08, 5.14, 6.71, 0.0,     7.08, 5.14, 6.71, 1.57,
    
        // Forklift
        4.22, 2.13, 2.49, 0.0,   4.22, 2.13, 2.49, 1.57,
    
        // Lorry
        7.65, 2.90, 3.19, 0.0,  7.65, 2.90, 3.19, 1.57,
    
        // ConstructionVehicle
        7.39, 3.33, 5.12, 0.0,  7.39, 3.33, 5.12, 1.57,

        // WheelCrane
        6.93, 2.51, 2.84, 0.0,  6.93, 2.51, 2.84, 1.57

    };

    // kitti
    // nvtype::Float3 anchor_bottom_heights{-1.78,-0.6,-0.6};

    // kl 14类
    // float anchor_bottom_heights[14] = {
    //     -0.6,    // Pedestrian
    //     -1.78,   // Car
    //     -0.085,  // IGV-Full
    //     -0.6,    // Truck
    //     0.115,   // Trailer-Empty
    //     0.115,   // Trailer-Full
    //     -0.085,  // IGV-Empty
    //     -0.085,  // Crane
    //     -0.8,    // OtherVehicle
    //     -0.2,    // Cone
    //     -0.5,    // ContainerForklift
    //     -0.6,    // Forklift
    //     -0.6,    // Lorry
    //     -0.6     // ConstructionVehicle
    // };

    // kl 15类
    float anchor_bottom_heights[15] = {
        0,    // Pedestrian
        0,   // Car
        0,  // IGV-Full
        0,    // Truck
        0,   // Trailer-Empty
        0,   // Trailer-Full
        0,  // IGV-Empty
        0,  // Crane
        0,    // OtherVehicle
        0,    // Cone
        0,    // ContainerForklift
        0,    // Forklift
        0,    // Lorry
        0,    // ConstructionVehicle
        0     // WheelCrane
    };
    int num_box_values = 7;
    float score_thresh = 0.5;
    float dir_offset = 0.78539;
    float nms_thresh = 0.2;
};

class PostProcess {
    public:
        virtual void forward(const float* cls, const float* box, const float* dir, void* stream) = 0;

        virtual std::vector<BoundingBox> bndBoxVec() = 0;
};

std::shared_ptr<PostProcess> create_postprocess(const PostProcessParameter& param);

};  // namespace lidar
};  // namespace pointpillar

#endif  // __LIDAR_POSTPROCESS_HPP__
