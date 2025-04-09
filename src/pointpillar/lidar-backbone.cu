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

#include <cuda_fp16.h>

#include <numeric>

#include "lidar-backbone.hpp"
#include "common/check.hpp"
#include "common/launch.cuh"
#include "common/tensorrt.hpp"
#include <fstream>
#include <iostream>

namespace pointpillar {
namespace lidar {

class BackboneImplement : public Backbone {
public:
    virtual ~BackboneImplement() {
        if (cls_) checkRuntime(cudaFree(cls_));
        if (box_) checkRuntime(cudaFree(box_));
        if (dir_) checkRuntime(cudaFree(dir_));
    }

    bool init(const std::string& model) {
        engine_ = TensorRT::load(model);
        if (engine_ == nullptr) return false;

        cls_dims_ = engine_->static_dims(3);
        box_dims_ = engine_->static_dims(4);
        dir_dims_ = engine_->static_dims(5);

        int32_t volumn = std::accumulate(cls_dims_.begin(), cls_dims_.end(), 1, std::multiplies<int32_t>());
        checkRuntime(cudaMalloc(&cls_, volumn * sizeof(float)));

        volumn = std::accumulate(box_dims_.begin(), box_dims_.end(), 1, std::multiplies<int32_t>());
        checkRuntime(cudaMalloc(&box_, volumn * sizeof(float)));

        volumn = std::accumulate(dir_dims_.begin(), dir_dims_.end(), 1, std::multiplies<int32_t>());
        checkRuntime(cudaMalloc(&dir_, volumn * sizeof(float)));
        return true;
    }

    virtual void print() override { engine_->print("Lidar Backbone"); }

    virtual void forward(const nvtype::half* voxels, const unsigned int* voxel_idxs, const unsigned int* params, void* stream = nullptr) override {
        cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);
        engine_->forward({voxels, voxel_idxs, params, cls_, box_, dir_}, static_cast<cudaStream_t>(_stream));

        // 以下代码是自己加的
        // 保存anchor
        // float *h_voxels_ = nullptr;
        // float *h_voxel_idxs = nullptr;
        // float *h_voxel_num = nullptr;
        // std::vector<int> voxel_idxs_dims,voxel_num_dims,voxels_dims;
        // voxels_dims=engine_->static_dims(0);
        // voxel_idxs_dims=engine_->static_dims(1);
        // voxel_num_dims=engine_->static_dims(2);

        // int32_t volumn0 = std::accumulate(voxels_dims.begin(), voxels_dims.end(), 1, std::multiplies<int32_t>());
        // int32_t volumn1 = std::accumulate(voxel_idxs_dims.begin(), voxel_idxs_dims.end(), 1, std::multiplies<int32_t>());
        // int32_t volumn2 = std::accumulate(voxel_num_dims.begin(), voxel_num_dims.end(), 1, std::multiplies<int32_t>());

        // // checkRuntime(cudaMallocHost(&h_voxels_, volumn0 * sizeof(float)));
        // // checkRuntime(cudaMemcpy(h_voxels_, voxels, volumn0 * sizeof(float), cudaMemcpyDeviceToHost));
        // // std::ofstream out("voxels.txt");
        // // if (out.is_open()) {
        // //     for (int i = 0; i < volumn0; i++) out << h_voxels_[i] << std::endl;
        // //     out.close();
        // // }

        // // int a=1;

        
        //volumn3=41997322=BHWC_cls=1x248x432x392
        int32_t volumn_cls = std::accumulate(cls_dims_.begin(), cls_dims_.end(), 1, std::multiplies<int32_t>());
        //volumn4=20998656=BHWC_box=1x248x432x196
        int32_t volumn4 = std::accumulate(box_dims_.begin(), box_dims_.end(), 1, std::multiplies<int32_t>());
        //volumn4=5999616=BHWC_dir=1x248x432x56
        int32_t volumn5 = std::accumulate(dir_dims_.begin(), dir_dims_.end(), 1, std::multiplies<int32_t>());
        
        float *h_cls = nullptr;
        // float *h_box = nullptr;;
        // float *h_dir = nullptr;;

        // int32_t volumn = std::accumulate(cls_dims_.begin(), cls_dims_.end(), 1, std::multiplies<int32_t>());
        checkRuntime(cudaMallocHost(&h_cls, volumn_cls * sizeof(float)));
        checkRuntime(cudaMemcpy(h_cls, cls_, volumn_cls * sizeof(float), cudaMemcpyDeviceToHost));

        std::ofstream out("../data_output/4d_array_trt.bin", std::ios::binary);
        if (out) {
            out.write(reinterpret_cast<const char*>(h_cls), volumn_cls*sizeof(h_cls));
            std::cout<<"!!!!!!!!!!!!!!!!!!!!!!"<<std::endl;
            std::cout << "Binary file saved to: data/4d_array_trt.bin" << std::endl;
            std::cout<<"!!!!!!!!!!!!!!!!!!!!!!"<<std::endl;
        } else {
            std::cerr << "Error: Failed to save file!" << std::endl;
            // return 1;
        }
        // std::ofstream out("cls.txt");
        // if (out.is_open()) {
        //     for (int i = 0; i < volumn; i++) out << h_cls[i] << std::endl;
        //     out.close();
        // }


        // volumn = std::accumulate(box_dims_.begin(), box_dims_.end(), 1, std::multiplies<int32_t>());
        // checkRuntime(cudaMallocHost(&h_box, volumn * sizeof(float)));
        // checkRuntime(cudaMemcpy(h_box, box_, volumn * sizeof(float), cudaMemcpyDeviceToHost));
        // std::ofstream out2("box.txt");
        // if (out2.is_open()) {
        //     for (int i = 0; i < volumn; i++) out2 << h_box[i] << std::endl;
        //     out2.close();
        // }

        // volumn = std::accumulate(dir_dims_.begin(), dir_dims_.end(), 1, std::multiplies<int32_t>());
        // checkRuntime(cudaMallocHost(&h_dir, volumn * sizeof(float)));
        // checkRuntime(cudaMemcpy(h_dir, dir_, volumn * sizeof(float), cudaMemcpyDeviceToHost));
        // std::ofstream out3("dir.txt");
        // if (out3.is_open()) {
        //     for (int i = 0; i < volumn; i++) out3 << h_dir[i] << std::endl;
        //     out3.close();
        // }

        //         // 释放内存
        // if (h_cls) cudaFreeHost(h_cls);
        // if (h_box) cudaFreeHost(h_box);
        // if (h_dir) cudaFreeHost(h_dir);

    }

    virtual float* cls() override { return cls_; }
    virtual float* box() override { return box_; }
    virtual float* dir() override { return dir_; }

private:
    std::shared_ptr<TensorRT::Engine> engine_;
    float *cls_ = nullptr;
    float *box_ = nullptr;
    float *dir_ = nullptr;
    std::vector<int> cls_dims_, box_dims_, dir_dims_;
};

std::shared_ptr<Backbone> create_backbone(const std::string& model) {
  std::shared_ptr<BackboneImplement> instance(new BackboneImplement());
  if (!instance->init(model)) {
    instance.reset();
  }
  return instance;
}

};  // namespace lidar
};  // namespace pointpillar