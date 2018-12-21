//
// Created by beichen2012 on 18-12-21.
//

#include "torchutils.h"

void Preprocess(cv::Mat& src, torch::Tensor& input, cv::Scalar mean, float scalor)
{
    cv::Mat img_float;
    src.convertTo(img_float, CV_32F);
    img_float -= mean;
    if(std::abs(1.0f - scalor) > 0.001)
    {
        img_float *= scalor;
    }

    input = torch::from_blob(img_float.data, {src.rows, src.cols, src.channels()});

//    input = torch::CPU(torch::kFloat32).tensorFromBlob(img_float.data, {src.cols, src.rows, src.channels()});
    input = input.permute({2, 0, 1});
    input.unsqueeze_(0);
}