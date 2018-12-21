//
// Created by beichen2012 on 18-12-21.
//
#pragma once
#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>

void Preprocess(cv::Mat& src, torch::Tensor& input, cv::Scalar mean, float scalor = 1.0f);
