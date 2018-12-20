#include <torch/torch.h>
#include <torch/script.h>
#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <spdhelper.hpp>
#include <opencv2/opencv.hpp>
#include <BTimer.hpp>

int main(int argc, char* argv[])
{
    ENTER_FUNC;
    BTimer timer;
    std::string root_path = R"(/home/beichen2012/gitee/pytorchRefineDet/)";
    std::string vgg_path = "refinedet_vgg16.pt";

    std::shared_ptr<torch::jit::script::Module> model = torch::jit::load(root_path + vgg_path);

    if(model == nullptr)
    {
        LOGE("model read failed!");
        return -1;
    }

//    std::string image_root = R"(/home/beichen2012/dataset/VOCdevkit/VOC2012/JPEGImages/)";
//    std::string image_path = image_root + "2012_004331.jpg";

//    cv::Mat src = cv::imread(image_path, 1);

//    cv::Mat input;
//    src.convertTo(input, CV_32FC3);

//    cv::Scalar smean = {104, 117, 123};
//    input -= smean;

//    cv::resize(input, input, {320,320});

//    std::vector<float> finput(3 * 320 * 320);
//    float* data = finput.data();

//    std::vector<cv::Mat> bgr;
//    for(int i = 0; i < 3; i++)
//    {
//        cv::Mat channel(320, 320, CV_32FC1, data);
//        data += 320 * 320;
//        bgr.push_back(channel);
//    }
//    cv::split(input, bgr);

//    torch::Tensor f = torch::from_blob(finput.data(), {1, 3,320,320});

//    std::vector<torch::jit::IValue> vi;
//    vi.push_back(f);

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::rand({1,3,320,320}, torch::device({torch::kCUDA, 0})));
    model->to(torch::Device{torch::DeviceType::CUDA, 0});

    for(int i = 0; i < 5; i++)
    {
        model->forward(inputs);
    }

    timer.reset();
    for(int i = 0; i < 10; i++)
        auto output = model->forward(inputs);
    LOGI("forward time cost: {} ms", timer.elapsed());

//    bool isTuple = output.isTuple();
//    auto ot = output.toTuple();
//    std::vector<torch::jit::IValue> a = ot->elements();

//    auto arm_loc = a[0].toTensor();


    LEAVE_FUNC;
    return 0;
}
