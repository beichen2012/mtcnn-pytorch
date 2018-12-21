//
// Created by beichen2012 on 18-12-21.
//
#pragma once
#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

//算法参数
typedef struct _tAlgParam
{
    int min_face;					//检测的最小人脸： 40
    float scale_factor;				//金字塔步长： 0.709
    float cls_thre[3];				//三个网络预测输出的阈值

}TAlgParam, *PTAlgParam;


typedef struct _tModelParam
{
    //模型文件
    std::vector<std::string> model_path;
    //各通道均值
    std::vector<cv::Scalar> mean_value;
    //缩放系数
    std::vector<float> scale_factor;
    //device
    torch::DeviceType device_type;
    int gpu_id;

    //算法参数
    TAlgParam alg_param;
}TModelParam, *PTModelParam;

class MTCNN
{
public:
/* 初始化检测器
	*/
    int InitDetector(PTModelParam pparam);

    /*检测人脸*/
    int DetectFace(cv::Mat& src, std::vector<cv::Rect>& outFaces);

    /* 反初始化检测器*/
    int UnInitDetector();

    // pnet 分层计算
    int RunPNetLayer(cv::Mat& src, int scale_idx, std::vector<float>& outBoxes, std::vector<float>& outScores);

protected:
    int PNET(cv::Mat& src, std::vector<cv::Rect>& outFaces);
    int RNET(cv::Mat& src, std::vector<cv::Rect>& outFaces);
    int ONET(cv::Mat& src, std::vector<cv::Rect>& outFaces);

    //计算缩放
    int GetPyrScalors(int W, int H, float scale_factor, int min_face_size);


    //pnet映射bbox
    int GenerateBBox(std::vector<float>& bbox, std::vector<int>& pos, float scalor) const;

    //矩形化
    int SquareBBox(std::vector<cv::Rect>& rect);
    //pad
    int PadBBox(cv::Rect rect, cv::Rect& rdst, cv::Rect& rsrc, int W, int H);

    //re-project to original image
    int ReProjectBBox(std::vector<float>& origin, std::vector<float>& bbox);

    //rnet onet的共用
    int RunRONet(std::shared_ptr<torch::jit::script::Module> net, cv::Mat& src, std::vector<cv::Rect>& outFaces,
                 int IMAGE_SIZE);


private:
    std::vector<float> mvScales;
    TModelParam mtParam;
    //pnet
    //std::map<std::thread::id, std::shared_ptr<Caffe2Predict> > mmpPNETS;
    //std::shared_ptr<BThreadPool> mWorker;
    std::shared_ptr<torch::jit::script::Module> mpPNET;
    //rnet
    std::shared_ptr<torch::jit::script::Module> mpRNET;
    //onet
    std::shared_ptr<torch::jit::script::Module> mpONET;

    //netoutput
    static const std::vector<std::string> pnet_output;
    static const std::vector<std::string> rnet_output;
    static const std::vector<std::string> onet_output;

private:
    //NMS
    //type == 0, Union
    //type == 1, Minimum
    int NMS(std::vector<float>& bbox, std::vector<float>& score, float thre, int type) const;

};



