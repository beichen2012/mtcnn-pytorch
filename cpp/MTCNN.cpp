//
// Created by beichen2012 on 18-12-21.
//

#include "MTCNN.h"
#include <numeric>
#include <utility>

#include <spdhelper.hpp>
#include <BTimer.hpp>
#include <future>
#include <Eigen/Eigen>
using namespace Eigen;
#include "torchutils.h"

#define PNET_IMAGE_SIZE 12
#define RNET_IMAGE_SIZE 24
#define ONET_IMAGE_SIZE 48

int MTCNN::InitDetector(PTModelParam pparam)
{
    //0. 判断输入参数是否合法（暂时不检查，后面补上）；


    mtParam = *pparam;
    //1, 初始化各网络
//    //pnetauto
//    mWorker.reset(new BThreadPool(PNET_THREAD_NUM));
//    auto&& thread_ids = mWorker->GetThreadIds();
//    for (auto& i : thread_ids)
//    {
//        std::shared_ptr<Caffe2Predict> net =
//                std::make_shared<Caffe2Predict>(pparam->model_path[0], pparam->trained_path[0], pparam->use_gpu);
//        net->SetTransformParam(pparam->mean_value[0], pparam->scale_factor[0]);
//        mmpPNETS[i] = std::move(net);
//    }

    //pnet
    mpPNET = torch::jit::load(pparam->model_path[0]);
    //rnet
    mpRNET = torch::jit::load(pparam->model_path[1]);
    //onet
    mpONET = torch::jit::load(pparam->model_path[2]);

    //2. to device
    mpPNET->to(torch::Device(pparam->device_type, pparam->gpu_id));
    mpRNET->to(torch::Device(pparam->device_type, pparam->gpu_id));
    mpONET->to(torch::Device(pparam->device_type, pparam->gpu_id));

    return 0;
}


int MTCNN::UnInitDetector()
{
    mpPNET.reset();
    mpRNET.reset();
    mpONET.reset();
    return 0;
}


int MTCNN::DetectFace(cv::Mat & src, std::vector<cv::Rect>& outFaces)
{
    BTimer t;
    //int64_t tp, tr, to;
    outFaces.clear();
    int ret = 0;
    t.reset();
    ret = PNET(src, outFaces);
    LOGI("Pnet time cost: {}", t.elapsed());

//    auto tp = src.clone();
//    for(auto& i : outFaces)
//        cv::rectangle(tp, i, {0,255,0}, 2);
//    cv::imshow("pnet", tp);
//    cv::waitKey(0);

    if (ret < 0)
        return ret;
    t.reset();
    ret = RNET(src, outFaces);
    LOGI("Rnet time cost: {}", t.elapsed());

    if (ret < 0)
        return ret;
    t.reset();
    ret = ONET(src, outFaces);
    LOGI("Onet time cost: {}", t.elapsed());

    return ret;
}


int MTCNN::PNET(cv::Mat & src, std::vector<cv::Rect>& outFaces)
{
    //1. 计算金字塔
    //BTimer t;
    GetPyrScalors(src.cols, src.rows, mtParam.alg_param.scale_factor, mtParam.alg_param.min_face);
    //LOGI("pyramid time cost: {}", t.elapsed());
    //LOGI("pyramid total num: {}", mvScales.size());
    if (mvScales.size() <= 0)
        return -1;

    //3. 分层计算

    std::vector<std::vector<float>> layerBoxes;
    std::vector<std::vector<float>> layerScores;
    layerBoxes.resize(mvScales.size());
    layerScores.resize(mvScales.size());

    for(int i = 0; i < mvScales.size(); i++)
    {
        RunPNetLayer(src, i, layerBoxes[i], layerScores[i]);
    }

    //5. NMS
    std::vector<float> bboxes;
    std::vector<float> score;
    for (int i = 0; i < layerBoxes.size(); i++)
    {
        bboxes.insert(bboxes.end(), layerBoxes[i].begin(), layerBoxes[i].end());
        score.insert(score.end(), layerScores[i].begin(), layerScores[i].end());
    }

    //t.reset();
    NMS(bboxes, score, 0.7, 0);
    //LOGI("Pnet NMS time cost: {}", t.elapsed());


    // 组合结果
    for (int i = 0; i < bboxes.size() / 4; i++)
    {
        cv::Rect r = cv::Rect(bboxes[4 * i], bboxes[4 * i + 1], bboxes[4 * i + 2],bboxes[4 * i + 3]);
        outFaces.push_back(r);
    }

    return 0;
}
int MTCNN::RunPNetLayer(cv::Mat& src, int scale_idx, std::vector<float>& outBoxes, std::vector<float>& outScores)
{
    //1. 这一层的Image
    //BTimer t;
    float scalor = mvScales[scale_idx];
    int nw = src.cols * scalor;
    int nh = src.rows * scalor;
    cv::Mat image;
    //t.reset();
    cv::resize(src, image, { nw, nh });
    //LOGI("pnet layer resize : {} us", t.elapsed_micro());

    auto& pnet = mpPNET;
    torch::Tensor input;
    //torch::Tensor input = torch::zeros({image.rows, image.cols, image.channels()});

    cv::Mat img_float;
    image.convertTo(img_float, CV_32F);
    img_float -= mtParam.mean_value[0];
    if(std::abs(1.0f - mtParam.scale_factor[0]) > 0.001)
    {
        img_float *= mtParam.scale_factor[0];
    }
    auto nn = input.numel();
    //memcpy(input.data_ptr(), img_float.data, input.numel() * sizeof(float));
    input = torch::from_blob(img_float.data, {image.rows, image.cols, image.channels()});

    input = input.permute({2, 0, 1});
    input.unsqueeze_(0);

    input = input.to(torch::Device(mtParam.device_type, mtParam.gpu_id));

    std::vector<torch::jit::IValue> inputs;
    inputs.emplace_back(std::move(input));

    auto&& out = pnet->forward(inputs);
    auto ot = out.toTuple();

    inputs = ot->elements();
    auto cls = inputs[0].toTensor().cpu();
    auto reg = inputs[1].toTensor().cpu();

    auto cls_map_h = cls.size(2);
    auto cls_map_w = cls.size(3);

    auto cls_viewer = cls.accessor<float, 4>();
    auto reg_viewer = reg.accessor<float, 4>();

    std::vector<float> bboxes;	//M * 4
    std::vector<float> score;	//M * 1
    std::vector<int> pos;		//M * 2
    for(int y = 0; y < cls_map_h; y++)
    {
        for(int x = 0; x < cls_map_w; x++)
        {
            if(cls_viewer[0][1][y][x] > mtParam.alg_param.cls_thre[0])
            {
                score.push_back(cls_viewer[0][1][y][x]);

                pos.push_back(x);
                pos.push_back(y);

                bboxes.push_back(reg_viewer[0][0][y][x]);
                bboxes.push_back(reg_viewer[0][1][y][x]);
                bboxes.push_back(reg_viewer[0][2][y][x]);
                bboxes.push_back(reg_viewer[0][3][y][x]);
            }
        }
    }

    //LOGI("pnet layer feature map extract time: {}", t.elapsed());
    if (bboxes.size() <= 0)
        return 0;

    //5. 坐标映射回原图
    //t.reset();
    GenerateBBox(bboxes,pos, scalor);
    //LOGI("pnet GenerateBBox time: {}", t.elapsed());

    //6. NMS
    //t.reset();
    NMS(bboxes, score, 0.5, 0);
    //LOGI("pnet NMS time: {}", t.elapsed());

    //7. 填充输出
    outBoxes = std::move(bboxes);
    outScores = std::move(score);

    return 0;
}
int MTCNN::NMS(std::vector<float>& bbox, std::vector<float>& score, float thre, int type) const
{
    typedef struct TBBox {
        int x1;
        int y1;
        int x2;
        int y2;
        int order;
        float score;
        float area;
        bool exist;
    }TBBox;
    std::vector<TBBox> tbox;
    for (int i = 0; i < score.size(); i++)
    {
        TBBox b;
        b.x1 = bbox[i * 4 + 0];
        b.y1 = bbox[i * 4 + 1];
        b.x2 = bbox[i * 4 + 2] + b.x1;
        b.y2 = bbox[i * 4 + 3] + b.y1;

        b.order = i;
        b.score = score[i];
        b.exist = true;
        b.area = bbox[i*4 + 2] * bbox[i*4 + 3];
        tbox.push_back(b);
    }

    //升序
    std::sort(tbox.begin(), tbox.end(), [](const TBBox& left, const TBBox& right) {
        return left.score < right.score;
    });

    std::vector<int> heros;
    int order = 0;
    float IOU = 0;
    float maxX = 0;
    float maxY = 0;
    float minX = 0;
    float minY = 0;

    while (tbox.size() > 0)
    {
        //最后的score是最大的
        TBBox tb = tbox.back();
        order = tb.order;
        tbox.pop_back();
        if (order < 0)
            continue;

        //保存当前box的索引
        heros.push_back(order);
        //遍历时忽略掉
        tb.exist = false;

        for (auto &i : tbox)
        {
            if (!i.exist)
                continue;
            //计算当前box与tb的交并比
            auto& ti = i;
            maxX = ti.x1 > tb.x1 ? ti.x1 : tb.x1;
            maxY = ti.y1 > tb.y1 ? ti.y1 : tb.y1;
            minX = ti.x2 < tb.x2 ? ti.x2 : tb.x2;
            minY = ti.y2 < tb.y2 ? ti.y2 : tb.y2;

            maxX = ((minX - maxX + 1) > 0) ? (minX - maxX + 1) : 0;
            maxY = ((minY - maxY + 1)>0) ? (minY - maxY + 1) : 0;
            //IOU reuse for the area of two bbox
            IOU = maxX * maxY;
            if (type == 0)
            {
                //IOU
                IOU = IOU / (tb.area + ti.area - IOU);
            }
            else
            {
                //IOM
                IOU = IOU / (tb.area > ti.area ? ti.area : tb.area);
            }
            //compare
            if (IOU > thre)
            {
                ti.exist = false;
                ti.order = -1;
            }
        }
    }

    //结果
    std::vector<float> outbox;
    std::vector<float> outscore;
    for (auto& i : heros)
    {
        outscore.push_back(score[i]);
        outbox.push_back(bbox[4 * i]);
        outbox.push_back(bbox[4 * i + 1]);
        outbox.push_back(bbox[4 * i + 2]);
        outbox.push_back(bbox[4 * i + 3]);
    }

    bbox.clear();
    bbox = std::move(outbox);
    score.clear();
    score = std::move(outscore);
    return 0;
}


//生成bbox
int MTCNN::GenerateBBox(std::vector<float>& bbox, std::vector<int>& pos, float scalor) const
{
    //0.
    static const int stride = 2;
    static const int cellsize = PNET_IMAGE_SIZE;
    //1. 转换成cv::Mat(后续可考虑使用eigen)
    int w = 4;
    int h = bbox.size() / w;			//h = pos.size() / 2;
    float gx;
    float gy;
    float* ptr;
    if (h != pos.size() / 2)
    {
        bbox.clear();
        return -1;
    }
    //h*w的矩阵，标识偏移量
    Map<Matrix<float, Eigen::Dynamic, Eigen::Dynamic, RowMajor> > reg(bbox.data(), h, w);
    //h*2的矩阵，标识Patch with height
    Map<Matrix<int, Eigen::Dynamic, Eigen::Dynamic, RowMajor> > gp(pos.data(), h, 2);
    MatrixXf Gstart = gp.cast<float>();
    Gstart.array() *= (stride / scalor);


    //reg 缩放回原图尺寸
    reg.col(0) *= (cellsize / scalor);
    reg.col(1) *= (cellsize / scalor);
    reg.col(2).array() = reg.col(2).array().exp();
    reg.col(3).array() = reg.col(3).array().exp();
    reg.col(2).array() *= (cellsize / scalor);
    reg.col(3).array() *= (cellsize / scalor);

    //reg加上起点及终点偏移量
    reg.col(0) += Gstart.col(0);
    reg.col(1) += Gstart.col(1);
//
    return 0;
}

int MTCNN::SquareBBox(std::vector<cv::Rect>& rect)
{
    int size;
    cv::Rect r;
    for (auto& i : rect)
    {
        size = std::max(i.width, i.height);
        r.x = i.x + i.width * 0.5 - size * 0.5;
        r.y = i.y + i.height * 0.5 - size * 0.5;
        r.width = size;
        r.height = size;
        i = r;
    }
    return 0;
}

int MTCNN::PadBBox(cv::Rect rect, cv::Rect & rdst, cv::Rect & rsrc, int W, int H)
{
    //original
    int x0 = rect.x;
    int y0 = rect.y;
    int x1 = rect.width + x0;
    int y1 = rect.height + y0;

    //src, dst
    int sx0, sy0, sx1, sy1;
    int dx0, dy0, dx1, dy1;
    int minx, miny;
    sx0 = x0;
    dx0 = 0;
    sy0 = y0;
    dy0 = 0;

    sx1 = x1;
    dx1 = rect.width;
    sy1 = y1;
    dy1 = rect.height;

    //边界判断
    if (x0 < 0)
    {
        sx0 = 0;
        dx0 = -x0;
    }
    if (y0 < 0)
    {
        sy0 = 0;
        dy0 = -y0;
    }
    if (x1 > W - 1)
    {
        sx1 = W - 1;
        dx1 = sx1 - sx0;
    }
    if (y1 > H - 1)
    {
        sy1 = H - 1;
        dy1 = sy1 - sy0;
    }
    //src,dst的shape要一致
    miny = std::min(dy1 - dy0, sy1 - sy0);
    dy1 = dy0 + miny;
    sy1 = sy0 + miny;
    minx = std::min(dx1 - dx0, sx1 - sx0);
    dx1 = dx0 + minx;
    sx1 = sx0 + minx;

    //填充输出
    rsrc = { sx0, sy0, sx1 - sx0, sy1 - sy0 };
    rdst = { dx0, dy0, dx1 - dx0, dy1 - dy0 };
    return 0;
}

int MTCNN::ReProjectBBox(std::vector<float>& origin, std::vector<float>& bbox)
{
    //wrap to matrix
    int N = bbox.size() / 4;

    //
    Map<Matrix<float, Eigen::Dynamic, Eigen::Dynamic, RowMajor>> mo(origin.data(), N, 4);
    Map<Matrix<float, Eigen::Dynamic, Eigen::Dynamic, RowMajor>> mb(bbox.data(), N, 4);

    //scale
    mb.col(0).array() *= mo.col(2).array();
    mb.col(1).array() *= mo.col(3).array();
    mb.col(0) += mo.col(0);
    mb.col(1) += mo.col(1);

    //w,h
    mb.col(2).array() = mb.col(2).array().exp();
    mb.col(3).array() = mb.col(3).array().exp();
    mb.col(2).array() *= mo.col(2).array();
    mb.col(3).array() *= mo.col(3).array();

    return 0;
}

int MTCNN::RunRONet(std::shared_ptr<torch::jit::script::Module> net,
                    cv::Mat & src,
                    std::vector<cv::Rect>& outFaces,
                    int IMAGE_SIZE)
{
    int stage = 1;
    if (IMAGE_SIZE == 48)
        stage = 2;
    cv::Scalar mean = mtParam.mean_value[stage];
    float scalor = mtParam.scale_factor[stage];
    float cls_thre = mtParam.alg_param.cls_thre[stage];

    //0.
    if (outFaces.size() <= 0)
    {
        return 0;
    }
    int W = src.cols;
    int H = src.rows;
    //1. 生成bbox
    SquareBBox(outFaces);

    //2. pad
    cv::Rect rdst, rsrc;
    int size;

    std::vector<cv::Mat> imgs;
    std::vector<float> bboxOriginal;
    for (int i = 0; i < outFaces.size(); i++)
    {
        size = outFaces[i].width;
        cv::Mat crop = cv::Mat::zeros(cv::Size(size, size), CV_8UC3);
        PadBBox(outFaces[i], rdst, rsrc, W, H);
        if (rdst.x < 0 || rdst.y < 0 || rsrc.x < 0 || rsrc.y < 0 ||
            rdst.width > size || rdst.height > size ||
            rsrc.width > W || rsrc.height > H)
        {
            continue;
        }
        //copy
        src(rsrc).copyTo((cv::Mat)crop(rdst));
        //
        cv::resize(crop, crop, cv::Size(IMAGE_SIZE, IMAGE_SIZE));
        cv::Mat crop_float;
        crop.convertTo(crop_float, CV_32F);
        crop_float -= mean;
        if (std::abs(1.0f - scalor) > 0.001)
            crop_float *= scalor;
        imgs.emplace_back(crop_float);

        bboxOriginal.push_back(outFaces[i].x);
        bboxOriginal.push_back(outFaces[i].y);
        bboxOriginal.push_back(outFaces[i].width);
        bboxOriginal.push_back(outFaces[i].height);
    }

    //3. predict
    int N = imgs.size();
    torch::Tensor input = torch::zeros({N, IMAGE_SIZE, IMAGE_SIZE, 3});
    float* head = input.data<float>();
    for(int i = 0; i < imgs.size(); i++)
    {
        memcpy(head, imgs[i].data, sizeof(float) * 3 * IMAGE_SIZE * IMAGE_SIZE);
        head += 3 * IMAGE_SIZE * IMAGE_SIZE;
    }

    input = input.permute({0, 3, 1, 2});
    input = input.to(torch::Device(mtParam.device_type, mtParam.gpu_id));

    std::vector<torch::jit::IValue> inputs = {input};

    auto&& out = net->forward(inputs);
    auto ot = out.toTuple();
    inputs = ot->elements();
    auto cls = inputs[0].toTensor().cpu();
    auto reg = inputs[1].toTensor().cpu();

    std::vector<float> bboxes;	//用于标识回归框
    std::vector<float> score;	//分数
    std::vector<int> keep_index;	//位置
    std::vector<float> resBBoxes;	// 原框及最终框（复用）

    auto cls_viewer = cls.accessor<float, 2>();//N * 2
    auto reg_viewer = reg.accessor<float, 2>();//N * 4

    for(int i = 0; i < N; i++)
    {
        if(cls_viewer[i][1] > cls_thre)
        {
            score.push_back(cls_viewer[i][1]);
            keep_index.push_back(i);

            bboxes.push_back(reg_viewer[i][0]);
            bboxes.push_back(reg_viewer[i][1]);
            bboxes.push_back(reg_viewer[i][2]);
            bboxes.push_back(reg_viewer[i][3]);
        }
    }

    outFaces.clear();
    if (score.size() <= 0)
        return 0;

    // reproject to original
    for (auto& i : keep_index)
    {
        resBBoxes.push_back(bboxOriginal[4 * i]);
        resBBoxes.push_back(bboxOriginal[4 * i + 1]);
        resBBoxes.push_back(bboxOriginal[4 * i + 2]);
        resBBoxes.push_back(bboxOriginal[4 * i + 3]);
    }
    bboxOriginal = std::move(resBBoxes);
    keep_index.clear();
    ReProjectBBox(bboxOriginal, bboxes);

    //NMS
    if(IMAGE_SIZE == 24)
        NMS(bboxes, score, 0.6, 0);
    else
        NMS(bboxes, score, 0.6, 1);

    //7. output
    outFaces.clear();
    for (int i = 0; i < bboxes.size() / 4; i++)
    {
        cv::Rect r = cv::Rect(bboxes[4 * i], bboxes[4 * i + 1], bboxes[4 * i + 2],bboxes[4 * i + 3]);
        outFaces.push_back(r);
    }
    return 0;
}

int MTCNN::RNET(cv::Mat & src, std::vector<cv::Rect>& outFaces)
{
    return RunRONet(mpRNET, src, outFaces, RNET_IMAGE_SIZE);
}

int MTCNN::ONET(cv::Mat & src, std::vector<cv::Rect>& outFaces)
{
    return RunRONet(mpONET, src, outFaces, ONET_IMAGE_SIZE);
    return 0;
}

int MTCNN::GetPyrScalors(int W, int H, float scale_factor, int min_face_size)
{
    //0, 准备金字塔层数
    mvScales.clear();
    float minl = H > W ? W : H;
    float base_scale = float(PNET_IMAGE_SIZE) / float(min_face_size);
    int face_count = 0;
    while (true)
    {
        float s = base_scale * (float)pow(double(scale_factor), double(face_count));
        if (floor(double(minl * s)) < PNET_IMAGE_SIZE)
            break;
        mvScales.push_back(s);
        face_count++;
    }
    return 0;
}


