#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <BTimer.hpp>
#include <spdhelper.hpp>
#include <BThreadPool.hpp>

std::string pnet_weight_path = std::string(MODEL_PATH) + "pnet.pt";

auto device_type = torch::DeviceType::CPU;

int testSingleThread()
{
  ENTER_FUNC;
  BTimer timer;
  BTimer ti;

  auto p = torch::jit::load(pnet_weight_path);

  if(p == nullptr)
  {
      LOGE("error load net from {}", pnet_weight_path);
      return -1;
  }

  std::vector<torch::jit::IValue> inputs;
  auto input = torch::rand({1,3,1080,1920});
  input = input.to(torch::Device(device_type, 0));
  inputs.emplace_back(input);
  LOGI("warm up...");
  timer.reset();
  for(int i = 0; i < 5; i++)
  {
      ti.reset();
      p->forward(inputs);
      LOGI("forward: {} ms", ti.elapsed());
  }
  LOGI("warm up over, time cost: {} ms", timer.elapsed());

  timer.reset();
  for(int i = 0; i < 50; i++)
  {
      ti.reset();
      p->forward(inputs);
      LOGI("forward: {} ms", ti.elapsed());
  }
  LOGI("run 50 iter, each iter mean time cost: {} ms", timer.elapsed() / 50.0);

  LEAVE_FUNC;
  return 0;
}

void testMultiThread()
{
  ENTER_FUNC;
  int thread_num = 2;
  BTimer timer;
  BThreadPool t{thread_num};

  std::vector<std::future<int>> res;
  timer.reset();
  for(int i = 0; i < thread_num; i++)
  {
    res.emplace_back(t.AddTask(testSingleThread));
  }
  for(auto& i: res)
    i.get();
  LOGI("whole time cost: {} ms", timer.elapsed());

  LEAVE_FUNC;
}



int main(int argc, char* argv[])
{
  ENTER_FUNC;
  if (argc < 2)
    testSingleThread();
  else
    testMultiThread();

  LEAVE_FUNC;
  return 0;
}

