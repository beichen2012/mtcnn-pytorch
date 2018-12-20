#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <string>
#include <memory>

int main() {
  torch::Tensor tensor = torch::rand({2, 3});
  std::cout << tensor << std::endl;

  std::string root_path = R"(/home/beichen2012/gitee/pytorchRefineDet/)";
  std::string vgg_path = "refinedet_vgg16.pt";

  std::shared_ptr<torch::jit::script::Module> model = torch::jit::load(root_path + vgg_path);


//  auto model = torch::jit::

  return 0;
}
