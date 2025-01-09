# Static quantization of a model consists of the following steps:

#     Fuse modules
#     Insert Quant/DeQuant Stubs
#     Prepare the fused module (insert observers before and after layers)
#     Calibrate the prepared module (pass it representative data)
#     Convert the calibrated module (replace with quantized version)

import torch
from torch import nn
import copy
from torchvision import datasets
from torchvision import transforms
from torch.quantization import quantize_dynamic

from mnist_net import Net


model = Net() 
model.load_state_dict(torch.load('mnist_cnn.pt'))
model.eval()



# 定义要量化的层类型
qconfig_spec = {nn.Linear, nn.Conv2d}
# 进行动态量化，指定量化的数据类型为torch.qint8
model_quantized = quantize_dynamic(
    model=model,
    qconfig_spec=qconfig_spec,
    dtype=torch.qint8,
    inplace=False
)

# torch.save(model_quantized.state_dict(), "mnist_cnn_ptq.pt")
torch.save(model_quantized, "mnist_cnn_ptq.pt")