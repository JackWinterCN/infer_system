# Static quantization of a model consists of the following steps:

#     Fuse modules
#     Insert Quant/DeQuant Stubs
#     Prepare the fused module (insert observers before and after layers)
#     Calibrate the prepared module (pass it representative data)
#     Convert the calibrated module (replace with quantized version)

import torch
from torch import nn
import copy
from mnist_net import Net
backend = "fbgemm"  # running on a x86 CPU. Use "qnnpack" if running on ARM.

# init_model = Net() 
# init_model.load_state_dict(torch.load('mnist_cnn.pt'))

# model = nn.Sequential(
#     # init_model
#   Net()
# )

# model = nn.Sequential(
#   torch.quantization.QuantStub(),
#   nn.Conv2d(1, 32, 3, 1),
#   nn.ReLU(),
#   nn.Conv2d(32, 64, 3, 1),
#   nn.ReLU(),
#   nn.MaxPool2d(2),
#   nn.Dropout(0.25),
#   nn.Flatten(),
#   nn.Linear(9216, 128),
#   nn.ReLU(),
#   nn.Dropout(0.5),
#   nn.LogSoftmax(1),
#   torch.quantization.DeQuantStub()
# )
# model = nn.Sequential(
#      torch.quantization.QuantStub(),
#      nn.Conv2d(1,64,3),
#      nn.ReLU(),
#      nn.Conv2d(64, 128, 3),
#      nn.ReLU(),
#      torch.quantization.DeQuantStub()
# )
model = nn.Sequential(
     nn.Conv2d(2,64,3),
     nn.ReLU(),
     nn.Conv2d(64, 128, 3),
     nn.ReLU()
)
## EAGER MODE
m = copy.deepcopy(model)
m.eval()
"""Fuse
- Inplace fusion replaces the first module in the sequence with the fused module, and the rest with identity modules
"""
# torch.quantization.fuse_modules(m, ['0','1'], inplace=True) # fuse first Conv-ReLU pair
# torch.quantization.fuse_modules(m, ['2','3'], inplace=True) # fuse second Conv-ReLU pair

"""Insert stubs"""
m = nn.Sequential(torch.quantization.QuantStub(), 
                  *m, 
                  torch.quantization.DeQuantStub())

"""Prepare"""
m.qconfig = torch.quantization.get_default_qconfig(backend)
torch.quantization.prepare(m, inplace=True)

"""Calibrate
- This example uses random data for convenience. Use representative (validation) data instead.
"""
with torch.inference_mode():
  for _ in range(10):
    x = torch.rand(1,2, 28, 28)
    m(x)
    
"""Convert"""
torch.quantization.convert(m, inplace=True)

"""Check"""
# print(m[[1]].weight().element_size()) # 1 byte instead of 4 bytes for FP32
torch.save(m, "mnist_a_ptq.pt")
# torch.save(m.state_dict(), "mnist_a_ptq.pt")

## FX GRAPH
# from torch.quantization import quantize_fx
# m = copy.deepcopy(model)
# m.eval()
# qconfig_dict = {"": torch.quantization.get_default_qconfig(backend)}
# # Prepare
# model_prepared = quantize_fx.prepare_fx(m, qconfig_dict)
# # Calibrate - Use representative (validation) data.
# with torch.inference_mode():
#   for _ in range(10):
#     x = torch.rand(1,2,28, 28)
#     model_prepared(x)
# # quantize
# model_quantized = quantize_fx.convert_fx(model_prepared)