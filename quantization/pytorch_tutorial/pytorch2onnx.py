import torch 
import onnx 
from torch import nn
from mnist_net import Net

device = torch.device('cuda:0') 

# 1. create model and load weight 
# model = Net() 
# model.load_state_dict(torch.load('mnist_cnn_ptq.pt'))

# model = nn.Sequential(
#   Net()
# )
# model = nn.Sequential(torch.quantization.QuantStub(), 
#                   *model, 
#                   torch.quantization.DeQuantStub())
# model.load_state_dict(torch.load('mnist_a_ptq.pt'))

# 2. read model with weight
model = torch.load('test_4.pt')
# model.to(device) 

input = torch.randn(1, 2, 28, 28)
# input.to(device)

onnx_model = 'test_4.onnx' 

torch.onnx.export(model, 
                  input, 
                  onnx_model,
                  input_names=['input'], 
                  output_names=['output'],
                  do_constant_folding = False,
                  opset_version=14)