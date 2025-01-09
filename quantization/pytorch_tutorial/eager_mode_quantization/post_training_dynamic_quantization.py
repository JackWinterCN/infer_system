import torch
import torch.ao.quantization

# define a floating point model
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 2)
        print('-'*80)
        print(self.fc.weight)
        

    def forward(self, x):
        x = self.fc(x)
        return x

# create a model instance
model_fp32 = M()

# create a quantized model instance
model_int8 = torch.ao.quantization.quantize_dynamic(
    model_fp32,  # the original model
    {torch.nn.Linear},  # a set of layers to dynamically quantize
    dtype=torch.qint8)  # the target dtype for quantized weights

print('-'*80)
print(model_int8)


# run the model
input_fp32 = torch.randn(1, 2, 3, 4)
res = model_int8(input_fp32)
print('-'*80)
print(input_fp32)
print('-'*80)
print(res)



input = torch.randn(1, 2, 3, 4)

onnx_model = 'ptdq_fp32.onnx' 
torch.onnx.export(model_fp32, 
                  input, 
                  onnx_model,
                  input_names=['input'], 
                  output_names=['output'],
                  do_constant_folding = False,
                  opset_version=14)

onnx_model = 'ptdq_int8.onnx' 
torch.onnx.export(model_int8, 
                  input, 
                  onnx_model,
                  input_names=['input'], 
                  output_names=['output'],
                  do_constant_folding = False,
                  opset_version=13)