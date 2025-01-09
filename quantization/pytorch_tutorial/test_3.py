import torch
from torch import nn

# toy model
m = nn.Sequential(
  nn.Conv2d(2, 64, (8,),stride=1),
  nn.ReLU(),
  nn.Linear(16,10),
  nn.LSTM(10, 10))

m.eval()

## EAGER MODE
from torch.quantization import quantize_dynamic
model_quantized = quantize_dynamic(
    model=m, qconfig_spec={nn.LSTM, nn.Linear}, dtype=torch.qint8, inplace=False
)


dummy_input = torch.randn(1, 2, 10, 1) #.cuda()
torch.onnx.export(
        m,
        dummy_input,
        "quant_mnist.onnx",
        verbose=True,
        input_names = [ "actual_input_1" ],
        output_names = [ "actual_output_1" ],
        opset_version=13,
        do_constant_folding=True
        )

print("量化训练完成!")

# ## FX MODE
# from torch.quantization import quantize_fx
# qconfig_dict = {"": torch.quantization.default_dynamic_qconfig}  # An empty key denotes the default applied to all modules
# # TypeError: prepare_fx() missing 1 required positional argument: 'example_inputs'
# model_prepared = quantize_fx.prepare_fx(m, qconfig_dict)
# model_quantized = quantize_fx.convert_fx(model_prepared)