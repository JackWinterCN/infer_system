import torch
from torch.ao.quantization.quantize_pt2e import prepare_pt2e
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 10)

    def forward(self, x):
       return self.linear(x)

# initialize a floating point model
float_model = M().eval()

# define calibration function
def calibrate(model, data_loader):
    model.eval()
    with torch.no_grad():
        for image, target in data_loader:
            model(image)

# Step 1. program capture
# NOTE: this API will be updated to torch.export API in the future, but the captured
# result should mostly stay the same
m = capture_pre_autograd_graph(m, *example_inputs)
# we get a model with aten ops

# Step 2. quantization
# backend developer will write their own Quantizer and expose methods to allow
# users to express how they
# want the model to be quantized
quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())
# or prepare_qat_pt2e for Quantization Aware Training
m = prepare_pt2e(m, quantizer)

# run calibration
# calibrate(m, sample_inference_data)
m = convert_pt2e(m)

# Step 3. lowering
# lower to target backend