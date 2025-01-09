import torch
import torchvision

class QuantizedModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model_fp32 = model
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        
    def forward(self, x):
        x = self.quant(x)
        x = self.model_fp32(x)
        x = self.dequant(x)
        return x
        
def fuse_resnet18(model):
    torch.quantization.fuse_modules(model, [["conv1", "bn1", "relu"]], inplace=True)
    for module_name, module in model.named_children():
        if "layer" in module_name:
            for basic_block_name, basic_block in module.named_children():
                torch.quantization.fuse_modules(basic_block, [["conv1", "bn1", "relu"], ["conv2", "bn2"]], inplace=True)
                for sub_block_name, sub_block in basic_block.named_children():
                    if sub_block_name == "downsample":
                        torch.quantization.fuse_modules(sub_block, [["0", "1"]], inplace=True)    

model = torchvision.models.resnet18()
fuse_resnet18(model)
quantized_model = QuantizedModel(model)

backend = "fbgemm"
model.qconfig = torch.quantization.get_default_qconfig(backend)
torch.backends.quantized.engine = backend
model_static_quantized = torch.quantization.prepare(quantized_model, inplace=False)
model_static_quantized = torch.quantization.convert(model, inplace=False)
torch.jit.save(torch.jit.script(model_static_quantized), '/data/Igor/projects/torch-cpp/tutorial.pt')

module = torch.jit.load('/data/Igor/projects/torch-cpp/tutorial.pt')

inputs = torch.ones((1,3,224,224))

module(inputs)