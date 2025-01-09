import torch
import torchvision.models as models

model = models.resnet50(pretrained=True) # 加载预训练的 ResNet50 模型
model.eval()
dummy_input = torch.randn(1, 3, 224, 224) # 创建一个示例输入
torch.onnx.export(model, dummy_input, "resnet50.onnx", opset_version=11) # 将模型导出为 ONNX 格式