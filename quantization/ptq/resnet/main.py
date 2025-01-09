import os
import sys
import time
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets
import torchvision.models as models

import torchvision.transforms as transforms


from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor

from torchvision.transforms._presets import ImageClassification
from torchvision.utils import _log_api_usage_once
from torchvision.models.resnet import Bottleneck


from torchvision.models.resnet import conv3x3

def export_resnet2onnx(model, name):
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224) # 创建一个示例输入
    torch.onnx.export(model, dummy_input, name + ".onnx", opset_version=13) # 将模型导出为 ONNX 格式

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.skip_add = nn.quantized.FloatFunctional() 

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.skip_add.add(out, identity)
        out = self.relu(out)

        return out

from torchvision.models.resnet import ResNet18_Weights, _resnet
from torchvision.models._api import register_model, Weights, WeightsEnum
from torchvision.models._utils import _make_divisible, _ovewrite_named_param, handle_legacy_interface

@register_model()
@handle_legacy_interface(weights=("pretrained", ResNet18_Weights.IMAGENET1K_V1))
def quantizedresnet18(*, weights: Optional[ResNet18_Weights] = None, progress: bool = True, **kwargs: Any):
    weights = ResNet18_Weights.verify(weights)

    return _resnet(BasicBlock, [2, 2, 2, 2], weights, progress, **kwargs)


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


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def evaluate(model, criterion, data_loader, neval_batches):
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    cnt = 0
    with torch.no_grad():
        for image, target in data_loader:
            output = model(image)
            loss = criterion(output, target)
            cnt += 1
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            print('.', end = '')
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))
            if cnt >= neval_batches:
                 return top1, top5

    return top1, top5

def load_model(model_file):
    # model = MobileNetV2()
    # state_dict = torch.load(model_file, weights_only=True)
    # model.load_state_dict(state_dict)
    # model = models.resnet50(pretrained=True)
    model = quantizedresnet18(pretrained=True)
    model.to('cpu')
    return model

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')




def prepare_data_loaders(data_path):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    dataset = torchvision.datasets.ImageNet(
        data_path, split="train", transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    dataset_test = torchvision.datasets.ImageNet(
        data_path, split="val", transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=train_batch_size,
        sampler=train_sampler)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=eval_batch_size,
        sampler=test_sampler)

    return data_loader, data_loader_test





# data_path = '~/.data/imagenet'
data_path = '/root/autodl-tmp/imagenet/'
saved_model_dir = 'data/'
float_model_file = 'resnet_pretrained_float.pth'
scripted_float_model_file = 'resnet_quantization_scripted.pth'
scripted_per_tensor_quantized_model_file = 'resnet_per_tensor_quantization_scripted_quantized.pth'
scripted_per_channel_quantized_model_file = 'resnet_per_channel_quantization_scripted_quantized.pth'

train_batch_size = 30
eval_batch_size = 50

data_loader, data_loader_test = prepare_data_loaders(data_path)
criterion = nn.CrossEntropyLoss()
float_model = load_model(saved_model_dir + float_model_file).to('cpu')

# Next, we'll "fuse modules"; this can both make the model faster by saving on memory access
# while also improving numerical accuracy. While this can be used with any model, this is
# especially common with quantized models.

# print('\n Inverted Residual Block: Before fusion \n\n', float_model.features[1].conv)
float_model.eval()

# Fuses modules
# float_model.fuse_model()

# Note fusion of Conv+BN+Relu and Conv+Relu
# print('\n Inverted Residual Block: After fusion\n\n',float_model.features[1].conv)


num_eval_batches = 5

print("Size of baseline model")
print_size_of_model(float_model)

top1, top5 = evaluate(float_model, criterion, data_loader_test, neval_batches=num_eval_batches)
print('Evaluation accuracy on %d images, %2.2f'%(num_eval_batches * eval_batch_size, top1.avg))
torch.jit.save(torch.jit.script(float_model), saved_model_dir + scripted_float_model_file)
export_resnet2onnx(float_model, scripted_float_model_file)



print('============================ Post-training static quantization - 1 ================================')
num_calibration_batches = 32

myModel = load_model(saved_model_dir + float_model_file).to('cpu')
myQuantizeModel = QuantizedModel(myModel)
myQuantizeModel.eval()

# Fuse Conv, bn and relu
# myQuantizeModel.fuse_model()

# Specify quantization configuration
# Start with simple min/max range estimation and per-tensor quantization of weights
myQuantizeModel.qconfig = torch.ao.quantization.default_qconfig
print(myQuantizeModel.qconfig)
torch.ao.quantization.prepare(myQuantizeModel, inplace=True)

# Calibrate first
print('Post Training Quantization Prepare: Inserting Observers')
# print('\n Inverted Residual Block:After observer insertion \n\n', myQuantizeModel.features[1].conv)

# Calibrate with the training set
evaluate(myQuantizeModel, criterion, data_loader, neval_batches=num_calibration_batches)
print('Post Training Quantization: Calibration done')

# Convert to quantized model
torch.ao.quantization.convert(myQuantizeModel, inplace=True)
# You may see a user warning about needing to calibrate the model. This warning can be safely ignored.
# This warning occurs because not all modules are run in each model runs, so some
# modules may not be calibrated.
print('Post Training Quantization: Convert done')
# print('\n Inverted Residual Block: After fusion and quantization, note fused modules: \n\n',myQuantizeModel.features[1].conv)

print("Size of model after quantization")
print_size_of_model(myQuantizeModel)

top1, top5 = evaluate(myQuantizeModel, criterion, data_loader_test, neval_batches=num_eval_batches)
print('Evaluation accuracy on %d images, %2.2f'%(num_eval_batches * eval_batch_size, top1.avg))

torch.jit.save(torch.jit.script(myQuantizeModel), saved_model_dir + scripted_per_tensor_quantized_model_file)

# export_resnet2onnx(myQuantizeModel, scripted_per_tensor_quantized_model_file)



print('============================ Post-training static quantization - 2 ================================')

my_model_2 = load_model(saved_model_dir + float_model_file).to('cpu')
per_channel_quantized_model = QuantizedModel(my_model_2)
per_channel_quantized_model.eval()
# per_channel_quantized_model.fuse_model()
# The old 'fbgemm' is still available but 'x86' is the recommended default.
per_channel_quantized_model.qconfig = torch.ao.quantization.get_default_qconfig('x86')
print(per_channel_quantized_model.qconfig)

torch.ao.quantization.prepare(per_channel_quantized_model, inplace=True)
evaluate(per_channel_quantized_model,criterion, data_loader, num_calibration_batches)
torch.ao.quantization.convert(per_channel_quantized_model, inplace=True)

print("Size of model after quantization")
print_size_of_model(per_channel_quantized_model)
top1, top5 = evaluate(per_channel_quantized_model, criterion, data_loader_test, neval_batches=num_eval_batches)
print('Evaluation accuracy on %d images, %2.2f'%(num_eval_batches * eval_batch_size, top1.avg))
torch.jit.save(torch.jit.script(per_channel_quantized_model), saved_model_dir + scripted_per_channel_quantized_model_file)

# export_resnet2onnx(per_channel_quantized_model, scripted_per_channel_quantized_model_file)


# (base) root@autodl-container-9db611b252-b759e1c0:~/autodl-tmp/workspace/infer_system/quantization/ptq/resnet# python3 main.py 
# /root/miniconda3/lib/python3.8/site-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
#   warnings.warn(
# /root/miniconda3/lib/python3.8/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=ResNet18_Weights.IMAGENET1K_V1`. You can also use `weights=ResNet18_Weights.DEFAULT` to get the most up-to-date weights.
#   warnings.warn(msg)
# Size of baseline model
# Size (MB): 46.828697
# ....................................................................................................Evaluation accuracy on 5000 images, 87.28
# ============================ Post-training static quantization - 1 ================================
# QConfig(activation=functools.partial(<class 'torch.ao.quantization.observer.MinMaxObserver'>, quant_min=0, quant_max=127){}, weight=functools.partial(<class 'torch.ao.quantization.observer.MinMaxObserver'>, dtype=torch.qint8, qscheme=torch.per_tensor_symmetric){})
# Post Training Quantization Prepare: Inserting Observers
# ................................Post Training Quantization: Calibration done
# Post Training Quantization: Convert done
# Size of model after quantization
# Size (MB): 11.825253
# ....................................................................................................Evaluation accuracy on 5000 images, 83.20
# ============================ Post-training static quantization - 2 ================================
# QConfig(activation=functools.partial(<class 'torch.ao.quantization.observer.HistogramObserver'>, reduce_range=True){}, weight=functools.partial(<class 'torch.ao.quantization.observer.PerChannelMinMaxObserver'>, dtype=torch.qint8, qscheme=torch.per_channel_symmetric){})
# /root/miniconda3/lib/python3.8/site-packages/torch/ao/quantization/observer.py:214: UserWarning: Please use quant_min and quant_max to specify the range for observers.                     reduce_range will be deprecated in a future release of PyTorch.
#   warnings.warn(
# ................................Size of model after quantization
# Size (MB): 11.926163
# ....................................................................................................Evaluation accuracy on 5000 images, 86.90