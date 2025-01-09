import os
import sys
import time
import numpy as np

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.transforms._presets import ImageClassification
from torchvision.utils import _log_api_usage_once
from torchvision.models.resnet import Bottleneck
from torchvision.models.resnet import conv3x3

from functools import partial
from typing import Any, Callable, List, Optional, Type, Union


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

def export_resnet2onnx(model, name):
    model.to(torch.device('cpu'))
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224) # 创建一个示例输入
    dummy_input.to(torch.device('cpu'))
    torch.onnx.export(model, dummy_input, name + ".onnx", opset_version=11) # 将模型导出为 ONNX 格式


def fuse_resnet18(model):
    torch.quantization.fuse_modules(model, [["conv1", "bn1", "relu"]], inplace=True)
    for module_name, module in model.named_children():
        if "layer" in module_name:
            for basic_block_name, basic_block in module.named_children():
                torch.quantization.fuse_modules(basic_block, [["conv1", "bn1"], ["conv2", "bn2"]], inplace=True)
                for sub_block_name, sub_block in basic_block.named_children():
                    if sub_block_name == "downsample":
                        torch.quantization.fuse_modules(sub_block, [["0", "1"]], inplace=True)    


# data_path = '~/.data/imagenet'
data_path = '/root/autodl-tmp/imagenet/'
saved_model_dir = 'data/'
float_model_file = 'resnet_pretrained_float.pth'
scripted_float_model_file = 'resnet_scripted.pth'
scripted_per_tensor_quantized_model_file = 'resnet_per_tensor_quantization_scripted_quantized.pth'
scripted_ptq_dynamic_model_file = 'resnet_ptq_dynamic_scripted_quantized.pth'
scripted_qat_model_file = 'resnet_qat_scripted_quantized.pth'

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
num_eval_batches = 30

print("Size of baseline model")
print_size_of_model(float_model)

top1, top5 = evaluate(float_model, criterion, data_loader_test, neval_batches=num_eval_batches)
print('Evaluation accuracy on %d images, %2.2f'%(num_eval_batches * eval_batch_size, top1.avg))
torch.jit.save(torch.jit.script(float_model), saved_model_dir + scripted_float_model_file)
export_resnet2onnx(float_model, saved_model_dir + scripted_float_model_file)


print('============================ Quantization-aware training ================================')
def train_one_epoch(model, criterion, optimizer, data_loader, device, ntrain_batches):
    model.train()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    avgloss = AverageMeter('Loss', '1.5f')

    cnt = 0
    for image, target in data_loader:
        start_time = time.time()
        print('.', end = '')
        cnt += 1
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], image.size(0))
        top5.update(acc5[0], image.size(0))
        avgloss.update(loss, image.size(0))
        if cnt >= ntrain_batches:
            print('Loss', avgloss.avg)

            print('Training: * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                  .format(top1=top1, top5=top5))
            return

    print('Full imagenet train set:  * Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
          .format(top1=top1, top5=top5))
    return


qat_model = load_model(saved_model_dir + float_model_file)
qat_model.eval()
fuse_resnet18(qat_model)

qat_model = QuantizedModel(qat_model)

optimizer = torch.optim.SGD(qat_model.parameters(), lr = 0.0001)
# optimizer = torch.optim.SGD(qat_model.parameters(),
#                       lr=0.1,
#                       momentum=0.9,
#                       weight_decay=1e-4)
# The old 'fbgemm' is still available but 'x86' is the recommended default.
qat_model.qconfig = torch.ao.quantization.get_default_qat_qconfig('x86')

qat_model.train()
# prepare_qat performs the “fake quantization”, preparing the model for quantization-aware training
torch.ao.quantization.prepare_qat(qat_model, inplace=True)
# print('Inverted Residual Block: After preparation for QAT, note fake-quantization modules \n',qat_model.features[1].conv)



num_train_batches = 30

# QAT takes time and one needs to train over a few epochs.
# Train and check accuracy after each epoch
for nepoch in range(20):
    train_one_epoch(qat_model, criterion, optimizer, data_loader, torch.device('cpu'), num_train_batches)
    if nepoch > 3:
        # Freeze quantizer parameters
        qat_model.apply(torch.ao.quantization.disable_observer)
    if nepoch > 2:
        # Freeze batch norm mean and variance estimates
        qat_model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

    # Check the accuracy after each epoch
    quantized_model = torch.ao.quantization.convert(qat_model.eval(), inplace=False)
    quantized_model.eval()
    top1, top5 = evaluate(quantized_model,criterion, data_loader_test, neval_batches=num_eval_batches)
    print('Epoch %d :Evaluation accuracy on %d images, %2.2f'%(nepoch, num_eval_batches * eval_batch_size, top1.avg))

print("Size of model after qat")
print_size_of_model(quantized_model)
torch.jit.save(torch.jit.script(quantized_model), saved_model_dir + scripted_qat_model_file)
# export_resnet2onnx(quantized_model, saved_model_dir + scripted_qat_model_file)






# (base) root@autodl-container-9db611b252-b759e1c0:~/autodl-tmp/workspace/infer_system/pytorch_quantization/eager_mode/qat_static# python3 my_resnet.py 
# /root/miniconda3/lib/python3.8/site-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
#   warnings.warn(
# /root/miniconda3/lib/python3.8/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=ResNet18_Weights.IMAGENET1K_V1`. You can also use `weights=ResNet18_Weights.DEFAULT` to get the most up-to-date weights.
#   warnings.warn(msg)
# Size of baseline model
# Size (MB): 46.828697
# ..............................Evaluation accuracy on 1500 images, 88.53
# ============= Diagnostic Run torch.onnx.export version 2.0.0+cu118 =============
# verbose: False, log level: Level.ERROR
# ======================= 0 NONE 0 NOTE 0 WARNING 0 ERROR ========================

# ============================ Quantization-aware training ================================
# /root/miniconda3/lib/python3.8/site-packages/torch/ao/quantization/observer.py:214: UserWarning: Please use quant_min and quant_max to specify the range for observers.                     reduce_range will be deprecated in a future release of PyTorch.
#   warnings.warn(
# ..............................Loss tensor(6.3789, grad_fn=<DivBackward0>)
# Training: * Acc@1 32.778 Acc@5 43.444
# ..............................Epoch 0 :Evaluation accuracy on 1500 images, 77.40
# ..............................Loss tensor(5.3793, grad_fn=<DivBackward0>)
# Training: * Acc@1 27.667 Acc@5 39.556
# ..............................Epoch 1 :Evaluation accuracy on 1500 images, 75.47
# ..............................Loss tensor(5.1885, grad_fn=<DivBackward0>)
# Training: * Acc@1 29.778 Acc@5 41.333
# ..............................Epoch 2 :Evaluation accuracy on 1500 images, 75.07
# ..............................Loss tensor(4.9600, grad_fn=<DivBackward0>)
# Training: * Acc@1 29.667 Acc@5 43.667
# ..............................Epoch 3 :Evaluation accuracy on 1500 images, 79.47
# ..............................Loss tensor(4.9998, grad_fn=<DivBackward0>)
# Training: * Acc@1 30.889 Acc@5 43.111
# ..............................Epoch 4 :Evaluation accuracy on 1500 images, 80.80
# ..............................Loss tensor(4.9766, grad_fn=<DivBackward0>)
# Training: * Acc@1 31.444 Acc@5 42.000
# ..............................Epoch 5 :Evaluation accuracy on 1500 images, 80.20
# ..............................Loss tensor(4.8145, grad_fn=<DivBackward0>)
# Training: * Acc@1 29.889 Acc@5 42.333
# ..............................Epoch 6 :Evaluation accuracy on 1500 images, 81.80
# ..............................Loss tensor(4.8383, grad_fn=<DivBackward0>)
# Training: * Acc@1 31.111 Acc@5 43.222
# ..............................Epoch 7 :Evaluation accuracy on 1500 images, 81.53
# ..............................Loss tensor(4.8142, grad_fn=<DivBackward0>)
# Training: * Acc@1 32.444 Acc@5 42.333
# ..............................Epoch 8 :Evaluation accuracy on 1500 images, 82.60
# ..............................Loss tensor(4.6355, grad_fn=<DivBackward0>)
# Training: * Acc@1 35.556 Acc@5 45.444
# ..............................Epoch 9 :Evaluation accuracy on 1500 images, 82.40
# ..............................Loss tensor(4.7931, grad_fn=<DivBackward0>)
# Training: * Acc@1 30.778 Acc@5 43.333
# ..............................Epoch 10 :Evaluation accuracy on 1500 images, 83.33
# ..............................Loss tensor(4.7084, grad_fn=<DivBackward0>)
# Training: * Acc@1 32.444 Acc@5 43.778
# ..............................Epoch 11 :Evaluation accuracy on 1500 images, 83.67
# ..............................Loss tensor(4.7679, grad_fn=<DivBackward0>)
# Training: * Acc@1 31.111 Acc@5 44.556
# ..............................Epoch 12 :Evaluation accuracy on 1500 images, 83.73
# ..............................Loss tensor(4.6223, grad_fn=<DivBackward0>)
# Training: * Acc@1 34.111 Acc@5 45.222
# ..............................Epoch 13 :Evaluation accuracy on 1500 images, 84.53
# ..............................Loss tensor(4.4814, grad_fn=<DivBackward0>)
# Training: * Acc@1 34.556 Acc@5 46.889
# ..............................Epoch 14 :Evaluation accuracy on 1500 images, 84.27
# ..............................Loss tensor(4.5657, grad_fn=<DivBackward0>)
# Training: * Acc@1 32.556 Acc@5 45.444
# ..............................Epoch 15 :Evaluation accuracy on 1500 images, 84.00
# ..............................Loss tensor(4.8448, grad_fn=<DivBackward0>)
# Training: * Acc@1 30.333 Acc@5 40.222
# ..............................Epoch 16 :Evaluation accuracy on 1500 images, 82.53
# ..............................Loss tensor(4.6029, grad_fn=<DivBackward0>)
# Training: * Acc@1 33.889 Acc@5 45.556
# ..............................Epoch 17 :Evaluation accuracy on 1500 images, 83.07
# ..............................Loss tensor(4.4885, grad_fn=<DivBackward0>)
# Training: * Acc@1 33.111 Acc@5 45.444
# ..............................Epoch 18 :Evaluation accuracy on 1500 images, 83.27
# ..............................Loss tensor(4.6164, grad_fn=<DivBackward0>)
# Training: * Acc@1 32.778 Acc@5 44.667
# ..............................Epoch 19 :Evaluation accuracy on 1500 images, 84.07
# Size of model after qat
# Size (MB): 11.835755
