import os
import sys
import time
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets
from torchvision.models.resnet import resnet18
import torchvision.transforms as transforms

# Set up warnings
import warnings
warnings.filterwarnings(
    action='ignore',
    category=DeprecationWarning,
    module=r'.*'
)
warnings.filterwarnings(
    action='default',
    module=r'torch.ao.quantization'
)

# Specify random seed for repeatable results
_ = torch.manual_seed(191009)


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
    """
    Computes the accuracy over the k top predictions for the specified
    values of k.
    """
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


def evaluate(model, criterion, data_loader):
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
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))
    print('')

    return top1, top5

def load_model(model_file):
    model = resnet18(pretrained=False)
    state_dict = torch.load(model_file, weights_only=True)
    model.load_state_dict(state_dict)
    model.to("cpu")
    return model

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print("Size (MB):", os.path.getsize("temp.p")/1e6)
    os.remove("temp.p")

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

data_path = '/root/autodl-tmp/imagenet/'
saved_model_dir = 'data/'
float_model_file = 'resnet18_pretrained_float.pth'

train_batch_size = 30
eval_batch_size = 50

data_loader, data_loader_test = prepare_data_loaders(data_path)
example_inputs = (next(iter(data_loader))[0])
criterion = nn.CrossEntropyLoss()
float_model = load_model(saved_model_dir + float_model_file).to("cpu")
float_model.eval()

# create another instance of the model since
# we need to keep the original model around
model_to_quantize = load_model(saved_model_dir + float_model_file).to("cpu")


# For post training quantization, we’ll need to set the model to the eval mode.
model_to_quantize.eval()

example_inputs = (torch.rand(2, 3, 224, 224),)
# for pytorch 2.5+
# exported_model = torch.export.export_for_training(model_to_quantize, example_inputs).module()

# for pytorch 2.4 and before
from torch._export import capture_pre_autograd_graph
exported_model = capture_pre_autograd_graph(model_to_quantize, example_inputs)

# or capture with dynamic dimensions
# for pytorch 2.5+
# dynamic_shapes = tuple(
#   {0: torch.export.Dim("dim")} if i == 0 else None
#   for i in range(len(example_inputs))
# )
# exported_model = torch.export.export_for_training(model_to_quantize, example_inputs, dynamic_shapes=dynamic_shapes).module()

# for pytorch 2.4 and before
# dynamic_shape API may vary as well
from torch._export import dynamic_dim
exported_model = capture_pre_autograd_graph(model_to_quantize, example_inputs, constraints=[dynamic_dim(example_inputs[0], 0)])

# Import the Backend Specific Quantizer and Configure how to Quantize the Model
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
  XNNPACKQuantizer,
  get_symmetric_quantization_config,
)
quantizer = XNNPACKQuantizer()
quantizer.set_global(get_symmetric_quantization_config())

# qconfig_opt is an optional quantization config
# can be a module type
# or torch functional op
quantizer.set_global(qconfig_opt).set_object_type(torch.nn.Conv2d, qconfig_opt).set_object_type(
    torch.nn.functional.linear, qconfig_opt).set_module_name("foo.bar", qconfig_opt)

# Prepare the Model for Post Training Quantization
# prepare_pt2e folds BatchNorm operators into preceding Conv2d operators, 
# and inserts observers in appropriate places in the model.
prepared_model = prepare_pt2e(exported_model, quantizer)
print(prepared_model.graph)


# Calibration
# The calibration function is run after the observers are inserted in the model. 
# The purpose for calibration is to run through some sample examples that is representative of the workload
# (for example a sample of the training data set) so that the observers in themodel are able to observe 
# the statistics of the Tensors and we can later use this information to calculate quantization parameters.
def calibrate(model, data_loader):
    model.eval()
    with torch.no_grad():
        for image, target in data_loader:
            model(image)
calibrate(prepared_model, data_loader_test)  # run calibration on sample data

# Convert the Calibrated Model to a Quantized Model
# convert_pt2e takes a calibrated model and produces a quantized model.
quantized_model = convert_pt2e(prepared_model)
print(quantized_model)


# Q/DQ Representation (default)
# Previous documentation for representations all quantized operators are represented as dequantize -> fp32_op -> qauntize.
def quantized_linear(x_int8, x_scale, x_zero_point, weight_int8, weight_scale, weight_zero_point, bias_fp32, output_scale, output_zero_point):
    x_fp32 = torch.ops.quantized_decomposed.dequantize_per_tensor(
             x_i8, x_scale, x_zero_point, x_quant_min, x_quant_max, torch.int8)
    weight_fp32 = torch.ops.quantized_decomposed.dequantize_per_tensor(
             weight_i8, weight_scale, weight_zero_point, weight_quant_min, weight_quant_max, torch.int8)
    weight_permuted = torch.ops.aten.permute_copy.default(weight_fp32, [1, 0]);
    out_fp32 = torch.ops.aten.addmm.default(bias_fp32, x_fp32, weight_permuted)
    out_i8 = torch.ops.quantized_decomposed.quantize_per_tensor(
    out_fp32, out_scale, out_zero_point, out_quant_min, out_quant_max, torch.int8)
    return out_i8

# Reference Quantized Model Representation
# We will have a special representation for selected ops, for example, quantized linear. 
# Other ops are represented as dq -> float32_op -> q and q/dq are decomposed into more primitive operators.
# You can get this representation by using convert_pt2e(..., use_reference_representation=True).
# Reference Quantized Pattern for quantized linear
def quantized_linear(x_int8, x_scale, x_zero_point, weight_int8, weight_scale, weight_zero_point, bias_fp32, output_scale, output_zero_point):
    x_int16 = x_int8.to(torch.int16)
    weight_int16 = weight_int8.to(torch.int16)
    acc_int32 = torch.ops.out_dtype(torch.mm, torch.int32, (x_int16 - x_zero_point), (weight_int16 - weight_zero_point))
    bias_scale = x_scale * weight_scale
    bias_int32 = out_dtype(torch.ops.aten.div.Tensor, torch.int32, bias_fp32, bias_scale)
    acc_int32 = acc_int32 + bias_int32
    acc_int32 = torch.ops.out_dtype(torch.ops.aten.mul.Scalar, torch.int32, acc_int32, x_scale * weight_scale / output_scale) + output_zero_point
    out_int8 = torch.ops.aten.clamp(acc_int32, qmin, qmax).to(torch.int8)
    return out_int8


# Checking Model Size and Accuracy Evaluation
# Baseline model size and accuracy
print("Size of baseline model")
print_size_of_model(float_model)

top1, top5 = evaluate(float_model, criterion, data_loader_test)
print("Baseline Float Model Evaluation accuracy: %2.2f, %2.2f"%(top1.avg, top5.avg))

# Quantized model size and accuracy
print("Size of model after quantization")
# export again to remove unused weights
quantized_model = torch.export.export_for_training(quantized_model, example_inputs).module()
print_size_of_model(quantized_model)

top1, top5 = evaluate(quantized_model, criterion, data_loader_test)
print("[before serilaization] Evaluation accuracy on test dataset: %2.2f, %2.2f"%(top1.avg, top5.avg))

# Save and Load Quantized Model

# 0. Store reference output, for example, inputs, and check evaluation accuracy:
example_inputs = (next(iter(data_loader))[0],)
ref = quantized_model(*example_inputs)
top1, top5 = evaluate(quantized_model, criterion, data_loader_test)
print("[before serialization] Evaluation accuracy on test dataset: %2.2f, %2.2f"%(top1.avg, top5.avg))

# 1. Export the model and Save ExportedProgram
pt2e_quantized_model_file_path = saved_model_dir + "resnet18_pt2e_quantized.pth"
# capture the model to get an ExportedProgram
quantized_ep = torch.export.export(quantized_model, example_inputs)
# use torch.export.save to save an ExportedProgram
torch.export.save(quantized_ep, pt2e_quantized_model_file_path)


# 2. Load the saved ExportedProgram
loaded_quantized_ep = torch.export.load(pt2e_quantized_model_file_path)
loaded_quantized_model = loaded_quantized_ep.module()

# 3. Check results for example inputs and check evaluation accuracy again:
res = loaded_quantized_model(*example_inputs)
print("diff:", ref - res)

top1, top5 = evaluate(loaded_quantized_model, criterion, data_loader_test)
print("[after serialization/deserialization] Evaluation accuracy on test dataset: %2.2f, %2.2f"%(top1.avg, top5.avg))