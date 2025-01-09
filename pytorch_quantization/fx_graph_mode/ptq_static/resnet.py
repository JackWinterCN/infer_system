import os
import sys
import time
import numpy as np

import torch
from torch.ao.quantization import get_default_qconfig, QConfigMapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx, fuse_fx
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


def evaluate(model, criterion, data_loader, neval_batches = 30):
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
    print('')

    return top1, top5

def load_model(model_file):
    model = resnet18(pretrained=False)
    state_dict = torch.load(model_file, weights_only=True)
    model.load_state_dict(state_dict)
    model.to("cpu")
    return model

def print_size_of_model(model):
    if isinstance(model, torch.jit.RecursiveScriptModule):
        torch.jit.save(model, "temp.p")
    else:
        torch.jit.save(torch.jit.script(model), "temp.p")
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


top1, top5 = evaluate(float_model, criterion, data_loader_test)
print("[before quantization] Evaluation accuracy on test dataset: %2.2f, %2.2f"%(top1.avg, top5.avg))


# create another instance of the model since
# we need to keep the original model around
model_to_quantize = load_model(saved_model_dir + float_model_file).to("cpu")

model_to_quantize.eval()



# Specify how to quantize the model with QConfigMapping
# qconfig_mapping = QConfigMapping.set_global(default_qconfig)

# qconfig is just a named tuple of the observers for activation and weight. 
# QConfigMapping contains mapping information from ops to qconfigs

# qconfig_mapping = (QConfigMapping()
#     .set_global(qconfig_opt)  # qconfig_opt is an optional qconfig, either a valid qconfig or None
#     .set_object_type(torch.nn.Conv2d, qconfig_opt)  # can be a callable...
#     .set_object_type("reshape", qconfig_opt)  # ...or a string of the method
#     .set_module_name_regex("foo.*bar.*conv[0-9]+", qconfig_opt) # matched in order, first match takes precedence
#     .set_module_name("foo.bar", qconfig_opt)
#     .set_module_name_object_type_order()
# )
    # priority (in increasing order): global, object_type, module_name_regex, module_name
    # qconfig == None means fusion and quantization should be skipped for anything
    # matching the rule (unless a higher priority match is found)
    
# The old 'fbgemm' is still available but 'x86' is the recommended default.
qconfig = get_default_qconfig("x86")
qconfig_mapping = QConfigMapping().set_global(qconfig)

# Prepare the Model for Post Training Static Quantization
# prepare_fx folds BatchNorm modules into previous Conv2d modules, 
# and insert observers in appropriate places in the model.
prepared_model = prepare_fx(model_to_quantize, qconfig_mapping, example_inputs)
# tracer.trace(model_to_quantize) -> Graph
# graph():
#     %x : torch.Tensor [#users=1] = placeholder[target=x]
#     %conv1 : [#users=1] = call_module[target=conv1](args = (%x,), kwargs = {})
#     %bn1 : [#users=1] = call_module[target=bn1](args = (%conv1,), kwargs = {})
#     %relu : [#users=1] = call_module[target=relu](args = (%bn1,), kwargs = {})
#     %maxpool : [#users=2] = call_module[target=maxpool](args = (%relu,), kwargs = {})
#     %layer1_0_conv1 : [#users=1] = call_module[target=layer1.0.conv1](args = (%maxpool,), kwargs = {})
#     %layer1_0_bn1 : [#users=1] = call_module[target=layer1.0.bn1](args = (%layer1_0_conv1,), kwargs = {})
#     %layer1_0_relu : [#users=1] = call_module[target=layer1.0.relu](args = (%layer1_0_bn1,), kwargs = {})
#     %layer1_0_conv2 : [#users=1] = call_module[target=layer1.0.conv2](args = (%layer1_0_relu,), kwargs = {})
#     %layer1_0_bn2 : [#users=1] = call_module[target=layer1.0.bn2](args = (%layer1_0_conv2,), kwargs = {})
#     %add : [#users=1] = call_function[target=operator.add](args = (%layer1_0_bn2, %maxpool), kwargs = {})
#     %layer1_0_relu_1 : [#users=2] = call_module[target=layer1.0.relu](args = (%add,), kwargs = {})
#     %layer1_1_conv1 : [#users=1] = call_module[target=layer1.1.conv1](args = (%layer1_0_relu_1,), kwargs = {})
#     %layer1_1_bn1 : [#users=1] = call_module[target=layer1.1.bn1](args = (%layer1_1_conv1,), kwargs = {})
#     %layer1_1_relu : [#users=1] = call_module[target=layer1.1.relu](args = (%layer1_1_bn1,), kwargs = {})
#     %layer1_1_conv2 : [#users=1] = call_module[target=layer1.1.conv2](args = (%layer1_1_relu,), kwargs = {})
#     %layer1_1_bn2 : [#users=1] = call_module[target=layer1.1.bn2](args = (%layer1_1_conv2,), kwargs = {})
#     %add_1 : [#users=1] = call_function[target=operator.add](args = (%layer1_1_bn2, %layer1_0_relu_1), kwargs = {})
#     %layer1_1_relu_1 : [#users=2] = call_module[target=layer1.1.relu](args = (%add_1,), kwargs = {})
#     %layer2_0_conv1 : [#users=1] = call_module[target=layer2.0.conv1](args = (%layer1_1_relu_1,), kwargs = {})
#     %layer2_0_bn1 : [#users=1] = call_module[target=layer2.0.bn1](args = (%layer2_0_conv1,), kwargs = {})
#     %layer2_0_relu : [#users=1] = call_module[target=layer2.0.relu](args = (%layer2_0_bn1,), kwargs = {})
#     %layer2_0_conv2 : [#users=1] = call_module[target=layer2.0.conv2](args = (%layer2_0_relu,), kwargs = {})
#     %layer2_0_bn2 : [#users=1] = call_module[target=layer2.0.bn2](args = (%layer2_0_conv2,), kwargs = {})
#     %layer2_0_downsample_0 : [#users=1] = call_module[target=layer2.0.downsample.0](args = (%layer1_1_relu_1,), kwargs = {})
#     %layer2_0_downsample_1 : [#users=1] = call_module[target=layer2.0.downsample.1](args = (%layer2_0_downsample_0,), kwargs = {})
#     %add_2 : [#users=1] = call_function[target=operator.add](args = (%layer2_0_bn2, %layer2_0_downsample_1), kwargs = {})
#     %layer2_0_relu_1 : [#users=2] = call_module[target=layer2.0.relu](args = (%add_2,), kwargs = {})
#     %layer2_1_conv1 : [#users=1] = call_module[target=layer2.1.conv1](args = (%layer2_0_relu_1,), kwargs = {})
#     %layer2_1_bn1 : [#users=1] = call_module[target=layer2.1.bn1](args = (%layer2_1_conv1,), kwargs = {})
#     %layer2_1_relu : [#users=1] = call_module[target=layer2.1.relu](args = (%layer2_1_bn1,), kwargs = {})
#     %layer2_1_conv2 : [#users=1] = call_module[target=layer2.1.conv2](args = (%layer2_1_relu,), kwargs = {})
#     %layer2_1_bn2 : [#users=1] = call_module[target=layer2.1.bn2](args = (%layer2_1_conv2,), kwargs = {})
#     %add_3 : [#users=1] = call_function[target=operator.add](args = (%layer2_1_bn2, %layer2_0_relu_1), kwargs = {})
#     %layer2_1_relu_1 : [#users=2] = call_module[target=layer2.1.relu](args = (%add_3,), kwargs = {})
#     %layer3_0_conv1 : [#users=1] = call_module[target=layer3.0.conv1](args = (%layer2_1_relu_1,), kwargs = {})
#     %layer3_0_bn1 : [#users=1] = call_module[target=layer3.0.bn1](args = (%layer3_0_conv1,), kwargs = {})
#     %layer3_0_relu : [#users=1] = call_module[target=layer3.0.relu](args = (%layer3_0_bn1,), kwargs = {})
#     %layer3_0_conv2 : [#users=1] = call_module[target=layer3.0.conv2](args = (%layer3_0_relu,), kwargs = {})
#     %layer3_0_bn2 : [#users=1] = call_module[target=layer3.0.bn2](args = (%layer3_0_conv2,), kwargs = {})
#     %layer3_0_downsample_0 : [#users=1] = call_module[target=layer3.0.downsample.0](args = (%layer2_1_relu_1,), kwargs = {})
#     %layer3_0_downsample_1 : [#users=1] = call_module[target=layer3.0.downsample.1](args = (%layer3_0_downsample_0,), kwargs = {})
#     %add_4 : [#users=1] = call_function[target=operator.add](args = (%layer3_0_bn2, %layer3_0_downsample_1), kwargs = {})
#     %layer3_0_relu_1 : [#users=2] = call_module[target=layer3.0.relu](args = (%add_4,), kwargs = {})
#     %layer3_1_conv1 : [#users=1] = call_module[target=layer3.1.conv1](args = (%layer3_0_relu_1,), kwargs = {})
#     %layer3_1_bn1 : [#users=1] = call_module[target=layer3.1.bn1](args = (%layer3_1_conv1,), kwargs = {})
#     %layer3_1_relu : [#users=1] = call_module[target=layer3.1.relu](args = (%layer3_1_bn1,), kwargs = {})
#     %layer3_1_conv2 : [#users=1] = call_module[target=layer3.1.conv2](args = (%layer3_1_relu,), kwargs = {})
#     %layer3_1_bn2 : [#users=1] = call_module[target=layer3.1.bn2](args = (%layer3_1_conv2,), kwargs = {})
#     %add_5 : [#users=1] = call_function[target=operator.add](args = (%layer3_1_bn2, %layer3_0_relu_1), kwargs = {})
#     %layer3_1_relu_1 : [#users=2] = call_module[target=layer3.1.relu](args = (%add_5,), kwargs = {})
#     %layer4_0_conv1 : [#users=1] = call_module[target=layer4.0.conv1](args = (%layer3_1_relu_1,), kwargs = {})
#     %layer4_0_bn1 : [#users=1] = call_module[target=layer4.0.bn1](args = (%layer4_0_conv1,), kwargs = {})
#     %layer4_0_relu : [#users=1] = call_module[target=layer4.0.relu](args = (%layer4_0_bn1,), kwargs = {})
#     %layer4_0_conv2 : [#users=1] = call_module[target=layer4.0.conv2](args = (%layer4_0_relu,), kwargs = {})
#     %layer4_0_bn2 : [#users=1] = call_module[target=layer4.0.bn2](args = (%layer4_0_conv2,), kwargs = {})
#     %layer4_0_downsample_0 : [#users=1] = call_module[target=layer4.0.downsample.0](args = (%layer3_1_relu_1,), kwargs = {})
#     %layer4_0_downsample_1 : [#users=1] = call_module[target=layer4.0.downsample.1](args = (%layer4_0_downsample_0,), kwargs = {})
#     %add_6 : [#users=1] = call_function[target=operator.add](args = (%layer4_0_bn2, %layer4_0_downsample_1), kwargs = {})
#     %layer4_0_relu_1 : [#users=2] = call_module[target=layer4.0.relu](args = (%add_6,), kwargs = {})
#     %layer4_1_conv1 : [#users=1] = call_module[target=layer4.1.conv1](args = (%layer4_0_relu_1,), kwargs = {})
#     %layer4_1_bn1 : [#users=1] = call_module[target=layer4.1.bn1](args = (%layer4_1_conv1,), kwargs = {})
#     %layer4_1_relu : [#users=1] = call_module[target=layer4.1.relu](args = (%layer4_1_bn1,), kwargs = {})
#     %layer4_1_conv2 : [#users=1] = call_module[target=layer4.1.conv2](args = (%layer4_1_relu,), kwargs = {})
#     %layer4_1_bn2 : [#users=1] = call_module[target=layer4.1.bn2](args = (%layer4_1_conv2,), kwargs = {})
#     %add_7 : [#users=1] = call_function[target=operator.add](args = (%layer4_1_bn2, %layer4_0_relu_1), kwargs = {})
#     %layer4_1_relu_1 : [#users=1] = call_module[target=layer4.1.relu](args = (%add_7,), kwargs = {})
#     %avgpool : [#users=1] = call_module[target=avgpool](args = (%layer4_1_relu_1,), kwargs = {})
#     %flatten : [#users=1] = call_function[target=torch.flatten](args = (%avgpool, 1), kwargs = {})
#     %fc : [#users=1] = call_module[target=fc](args = (%flatten,), kwargs = {})
#     return fc


print(prepared_model.graph)

# Calibration function is run after the observers are inserted in the model.
def calibrate(model, data_loader, neval_batches = 30):
    model.eval()
    cnt = 0
    with torch.no_grad():
        for image, target in data_loader:
            model(image)
            cnt += 1
            if cnt >= neval_batches:
                return
calibrate(prepared_model, data_loader_test)  # run calibration on sample data

# convert_fx takes a calibrated model and produces a quantized model.
quantized_model = convert_fx(prepared_model)
print(quantized_model)


# We can now print the size and accuracy of the quantized model.
print("Size of model before quantization")
print_size_of_model(float_model)
print("Size of model after quantization")
print_size_of_model(quantized_model)
top1, top5 = evaluate(quantized_model, criterion, data_loader_test)
print("[before serilaization] Evaluation accuracy on test dataset: %2.2f, %2.2f"%(top1.avg, top5.avg))

fx_graph_mode_model_file_path = saved_model_dir + "resnet18_fx_graph_mode_quantized.pth"

# this does not run due to some erros loading convrelu module:
# ModuleAttributeError: 'ConvReLU2d' object has no attribute '_modules'
# save the whole model directly
# torch.save(quantized_model, fx_graph_mode_model_file_path)
# loaded_quantized_model = torch.load(fx_graph_mode_model_file_path, weights_only=False)

# save with state_dict
# torch.save(quantized_model.state_dict(), fx_graph_mode_model_file_path)
# import copy
# model_to_quantize = copy.deepcopy(float_model)
# prepared_model = prepare_fx(model_to_quantize, {"": qconfig})
# loaded_quantized_model = convert_fx(prepared_model)
# loaded_quantized_model.load_state_dict(torch.load(fx_graph_mode_model_file_path), weights_only=True)

# save with script
torch.jit.save(torch.jit.script(quantized_model), fx_graph_mode_model_file_path)
loaded_quantized_model = torch.jit.load(fx_graph_mode_model_file_path)

top1, top5 = evaluate(loaded_quantized_model, criterion, data_loader_test)
print("[after serialization/deserialization] Evaluation accuracy on test dataset: %2.2f, %2.2f"%(top1.avg, top5.avg))


# Debugging Quantized Model
# We can also print the weight for quantized a non-quantized convolution op to see the difference, 
# we’ll first call fuse explicitly to fuse the convolution and batch norm in the model: 
# Note that fuse_fx only works in eval mode.
fused = fuse_fx(float_model)

conv1_weight_after_fuse = fused.conv1[0].weight[0]
conv1_weight_after_quant = quantized_model.conv1.weight().dequantize()[0]

print(torch.max(abs(conv1_weight_after_fuse - conv1_weight_after_quant)))


# Comparison with Baseline Float Model and Eager Mode Quantization
print('============ Comparison with Baseline Float Model and Eager Mode Quantization =================')

scripted_float_model_file = "resnet18_scripted.pth"

print("Size of baseline model")
print_size_of_model(float_model)

top1, top5 = evaluate(float_model, criterion, data_loader_test)
print("Baseline Float Model Evaluation accuracy: %2.2f, %2.2f"%(top1.avg, top5.avg))
torch.jit.save(torch.jit.script(float_model), saved_model_dir + scripted_float_model_file)

# we compare the model quantized with FX graph mode quantization with the model quantized in eager mode.
# FX graph mode and eager mode produce very similar quantized models, 
# so the expectation is that the accuracy and speedup are similar as well.
print("Size of Fx graph mode quantized model")
print_size_of_model(quantized_model)
top1, top5 = evaluate(quantized_model, criterion, data_loader_test)
print("FX graph mode quantized model Evaluation accuracy on test dataset: %2.2f, %2.2f"%(top1.avg, top5.avg))

from torchvision.models.quantization.resnet import resnet18
eager_quantized_model = resnet18(pretrained=True, quantize=True).eval()
print("Size of eager mode quantized model")
eager_quantized_model = torch.jit.script(eager_quantized_model)
print_size_of_model(eager_quantized_model)
top1, top5 = evaluate(eager_quantized_model, criterion, data_loader_test)
print("eager mode quantized model Evaluation accuracy on test dataset: %2.2f, %2.2f"%(top1.avg, top5.avg))
eager_mode_model_file = "resnet18_eager_mode_quantized.pth"
torch.jit.save(eager_quantized_model, saved_model_dir + eager_mode_model_file)

###################################################### output #####################################################

# (base) root@autodl-container-9db611b252-b759e1c0:~/autodl-tmp/workspace/infer_system/pytorch_quantization/fx_graph_mode/ptq_static# python3 resnet.py 
# /root/miniconda3/lib/python3.8/site-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
#   warnings.warn(
# /root/miniconda3/lib/python3.8/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=None`.
#   warnings.warn(msg)
# /root/miniconda3/lib/python3.8/site-packages/torch/_utils.py:776: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
#   return self.fget.__get__(instance, owner)()
# ..............................[before quantization] Evaluation accuracy on test dataset: 88.53, 97.60
# /root/miniconda3/lib/python3.8/site-packages/torch/ao/quantization/observer.py:214: UserWarning: Please use quant_min and quant_max to specify the range for observers.                     reduce_range will be deprecated in a future release of PyTorch.
#   warnings.warn(
# graph():
#     %x : torch.Tensor [#users=1] = placeholder[target=x]
#     %activation_post_process_0 : [#users=1] = call_module[target=activation_post_process_0](args = (%x,), kwargs = {})
#     %conv1 : [#users=1] = call_module[target=conv1](args = (%activation_post_process_0,), kwargs = {})
#     %activation_post_process_1 : [#users=1] = call_module[target=activation_post_process_1](args = (%conv1,), kwargs = {})
#     %maxpool : [#users=1] = call_module[target=maxpool](args = (%activation_post_process_1,), kwargs = {})
#     %activation_post_process_2 : [#users=2] = call_module[target=activation_post_process_2](args = (%maxpool,), kwargs = {})
#     %layer1_0_conv1 : [#users=1] = call_module[target=layer1.0.conv1](args = (%activation_post_process_2,), kwargs = {})
#     %activation_post_process_3 : [#users=1] = call_module[target=activation_post_process_3](args = (%layer1_0_conv1,), kwargs = {})
#     %layer1_0_conv2 : [#users=1] = call_module[target=layer1.0.conv2](args = (%activation_post_process_3,), kwargs = {})
#     %activation_post_process_4 : [#users=1] = call_module[target=activation_post_process_4](args = (%layer1_0_conv2,), kwargs = {})
#     %add : [#users=1] = call_function[target=operator.add](args = (%activation_post_process_4, %activation_post_process_2), kwargs = {})
#     %layer1_0_relu_1 : [#users=1] = call_module[target=layer1.0.relu](args = (%add,), kwargs = {})
#     %activation_post_process_5 : [#users=2] = call_module[target=activation_post_process_5](args = (%layer1_0_relu_1,), kwargs = {})
#     %layer1_1_conv1 : [#users=1] = call_module[target=layer1.1.conv1](args = (%activation_post_process_5,), kwargs = {})
#     %activation_post_process_6 : [#users=1] = call_module[target=activation_post_process_6](args = (%layer1_1_conv1,), kwargs = {})
#     %layer1_1_conv2 : [#users=1] = call_module[target=layer1.1.conv2](args = (%activation_post_process_6,), kwargs = {})
#     %activation_post_process_7 : [#users=1] = call_module[target=activation_post_process_7](args = (%layer1_1_conv2,), kwargs = {})
#     %add_1 : [#users=1] = call_function[target=operator.add](args = (%activation_post_process_7, %activation_post_process_5), kwargs = {})
#     %layer1_1_relu_1 : [#users=1] = call_module[target=layer1.1.relu](args = (%add_1,), kwargs = {})
#     %activation_post_process_8 : [#users=2] = call_module[target=activation_post_process_8](args = (%layer1_1_relu_1,), kwargs = {})
#     %layer2_0_conv1 : [#users=1] = call_module[target=layer2.0.conv1](args = (%activation_post_process_8,), kwargs = {})
#     %activation_post_process_9 : [#users=1] = call_module[target=activation_post_process_9](args = (%layer2_0_conv1,), kwargs = {})
#     %layer2_0_conv2 : [#users=1] = call_module[target=layer2.0.conv2](args = (%activation_post_process_9,), kwargs = {})
#     %activation_post_process_10 : [#users=1] = call_module[target=activation_post_process_10](args = (%layer2_0_conv2,), kwargs = {})
#     %layer2_0_downsample_0 : [#users=1] = call_module[target=layer2.0.downsample.0](args = (%activation_post_process_8,), kwargs = {})
#     %activation_post_process_11 : [#users=1] = call_module[target=activation_post_process_11](args = (%layer2_0_downsample_0,), kwargs = {})
#     %add_2 : [#users=1] = call_function[target=operator.add](args = (%activation_post_process_10, %activation_post_process_11), kwargs = {})
#     %layer2_0_relu_1 : [#users=1] = call_module[target=layer2.0.relu](args = (%add_2,), kwargs = {})
#     %activation_post_process_12 : [#users=2] = call_module[target=activation_post_process_12](args = (%layer2_0_relu_1,), kwargs = {})
#     %layer2_1_conv1 : [#users=1] = call_module[target=layer2.1.conv1](args = (%activation_post_process_12,), kwargs = {})
#     %activation_post_process_13 : [#users=1] = call_module[target=activation_post_process_13](args = (%layer2_1_conv1,), kwargs = {})
#     %layer2_1_conv2 : [#users=1] = call_module[target=layer2.1.conv2](args = (%activation_post_process_13,), kwargs = {})
#     %activation_post_process_14 : [#users=1] = call_module[target=activation_post_process_14](args = (%layer2_1_conv2,), kwargs = {})
#     %add_3 : [#users=1] = call_function[target=operator.add](args = (%activation_post_process_14, %activation_post_process_12), kwargs = {})
#     %layer2_1_relu_1 : [#users=1] = call_module[target=layer2.1.relu](args = (%add_3,), kwargs = {})
#     %activation_post_process_15 : [#users=2] = call_module[target=activation_post_process_15](args = (%layer2_1_relu_1,), kwargs = {})
#     %layer3_0_conv1 : [#users=1] = call_module[target=layer3.0.conv1](args = (%activation_post_process_15,), kwargs = {})
#     %activation_post_process_16 : [#users=1] = call_module[target=activation_post_process_16](args = (%layer3_0_conv1,), kwargs = {})
#     %layer3_0_conv2 : [#users=1] = call_module[target=layer3.0.conv2](args = (%activation_post_process_16,), kwargs = {})
#     %activation_post_process_17 : [#users=1] = call_module[target=activation_post_process_17](args = (%layer3_0_conv2,), kwargs = {})
#     %layer3_0_downsample_0 : [#users=1] = call_module[target=layer3.0.downsample.0](args = (%activation_post_process_15,), kwargs = {})
#     %activation_post_process_18 : [#users=1] = call_module[target=activation_post_process_18](args = (%layer3_0_downsample_0,), kwargs = {})
#     %add_4 : [#users=1] = call_function[target=operator.add](args = (%activation_post_process_17, %activation_post_process_18), kwargs = {})
#     %layer3_0_relu_1 : [#users=1] = call_module[target=layer3.0.relu](args = (%add_4,), kwargs = {})
#     %activation_post_process_19 : [#users=2] = call_module[target=activation_post_process_19](args = (%layer3_0_relu_1,), kwargs = {})
#     %layer3_1_conv1 : [#users=1] = call_module[target=layer3.1.conv1](args = (%activation_post_process_19,), kwargs = {})
#     %activation_post_process_20 : [#users=1] = call_module[target=activation_post_process_20](args = (%layer3_1_conv1,), kwargs = {})
#     %layer3_1_conv2 : [#users=1] = call_module[target=layer3.1.conv2](args = (%activation_post_process_20,), kwargs = {})
#     %activation_post_process_21 : [#users=1] = call_module[target=activation_post_process_21](args = (%layer3_1_conv2,), kwargs = {})
#     %add_5 : [#users=1] = call_function[target=operator.add](args = (%activation_post_process_21, %activation_post_process_19), kwargs = {})
#     %layer3_1_relu_1 : [#users=1] = call_module[target=layer3.1.relu](args = (%add_5,), kwargs = {})
#     %activation_post_process_22 : [#users=2] = call_module[target=activation_post_process_22](args = (%layer3_1_relu_1,), kwargs = {})
#     %layer4_0_conv1 : [#users=1] = call_module[target=layer4.0.conv1](args = (%activation_post_process_22,), kwargs = {})
#     %activation_post_process_23 : [#users=1] = call_module[target=activation_post_process_23](args = (%layer4_0_conv1,), kwargs = {})
#     %layer4_0_conv2 : [#users=1] = call_module[target=layer4.0.conv2](args = (%activation_post_process_23,), kwargs = {})
#     %activation_post_process_24 : [#users=1] = call_module[target=activation_post_process_24](args = (%layer4_0_conv2,), kwargs = {})
#     %layer4_0_downsample_0 : [#users=1] = call_module[target=layer4.0.downsample.0](args = (%activation_post_process_22,), kwargs = {})
#     %activation_post_process_25 : [#users=1] = call_module[target=activation_post_process_25](args = (%layer4_0_downsample_0,), kwargs = {})
#     %add_6 : [#users=1] = call_function[target=operator.add](args = (%activation_post_process_24, %activation_post_process_25), kwargs = {})
#     %layer4_0_relu_1 : [#users=1] = call_module[target=layer4.0.relu](args = (%add_6,), kwargs = {})
#     %activation_post_process_26 : [#users=2] = call_module[target=activation_post_process_26](args = (%layer4_0_relu_1,), kwargs = {})
#     %layer4_1_conv1 : [#users=1] = call_module[target=layer4.1.conv1](args = (%activation_post_process_26,), kwargs = {})
#     %activation_post_process_27 : [#users=1] = call_module[target=activation_post_process_27](args = (%layer4_1_conv1,), kwargs = {})
#     %layer4_1_conv2 : [#users=1] = call_module[target=layer4.1.conv2](args = (%activation_post_process_27,), kwargs = {})
#     %activation_post_process_28 : [#users=1] = call_module[target=activation_post_process_28](args = (%layer4_1_conv2,), kwargs = {})
#     %add_7 : [#users=1] = call_function[target=operator.add](args = (%activation_post_process_28, %activation_post_process_26), kwargs = {})
#     %layer4_1_relu_1 : [#users=1] = call_module[target=layer4.1.relu](args = (%add_7,), kwargs = {})
#     %activation_post_process_29 : [#users=1] = call_module[target=activation_post_process_29](args = (%layer4_1_relu_1,), kwargs = {})
#     %avgpool : [#users=1] = call_module[target=avgpool](args = (%activation_post_process_29,), kwargs = {})
#     %activation_post_process_30 : [#users=1] = call_module[target=activation_post_process_30](args = (%avgpool,), kwargs = {})
#     %flatten : [#users=1] = call_function[target=torch.flatten](args = (%activation_post_process_30, 1), kwargs = {})
#     %activation_post_process_31 : [#users=1] = call_module[target=activation_post_process_31](args = (%flatten,), kwargs = {})
#     %fc : [#users=1] = call_module[target=fc](args = (%activation_post_process_31,), kwargs = {})
#     %activation_post_process_32 : [#users=1] = call_module[target=activation_post_process_32](args = (%fc,), kwargs = {})
#     return activation_post_process_32
# GraphModule(
#   (conv1): QuantizedConvReLU2d(3, 64, kernel_size=(7, 7), stride=(2, 2), scale=0.02770872600376606, zero_point=0, padding=(3, 3))
#   (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
#   (layer1): Module(
#     (0): Module(
#       (conv1): QuantizedConvReLU2d(64, 64, kernel_size=(3, 3), stride=(1, 1), scale=0.013096675276756287, zero_point=0, padding=(1, 1))
#       (conv2): QuantizedConv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), scale=0.04380526766180992, zero_point=73, padding=(1, 1))
#     )
#     (1): Module(
#       (conv1): QuantizedConvReLU2d(64, 64, kernel_size=(3, 3), stride=(1, 1), scale=0.01650076173245907, zero_point=0, padding=(1, 1))
#       (conv2): QuantizedConv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), scale=0.05765068531036377, zero_point=78, padding=(1, 1))
#     )
#   )
#   (layer2): Module(
#     (0): Module(
#       (conv1): QuantizedConvReLU2d(64, 128, kernel_size=(3, 3), stride=(2, 2), scale=0.014281945303082466, zero_point=0, padding=(1, 1))
#       (conv2): QuantizedConv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), scale=0.03979799151420593, zero_point=59, padding=(1, 1))
#       (downsample): Module(
#         (0): QuantizedConv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), scale=0.03292635455727577, zero_point=67)
#       )
#     )
#     (1): Module(
#       (conv1): QuantizedConvReLU2d(128, 128, kernel_size=(3, 3), stride=(1, 1), scale=0.01630367338657379, zero_point=0, padding=(1, 1))
#       (conv2): QuantizedConv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), scale=0.04399171099066734, zero_point=67, padding=(1, 1))
#     )
#   )
#   (layer3): Module(
#     (0): Module(
#       (conv1): QuantizedConvReLU2d(128, 256, kernel_size=(3, 3), stride=(2, 2), scale=0.022209102287888527, zero_point=0, padding=(1, 1))
#       (conv2): QuantizedConv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), scale=0.04951776564121246, zero_point=47, padding=(1, 1))
#       (downsample): Module(
#         (0): QuantizedConv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), scale=0.01479915902018547, zero_point=86)
#       )
#     )
#     (1): Module(
#       (conv1): QuantizedConvReLU2d(256, 256, kernel_size=(3, 3), stride=(1, 1), scale=0.015260775573551655, zero_point=0, padding=(1, 1))
#       (conv2): QuantizedConv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), scale=0.050822507590055466, zero_point=72, padding=(1, 1))
#     )
#   )
#   (layer4): Module(
#     (0): Module(
#       (conv1): QuantizedConvReLU2d(256, 512, kernel_size=(3, 3), stride=(2, 2), scale=0.013010699301958084, zero_point=0, padding=(1, 1))
#       (conv2): QuantizedConv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), scale=0.04366626217961311, zero_point=68, padding=(1, 1))
#       (downsample): Module(
#         (0): QuantizedConv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), scale=0.037426918745040894, zero_point=65)
#       )
#     )
#     (1): Module(
#       (conv1): QuantizedConvReLU2d(512, 512, kernel_size=(3, 3), stride=(1, 1), scale=0.014271634630858898, zero_point=0, padding=(1, 1))
#       (conv2): QuantizedConv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), scale=0.24437215924263, zero_point=44, padding=(1, 1))
#     )
#   )
#   (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
#   (fc): QuantizedLinear(in_features=512, out_features=1000, scale=0.32927021384239197, zero_point=33, qscheme=torch.per_channel_affine)
# )



# def forward(self, x : torch.Tensor) -> torch.Tensor:
#     conv1_input_scale_0 = self.conv1_input_scale_0
#     conv1_input_zero_point_0 = self.conv1_input_zero_point_0
#     quantize_per_tensor = torch.quantize_per_tensor(x, conv1_input_scale_0, conv1_input_zero_point_0, torch.quint8);  x = conv1_input_scale_0 = conv1_input_zero_point_0 = None
#     conv1 = self.conv1(quantize_per_tensor);  quantize_per_tensor = None
#     maxpool = self.maxpool(conv1);  conv1 = None
#     layer1_0_conv1 = getattr(self.layer1, "0").conv1(maxpool)
#     layer1_0_conv2 = getattr(self.layer1, "0").conv2(layer1_0_conv1);  layer1_0_conv1 = None
#     layer1_0_relu_scale_0 = self.layer1_0_relu_scale_0
#     layer1_0_relu_zero_point_0 = self.layer1_0_relu_zero_point_0
#     add_relu = torch.ops.quantized.add_relu(layer1_0_conv2, maxpool, layer1_0_relu_scale_0, layer1_0_relu_zero_point_0);  layer1_0_conv2 = maxpool = layer1_0_relu_scale_0 = layer1_0_relu_zero_point_0 = None
#     layer1_1_conv1 = getattr(self.layer1, "1").conv1(add_relu)
#     layer1_1_conv2 = getattr(self.layer1, "1").conv2(layer1_1_conv1);  layer1_1_conv1 = None
#     layer1_1_relu_scale_0 = self.layer1_1_relu_scale_0
#     layer1_1_relu_zero_point_0 = self.layer1_1_relu_zero_point_0
#     add_relu_1 = torch.ops.quantized.add_relu(layer1_1_conv2, add_relu, layer1_1_relu_scale_0, layer1_1_relu_zero_point_0);  layer1_1_conv2 = add_relu = layer1_1_relu_scale_0 = layer1_1_relu_zero_point_0 = None
#     layer2_0_conv1 = getattr(self.layer2, "0").conv1(add_relu_1)
#     layer2_0_conv2 = getattr(self.layer2, "0").conv2(layer2_0_conv1);  layer2_0_conv1 = None
#     layer2_0_downsample_0 = getattr(getattr(self.layer2, "0").downsample, "0")(add_relu_1);  add_relu_1 = None
#     layer2_0_relu_scale_0 = self.layer2_0_relu_scale_0
#     layer2_0_relu_zero_point_0 = self.layer2_0_relu_zero_point_0
#     add_relu_2 = torch.ops.quantized.add_relu(layer2_0_conv2, layer2_0_downsample_0, layer2_0_relu_scale_0, layer2_0_relu_zero_point_0);  layer2_0_conv2 = layer2_0_downsample_0 = layer2_0_relu_scale_0 = layer2_0_relu_zero_point_0 = None
#     layer2_1_conv1 = getattr(self.layer2, "1").conv1(add_relu_2)
#     layer2_1_conv2 = getattr(self.layer2, "1").conv2(layer2_1_conv1);  layer2_1_conv1 = None
#     layer2_1_relu_scale_0 = self.layer2_1_relu_scale_0
#     layer2_1_relu_zero_point_0 = self.layer2_1_relu_zero_point_0
#     add_relu_3 = torch.ops.quantized.add_relu(layer2_1_conv2, add_relu_2, layer2_1_relu_scale_0, layer2_1_relu_zero_point_0);  layer2_1_conv2 = add_relu_2 = layer2_1_relu_scale_0 = layer2_1_relu_zero_point_0 = None
#     layer3_0_conv1 = getattr(self.layer3, "0").conv1(add_relu_3)
#     layer3_0_conv2 = getattr(self.layer3, "0").conv2(layer3_0_conv1);  layer3_0_conv1 = None
#     layer3_0_downsample_0 = getattr(getattr(self.layer3, "0").downsample, "0")(add_relu_3);  add_relu_3 = None
#     layer3_0_relu_scale_0 = self.layer3_0_relu_scale_0
#     layer3_0_relu_zero_point_0 = self.layer3_0_relu_zero_point_0
#     add_relu_4 = torch.ops.quantized.add_relu(layer3_0_conv2, layer3_0_downsample_0, layer3_0_relu_scale_0, layer3_0_relu_zero_point_0);  layer3_0_conv2 = layer3_0_downsample_0 = layer3_0_relu_scale_0 = layer3_0_relu_zero_point_0 = None
#     layer3_1_conv1 = getattr(self.layer3, "1").conv1(add_relu_4)
#     layer3_1_conv2 = getattr(self.layer3, "1").conv2(layer3_1_conv1);  layer3_1_conv1 = None
#     layer3_1_relu_scale_0 = self.layer3_1_relu_scale_0
#     layer3_1_relu_zero_point_0 = self.layer3_1_relu_zero_point_0
#     add_relu_5 = torch.ops.quantized.add_relu(layer3_1_conv2, add_relu_4, layer3_1_relu_scale_0, layer3_1_relu_zero_point_0);  layer3_1_conv2 = add_relu_4 = layer3_1_relu_scale_0 = layer3_1_relu_zero_point_0 = None
#     layer4_0_conv1 = getattr(self.layer4, "0").conv1(add_relu_5)
#     layer4_0_conv2 = getattr(self.layer4, "0").conv2(layer4_0_conv1);  layer4_0_conv1 = None
#     layer4_0_downsample_0 = getattr(getattr(self.layer4, "0").downsample, "0")(add_relu_5);  add_relu_5 = None
#     layer4_0_relu_scale_0 = self.layer4_0_relu_scale_0
#     layer4_0_relu_zero_point_0 = self.layer4_0_relu_zero_point_0
#     add_relu_6 = torch.ops.quantized.add_relu(layer4_0_conv2, layer4_0_downsample_0, layer4_0_relu_scale_0, layer4_0_relu_zero_point_0);  layer4_0_conv2 = layer4_0_downsample_0 = layer4_0_relu_scale_0 = layer4_0_relu_zero_point_0 = None
#     layer4_1_conv1 = getattr(self.layer4, "1").conv1(add_relu_6)
#     layer4_1_conv2 = getattr(self.layer4, "1").conv2(layer4_1_conv1);  layer4_1_conv1 = None
#     layer4_1_relu_scale_0 = self.layer4_1_relu_scale_0
#     layer4_1_relu_zero_point_0 = self.layer4_1_relu_zero_point_0
#     add_relu_7 = torch.ops.quantized.add_relu(layer4_1_conv2, add_relu_6, layer4_1_relu_scale_0, layer4_1_relu_zero_point_0);  layer4_1_conv2 = add_relu_6 = layer4_1_relu_scale_0 = layer4_1_relu_zero_point_0 = None
#     avgpool = self.avgpool(add_relu_7);  add_relu_7 = None
#     flatten = torch.flatten(avgpool, 1);  avgpool = None
#     fc = self.fc(flatten);  flatten = None
#     dequantize_32 = fc.dequantize();  fc = None
#     return dequantize_32
    
# # To see more debug info, please use `graph_module.print_readable()`
# Size of model before quantization
# Size (MB): 46.877553
# Size of model after quantization
# /root/miniconda3/lib/python3.8/site-packages/torch/jit/_check.py:172: UserWarning: The TorchScript type system doesn't support instance-level annotations on empty non-base types in `__init__`. Instead, either 1) use a type annotation in the class body, or 2) wrap the type in `torch.jit.Attribute`.
#   warnings.warn("The TorchScript type system doesn't support "
# Size (MB): 11.844807
# ..............................[before serilaization] Evaluation accuracy on test dataset: 88.73, 97.67
# ..............................[after serialization/deserialization] Evaluation accuracy on test dataset: 88.73, 97.67
# tensor(0.0007, grad_fn=<MaxBackward1>)
# ============ Comparison with Baseline Float Model and Eager Mode Quantization =================
# Size of baseline model
# Size (MB): 46.877553
# ..............................Baseline Float Model Evaluation accuracy: 88.53, 97.60
# Size of Fx graph mode quantized model
# Size (MB): 11.844807
# ..............................FX graph mode quantized model Evaluation accuracy on test dataset: 88.73, 97.67
# /root/miniconda3/lib/python3.8/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=ResNet18_QuantizedWeights.IMAGENET1K_FBGEMM_V1`. You can also use `weights=ResNet18_QuantizedWeights.DEFAULT` to get the most up-to-date weights.
#   warnings.warn(msg)
# Downloading: "https://download.pytorch.org/models/quantized/resnet18_fbgemm_16fa66dd.pth" to /root/.cache/torch/hub/checkpoints/resnet18_fbgemm_16fa66dd.pth
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 11.2M/11.2M [00:02<00:00, 4.56MB/s]
# /root/miniconda3/lib/python3.8/site-packages/torch/_utils.py:320: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
#   scales = torch.tensor(scales, dtype=torch.double, device=storage.device)
# /root/miniconda3/lib/python3.8/site-packages/torch/_utils.py:322: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
#   zero_points, dtype=torch.long, device=storage.device
# /root/miniconda3/lib/python3.8/site-packages/torch/_utils.py:335: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
#   device=storage.device,
# Size of eager mode quantized model
# Size (MB): 11.840057
# ..............................eager mode quantized model Evaluation accuracy on test dataset: 88.53, 97.67