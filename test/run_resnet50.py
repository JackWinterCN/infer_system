import torch
import torchvision.models as models
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

# 步骤 1：加载 PyTorch 模型并转换为 ONNX 格式
model = models.resnet50(pretrained=True) # 加载预训练的 ResNet50 模型
model.eval()
dummy_input = torch.randn(1, 3, 224, 224) # 创建一个示例输入
torch.onnx.export(model, dummy_input, "resnet50.onnx", opset_version=11) # 将模型导出为 ONNX 格式

# 步骤 2：使用 TensorRT 将 ONNX 模型转换为 TensorRT 引擎
TRT_LOGGER = trt.Logger(trt.Logger.WARNING) # 创建一个 Logger
EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH) # 如果是动态输入，需要显式指定 EXPLICIT_BATCH
with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
    # 创建一个 Builder 和 Network
    # builder 创建计算图 INetworkDefinition
    builder.max_workspace_size = 1 << 30  # 1GB ICudaEngine 执行时 GPU 最大需要的空间
    builder.max_batch_size = 1 # 执行时最大可以使用的 batchsize

    with open("resnet50.onnx", "rb") as model_file:
        parser.parse(model_file.read())  # 解析 ONNX 文件

    engine = builder.build_cuda_engine(network)  # 构建 TensorRT 引擎

    with open("resnet50.trt", "wb") as f:
        # 将引擎保存到文件
        f.write(engine.serialize())

# 步骤 3：使用 TensorRT 引擎进行推理
def load_engine(engine_file_path):
    # 加载 TensorRT 引擎
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

engine = load_engine("resnet50.trt")
context = engine.create_execution_context() # 将引擎应用到不同的 GPU 上配置执行环境

# 准备输入和输出缓冲区
input_shape = (1, 3, 224, 224)
output_shape = (1, 1000)
input_size = trt.volume(input_shape) * trt.float32.itemsize
output_size = trt.volume(output_shape) * trt.float32.itemsize
d_input = cuda.mem_alloc(input_size)
d_output = cuda.mem_alloc(output_size)
stream = cuda.Stream() # 创建流
input_data = np.random.random(input_shape).astype(np.float32)# 创建输入数据
cuda.memcpy_htod_async(d_input, input_data, stream) # 复制输入数据到 GPU

# 推理
context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
# 从 GPU 复制输出数据
output_data = np.empty(output_shape, dtype=np.float32)
cuda.memcpy_dtoh_async(output_data, d_output, stream) # 获取推理结果，并将结果拷贝到主存
stream.synchronize() # 同步流
print("Predicted output:", output_data)
