import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.quantization

# 定义一个简单的卷积神经网络
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 320)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 准备训练数据
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# 实例化模型、损失函数和优化器
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# if torch.cuda.is_available():
#     print("CUDA is enable!")
#     model = model.cuda()

model.train()


# 训练模型
for epoch in range(5):  # 训练5个epoch
    for images, labels in train_loader:
        # if torch.cuda.is_available():
        #     images  = images.cuda()
        #     labels = labels.cuda()
       
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/5], Loss: {loss.item():.4f}')

# 模型量化
model.eval()
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model, inplace=True)

# 完成量化感知训练
for images, labels in train_loader:
    # if torch.cuda.is_available():
    #     images  = images.cuda()
    #     labels = labels.cuda()
    model(images)
torch.quantization.convert(model, inplace=True)

# model.cuda()
dummy_input = torch.randn(1, 1, 28, 28) #.cuda()
torch.onnx.export(
        model,
        dummy_input,
        "quant_mnist.onnx",
        verbose=True,
        input_names = [ "actual_input_1" ],
        output_names = [ "actual_output_1" ],
        opset_version=17,
        do_constant_folding=True
        )

print("量化训练完成!")