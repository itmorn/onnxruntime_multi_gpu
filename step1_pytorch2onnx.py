# 将torch的预训练模型转为onnx格式
import torch
from torchvision.models import resnet50, ResNet50_Weights

device = 'cuda' if torch.cuda.is_available() else 'cpu'
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model.eval()

data = torch.rand(1, 3, 224, 224)
torch.save(model, 'resnet.pth')
torch.onnx.export(model, data, 'resnet.onnx', opset_version=16, input_names=['input'], output_names=['output'])
print("done")