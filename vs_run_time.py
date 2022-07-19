"""
@Auth: itmorn
@Date: 2022/7/18-15:17
@Email: 12567148@qq.com
"""
from torchvision.models import resnet50, ResNet50_Weights
import torch
import onnxruntime
from timeit import timeit
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)

data = np.random.rand(1, 3, 224, 224).astype(np.float32)
torch_model = torch.load('resnet.pth').to(device)
torch_data = torch.from_numpy(data).to(device)
onnx_model = onnxruntime.InferenceSession('resnet.onnx', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
torch_model.eval()

img2 = torch.randn(3, 224, 224)
img2 = img2.to(device)
preprocess = weights.transforms()
batch = preprocess(img2).unsqueeze(0)

# Change the shape to the actual shape of the output being bound
Y_ortvalue = onnxruntime.OrtValue.ortvalue_from_shape_and_type([1, 1000], np.float32, 'cuda', 0)
session = onnxruntime.InferenceSession('resnet.onnx', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
io_binding = session.io_binding()

io_binding.bind_input(
    name='input',
    device_type='cuda',
    device_id=0,
    element_type=np.float32,
    shape=tuple(batch.shape),
    buffer_ptr=batch.data_ptr(),
)

io_binding.bind_ortvalue_output('output', Y_ortvalue)


def torch_inf():
    res = torch_model(torch_data)


def onnx_inf():
    res = onnx_model.run(None, {
        onnx_model.get_inputs()[0].name: data #从内存读取到显存有开销
    })


def onnx_inf2():
    res = session.run_with_iobinding(io_binding)


n = 2
onnx_t = timeit(lambda: onnx_inf(), number=n) / n
onnx_t2 = timeit(lambda: onnx_inf2(), number=n) / n
torch_t = timeit(lambda: torch_inf(), number=n) / n

n = 500
onnx_t = timeit(lambda: onnx_inf(), number=n) / n
onnx_t2 = timeit(lambda: onnx_inf2(), number=n) / n
torch_t = timeit(lambda: torch_inf(), number=n) / n

print(f"PyTorch {torch_t} , ONNX {onnx_t} , ONNX2 {onnx_t2}")
