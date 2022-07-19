import os

def set_process_gpu():
    # worker_id 从1开始，可以手工映射到对应的显卡
    worker_id = int(os.environ.get('APP_WORKER_ID', 1))
    if worker_id <= 3:
        gpu_index = 0
    else:
        gpu_index = 1
    print('current worker id  {} set the gpu id :{}'.format(worker_id, gpu_index))
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_index}"
    return gpu_index

device_id = set_process_gpu()

import torch
from flask import Flask
from flask import request, jsonify
import cv2
import base64
from gevent import pywsgi

from torchvision.models import resnet50, ResNet50_Weights
import onnxruntime
import numpy as np

app = Flask(__name__)


@app.route('/aa', methods=["POST"])
def post_demo():
    dic_client = request.json
    pic_str = dic_client.get("img")

    img_data = base64.b64decode(pic_str)
    nparr = np.frombuffer(img_data, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    img2 = np.transpose(img_np, [2, 0, 1])[::-1].copy()
    img2 = torch.from_numpy(img2)

    return method_name(img2)


def method_name(img2):
    img2 = img2.to(device)
    batch = preprocess(img2).unsqueeze(0)
    a =batch[0][0][0][0].cpu() # 需要访存一下，不然会导致有的样本直接预测出错，应该是一个底层bug

    output = torch_model(batch)
    prediction = output[0].softmax(0)
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    category_name = weights.meta["categories"][class_id]
    line = f"{category_name}: {100 * score:.1f}%"
    # print(line)
    return jsonify({"ok": 1,"score": score})


device = 'cuda' if torch.cuda.is_available() else 'cpu'
weights = ResNet50_Weights.DEFAULT
preprocess = weights.transforms()

torch_model = torch.load('resnet.pth').to(device)
torch_model.eval()

data = np.random.rand(1, 3, 224, 224).astype(np.float32)
torch_data = torch.from_numpy(data).to(device)
torch_model(torch_data)

print("done")
# if __name__ == '__main__':
#     server = pywsgi.WSGIServer(('127.0.0.1', 7832), app)
#     server.serve_forever()
#     # app.run(host="0.0.0.0", port=7832, debug=False)
#     # app.run(host="127.0.0.1", port=7832, debug=False)
