"""
@Auth: itmorn
@Date: 2022/7/15-15:12
@Email: 12567148@qq.com
"""
import time
from multiprocessing import Pool, cpu_count

import base64
import numpy as np
import os
import traceback
import requests
import json

def thread_task(img):
    try:
        url = "http://127.0.0.1:7832/aa"

        with open(img, 'rb') as img_file:  # 二进制打开图片文件
            img_b64encode = base64.b64encode(img_file.read())  # base64编码
        img_b64decode_str = img_b64encode.decode()
        payload = json.dumps({"img":img_b64decode_str})
        headers = {
            'Content-Type': 'application/json'
        }

        response = requests.request("POST", url, headers=headers, data=payload)
        jsn = json.loads(response.text)
        if jsn["ok"]!=1:
            print(response.text)
        return jsn["score"]

    except Exception as e:
        print("Exception: " + str(e))
        traceback.print_exc()
        # raise Exception(str(e))


if __name__ == '__main__':
    dir_in = "imgs_imagenet/"
    # dir_in = "/data01/zhaoyichen/data/ImageNet1k/val/n03785016/"
    lst_img = [dir_in+i for i in os.listdir(dir_in)]*500
    # lst_img = []
    # for home, dirs, files in os.walk("/data01/zhaoyichen/data/ImageNet1k/val"):
    #     for filename in files:
    #         # print(filename)
    #         fullname = os.path.join(home, filename)
    #         lst_img.append(fullname)
    print(len(lst_img))
    p = Pool(50)
    print("cpu数量为：%d" % cpu_count())
    print("主线程id为: %d" % os.getpid())
    print("线程开始处理了")
    a = time.time()
    try:
        # 线程池有3个线程, 线程数量可以大于cpu_count()的数量, 且os.getpid()获取的数值都不一样
        result_list = p.map(thread_task, lst_img)
        print(result_list)
        print(np.sum(result_list))

        print("等待所有线程执行完成")
        p.close()
        p.terminate()
    except Exception as e:
        print(e)
    finally:
        print("===============close===============")
        p.close()
        p.terminate()
    print((time.time()-a))