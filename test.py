import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision.transforms import ToTensor
#
# img = cv2.imread("000019.jpg")
# # 获取高宽信息
# h, w, _ = img.shape
# # 调整图像大小
# image = cv2.resize(img, (448, 448))
# # CV2读取的图像是BGR，这里转回RGB模式
# img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# # 图像均值
# mean = (123, 117, 104)  # RGB
# # 减去均值进行标准化操作
# img = img - np.array(mean, dtype=np.float32)
# # 创建数据增强函数
# transform = ToTensor()
# # 图像转为tensor，因为cv2读取的图像是numpy格式
# img = transform(img)
# # 输入要求是BCHW，加一个batch维度
# img = img.unsqueeze(0)
# img = img.to("cuda")
#
# # cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (144, 144, 255))  # 画框
# cv2.rectangle(image, (142,67),(273,343),(144, 144, 255))
# plt.imsave('test_001.jpg', image)

a = torch.FloatTensor(
    [
        [[1,2,3],[1,2,3]],
        [[1,2,3],[1,2,3]]
    ]
)
print(a.shape)
a[:,:,:2] = torch.softmax(a[:,:,:2],dim=2)
print(a)