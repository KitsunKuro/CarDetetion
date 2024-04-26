!pip install ultralytics

def imhist(img):
    m,n = img.shape

    # Hist là 1 danh sách chứa tần số histogram của ảnh
    hist = [0]*256
    for i in range (m):
      for j in range(n):
        hist[img[i,j]] += 1

    # Giá trị trả về là một mảng ~
    return np.array(hist)

def histrogram(img):
   m,n = img.shape
   # Hist là 1 danh sách chứa tần số histogram của ảnh
   hist = [0.0]*256
   for i in range (m):
    for j in range(n):
      hist[img[i,j]] += 1
   return np.array(hist)/(m*n)

def cumsum(hist):
  presum = [0.0] * 256
  presum[0] = hist[0]
  for i in range (1,256):
    presum[i] = presum[i-1] + hist[i]
  return np.array(presum)

def histEq(img):
 hist = histrogram(img)
 preImg = cumsum(hist)
 sk = 255*preImg
 imgHisteq = np.zeros(img.shape,np.uint8)
 for i in range(img.shape[0]):
  for j in range(img.shape[1]):
    imgHisteq[i,j] = sk[img[i,j]]
 return imgHisteq

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Xây dựng / khởi tạo mô hình?
model = YOLO("yolov8x.pt")

# Load lại mô hình đã được train
# model = YOLO("/content/gdrive/MyDrive/Colab Notebooks/Project/runs/detect/train3/weights/best.pt")

# Train mô hình
# model.train(data=os.path.join(ROOT_DIR,"config.yaml"),epochs=2) # train model
# model.train(data='coco8.yaml',epochs=1) # train model

fig = plt.figure(figsize=(16,9))
(ax1,ax2),(ax3,ax4) = fig.subplots(2,2)

# Đọc ảnh xám
img = cv2.imread('/content/0.png',0)
ax1.imshow(img,"gray")
ax1.set_title("Ảnh ban đầu")

# Cân bằng ánh sáng
canbanganhsang = histEq(img)
# ax2.imshow(canbanganhsang,"gray")

# Loại mịn bằng gaussian
img_blur = cv2.GaussianBlur(canbanganhsang, (3, 3), 0)
ax2.imshow(img_blur,"gray")
ax2.set_title("Ảnh sau khi làm mịn + cân bằng ánh sáng")

# Xử lý hình thái học
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
close = cv2.morphologyEx(img_blur,cv2.MORPH_CLOSE,kernel,iterations = 1)
ax3.imshow(close,"gray")
ax3.set_title("Ảnh sau khi xử lý hình thái học")

# Lấy ra ảnh màu để xét
img_color = cv2.cvtColor(close,cv2.COLOR_BGR2RGB)

result = model(img_color)

img_color = cv2.cvtColor(cv2.imread("/content/0.png"), cv2.COLOR_BGR2RGB)
cars = 0
for r in result:
  xyxys = r.boxes.xyxy
  for xyxy in xyxys:
    cv2.rectangle(img_color,(int(xyxy[0]),int(xyxy[1])),(int(xyxy[2]),int(xyxy[3])),(255,0,0),3);
    cars += 1

ax4.set_title("Ảnh sau khi xử lý")
ax4.imshow(img_color)
print("Số ô tô phát hiện ra là :" + str(cars))