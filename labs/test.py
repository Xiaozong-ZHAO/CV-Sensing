import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像并转换为灰度图
image = cv2.imread('C:/Users/Lenovo/comp0241_24/dataset/Lenna.png')
# show the image

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 使用Canny算法进行边缘检测，调节阈值
edges = cv2.Canny(gray, 100, 200, apertureSize=3)

# 使用概率霍夫变换检测直线
linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

# 复制一份图像，用于绘制直线
image_with_lines = image.copy()

# 在图像上绘制检测到的直线
if linesP is not None:
    for line in linesP:
        x1, y1, x2, y2 = line[0]
        cv2.line(image_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿色的线，线宽2

# 创建两个子图并行显示原图和检测后的图像
plt.figure(figsize=(15, 10))

# 显示原图
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('off')

# 显示经过霍夫变换检测直线的图像
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(image_with_lines, cv2.COLOR_BGR2RGB))
plt.title("Image with Detected Lines (Hough Transform)")
plt.axis('off')

# 显示图像
plt.show()