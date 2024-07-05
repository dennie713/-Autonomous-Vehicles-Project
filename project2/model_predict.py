import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd

def getSignNamesData():
    return pd.read_csv('./signnames.csv').values 
# 加载模型
model = tf.keras.models.load_model('traffic_sign_detection_LeNet_ReLU_100_1.hdf5')
# model = tf.keras.models.load_model('traffic_sign_detection_LeNet_ELU_100.hdf5')
# model = tf.keras.models.load_model('traffic_sign_detection_LeNet_Sigmoid_100.hdf5')
NImages = 10
X_real = np.zeros((NImages,32,32,3)).astype(np.uint8)
y_real = np.array([17,12,14,11,38,4,35,33,25,13])
signsNamesData = getSignNamesData()
signNames = []
i=-1
for sign in signsNamesData:
    i=i+1
    signNames.append(str(i)+'-'+sign[1])
plt.figure()
for i in range(NImages):
    print(i+1)
    image = cv2.imread('testImages/'+str(i+1)+'.png')
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转换为灰度图像
    prediction = model.predict(np.array([image_gray]))
    plt.subplot(2, 5, i+1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(signNames[np.argmax(prediction)])
plt.tight_layout()
plt.show()