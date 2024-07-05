from sklearn.utils import shuffle
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import pickle
import numpy as np
import random
import pandas as pd
import cv2
import glob

# def get_image_label_resize(label, filelist, dim = (32, 32), dataset = 'Train'):
#     x = np.array([cv2.resize(cv2.imread(fname), dim, interpolation = cv2.INTER_AREA) for fname in filelist])
#     y = np.array([label] * len(filelist))
#     return (x, y)  
# filelist = glob.glob('gtsrb-german-traffic-sign/Train/'+'0'+'/*.png')
# trainx, trainy = get_image_label_resize(0, glob.glob('gtsrb-german-traffic-sign/Train/'+str(0)+'/*.png'))
# # go through all others labels and store images into np array
# for label in range(1, 43):
#     filelist = glob.glob('gtsrb-german-traffic-sign/Train/'+str(label)+'/*.png')
#     x, y = get_image_label_resize(label, filelist)
#     trainx = np.concatenate((trainx ,x))
#     trainy = np.concatenate((trainy ,y))

training_file = 'traffic-signs-data/train.p'
validation_file='traffic-signs-data/valid.p'
testing_file = 'traffic-signs-data/test.p'
with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
x_train, y_train = train['features'], train['labels']
x_valid, y_valid = valid['features'], valid['labels']
x_test, y_test = test['features'], test['labels']
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_valid shape:", x_valid.shape)
print("y_valid shape:", y_valid.shape)
print("X_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)

# #
# Number of training example
n_train = x_train.shape[0]
# Number of valid example
n_valid=x_valid.shape[0]
# Number of testing examples.
n_test = x_test.shape[0]
image_shape = [x_train.shape[1],x_train.shape[2],x_train.shape[3]]

# How many unique classes/labels there are in the dataset.
def getLabelsCount(labels):
    d = dict(zip(labels, [0] * len(labels)))
    for x in labels:
        d[x] += 1
    return d
signsDicts = getLabelsCount(y_train)
n_classes = len(signsDicts)

print("Number of training examples =", n_train)
print("Number of validation examples =", n_valid)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

# #
def getSignNamesData():
    return pd.read_csv('./signnames.csv').values 
    #return pd.read_csv('./signnames.csv').as_matrix()
signsNamesData = getSignNamesData()
signNames = []
i=0
for sign in signsNamesData:
    i=i+1
    signNames.append(str(i)+'-'+sign[1])
print(signNames)
# #
plt.figure()
hist, bins = np.histogram(y_train, bins = n_classes)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align = 'center', width = width)
plt.title('Histogram of lable frequency')

# #
def gray_equlize(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    equ = cv2.equalizeHist(gray)
    return equ
x_train = np.array([gray_equlize(img) for img in x_train])
x_test = np.array([gray_equlize(img) for img in x_test])
x_valid = np.array([gray_equlize(img) for img in x_valid])
#
fig, axs = plt.subplots(7, 7, figsize = (15, 12))
fig.subplots_adjust(hspace = 0.2, wspace = 0.001)
axs = axs.ravel()
for i in range(49):
    index = random.randint(0, len(x_train))
    image = x_train[index]
    axs[i].axis('off')
    axs[i].imshow(image)
    axs[i].set_title(y_train[index])

# #
model = models.Sequential()
# Conv 32x32x1 => 28x28x6.
model.add(layers.Conv2D(filters = 6, kernel_size = (5, 5), strides=(1, 1), padding='valid', 
                        activation='sigmoid', data_format = 'channels_last', input_shape = (32, 32, 1))) #relu 、ELU
# Maxpool 28x28x6 => 14x14x6
model.add(layers.MaxPooling2D((2, 2)))
# Conv 14x14x6 => 10x10x16
model.add(layers.Conv2D(16, (5, 5), activation='sigmoid')) #relu 、ELU
# Maxpool 10x10x16 => 5x5x16
model.add(layers.MaxPooling2D((2, 2)))
# Flatten 5x5x16 => 400
model.add(layers.Flatten())
# Fully connected 400 => 120
model.add(layers.Dense(120, activation='sigmoid')) #relu 、ELU
# Fully connected 120 => 84
model.add(layers.Dense(84, activation='sigmoid')) #relu 、ELU
# Dropout
model.add(layers.Dropout(0.2))
# Fully connected, output layer 84 => 43
model.add(layers.Dense(43, activation='softmax'))
model.summary()

# #
x_train = np.expand_dims(x_train, axis=-1)
data_aug = ImageDataGenerator(
    featurewise_center=False, 
    featurewise_std_normalization=False, 
    rotation_range= 10,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.11,
    horizontal_flip=False,
    vertical_flip=False)
# Define a Callback class that stops training once accuracy reaches 98.0%
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.97):
      print("\nReached 97.0% accuracy so cancelling training!")
      self.model.stop_training = True
callbacks = myCallback()
# specify optimizer, loss function and metric
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

# training batch_size=128, epochs=30
conv = model.fit(data_aug.flow(x_train, y_train, batch_size=128), epochs=100,
                 validation_data=(x_valid, y_valid),verbose = 2,
                callbacks=[callbacks])
print(conv.history.keys())
# summarize history for accuracy
plt.figure()
plt.plot(conv.history['accuracy'],'-o')
plt.plot(conv.history['val_accuracy'],'-o')
plt.title('Training and Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['Training Accuracy', 'Validation Accuracy'], loc='lower right')
# summarize history for loss
plt.figure()
plt.plot(conv.history['loss'],'-o')
plt.plot(conv.history['val_loss'],'-o')
plt.title('Training and Validation loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Training loss', 'Validation loss'], loc='upper right')
model.evaluate(x=x_test, y=y_test)

# index = np.random.randint(0, n_test)
# im = x_test[index]
# fig, ax = plt.subplots()
# ax.set_title(sign.loc[sign['ClassId'] ==np.argmax(model.predict(np.array([im]))), 'SignName'].values[0])
# ax.imshow(im.squeeze(), cmap = 'gray')

model.save('traffic_sign_detection_LeNet_Sigmoid_100.hdf5')
# loaded_model = load_model('traffic sign detection.hdf5')

plt.show()



