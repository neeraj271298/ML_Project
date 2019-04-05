# accuracy 92.14 % 
# loading MNIST digit dataset 
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0],28,28,1).astype('float32')
x_test = x_test.reshape(x_test.shape[0],28,28,1).astype('float32')
x_train = x_train/255
x_test = x_test/255

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)


#importing required libraries
import numpy as np
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Conv2D, Activation, MaxPool2D, Flatten, Dropout, BatchNormalization
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint



model = Sequential()

model.add(Conv2D(64, 3, data_format="channels_last", kernel_initializer="he_normal", input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(Conv2D(64, 3))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.6))

model.add(Conv2D(32, 3))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(Conv2D(32, 3))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(Conv2D(32, 3))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.6))

model.add(Flatten())
model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.6))

model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# save best weights
checkpointer = ModelCheckpoint(filepath='face_model.h5', verbose=1, save_best_only=True)

# importing EarlyStopping and ReduceLROnPlateau
# EarlyStopping for preventing overfitting
# ReduceLearningOnPlateau benefit for reducing the learning rate
from keras.callbacks import EarlyStopping,ReduceLROnPlateau
early = EarlyStopping(patience = 1)
learning_rate = ReduceLROnPlateau(monitor='val_acc',patience = 1,verbose=1,factor=0.5,min_lr=0.00001)
callback = [early,learning_rate,checkpointer]

# num epochs
epochs = 10

# run model
hist = model.fit(x_train, y_train, epochs=epochs,
                 shuffle=True,
                 batch_size=100, validation_data=(x_test, y_test),
                 callbacks=callback, verbose=2)

# save the model to json file
model_json = model.to_json()
with open("face_model.json", "w") as json_file:
    json_file.write(model_json)

# load model library
from keras.models import model_from_json

# loading model
with open('model.json',"r") as file:
    loaded_json = file.read()
    loaded = model_from_json(loaded_json)
 
#load model weights
loaded.load_weights('model.h5')
loaded.summary()


# recognize digit from real time
import cv2
rgb = cv2.VideoCapture(0)
while True:
    _ , frame = rgb.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(gray, (28, 28))
    res = loaded.predict(roi[np.newaxis, :, :, np.newaxis])
    print(np.argmax(res))
    
    if cv2.waitKey(1) == 27:
            break
    cv2.imshow('Filter', frame)
cv2.destroyAllWindows()


# recognize digit from image
# load image form directory 
from PIL import Image
image = Image.open('9.jpg').convert('L')
image = image.resize((28,28))
imageA = np.array(image)
imageA = imageA.reshape(1,28,28,1).astype('float32')
imageA=imageA/255
res = loaded.predict(imageA)
print(np.argmax(res))









            break"""
