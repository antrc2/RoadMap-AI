from datasets import load_dataset
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Activation, Flatten, Conv2D, Dense, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
import numpy as np

datasets = load_dataset("ylecun/mnist")

data_train , data_test = datasets['train'], datasets['test']

x_train_feature, y_train_labels = data_train['image'], data_train['label']
x_test_feature, y_test_labels  = data_test['image'], data_test['label']

width=28
height=28
classes = 10
shape = (width,height,1)


model = Sequential()
model.add(Conv2D(1, (3, 3), padding='same', input_shape=shape))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(classes))
model.add(Activation('softmax'))  # nếu là phân loại nhiều lớp


model.summary()

aug = ImageDataGenerator(rotation_range=0.18, zoom_range=0.1,width_shift_range=0.2,height_shift_range=0.2,horizontal_flip=True)

learning_rate = 0.01
epochs = 5
batch_size = 64

opt = SGD(learning_rate=learning_rate,momentum=0.9)

model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=['acc'])

H = model.fit(x_train_feature,y_train_labels,batch_size=batch_size,epochs=epochs)

model.save("mnist_classification.h5")