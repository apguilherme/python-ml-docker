from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

(X_train, y_train), (X_test, y_test) = mnist.load_data()
plt.subplot(221)
plt.imshow(X_train[0], cmap=plt.get_cmap("gray"))
plt.subplot(222)
plt.imshow(X_train[1], cmap=plt.get_cmap("gray"))
plt.subplot(223)
plt.imshow(X_train[2], cmap=plt.get_cmap("gray"))
plt.subplot(224)
plt.imshow(X_train[3], cmap=plt.get_cmap("gray"))
#plt.show()

K.set_image_data_format("channels_last")
seed = 7
np.random.seed(seed)
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype("float32")
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype("float32")
X_train = X_train/255 # normalize
X_test = X_test/255
y_train = np_utils.to_categorical(y_train) # one hot encoding
y_test = np_utils.to_categorical(y_test)
num_classes = y_train.shape[1]

def baseline_model():
    model = Sequential()
    model.add(Conv2D(8, (3,3), input_shape=(1,28,28), activation='relu', padding="same"))
    model.add(MaxPooling2D((2,2), padding="same"))
    model.add(Flatten())
    model.add(Dense(4, activation="softmax"))
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

# build model
model = baseline_model()
# fit
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32, verbose=2)
model.save("model_img.h5")
# final eval
scores = model.evaluate(X_test, y_test, verbose=0)
print("CNN error: %.2f%%" % (100 - scores[1]*100))

