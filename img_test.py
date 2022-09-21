# generate images for testing
from PIL import Image
from keras.datasets import mnist
import numpy as np

(X_train, y_train), (X_test, y_test) = mnist.load_data()
for i in np.random.randint(1, 1001, 10):
    arr2img = Image.fromarray(X_train[i])
    arr2img.save("{}.png".format(i), "PNG")
