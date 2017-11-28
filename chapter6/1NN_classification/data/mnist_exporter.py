#####################################################################
# run __ONE__ of the following commands:
# pip install --user tensorflow (if you have no CUDA-enabled GPU)
# pip install --user tensorflow-gpu
#
# afterwards install tflearn
# pip install --user tflearn
#
# Numpy should come bundled with tensorflow. Run this file et voila!
#####################################################################

import tflearn

# Data loading and preprocessing
import tflearn.datasets.mnist as mnist
X, Y, testX, testY = mnist.load_data(one_hot=True)

import array as ar
import numpy as np

with open("X.bin", "wb") as f:
    images = np.vstack((X, testX))
    print(images.shape)
    f.write(ar.array("f", images.flatten()))

with open("Y.bin", "wb") as f:
    labels = np.vstack((Y, testY))
    print(labels.shape)
    f.write(ar.array("f", labels.flatten()))
