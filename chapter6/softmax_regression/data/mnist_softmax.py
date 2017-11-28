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

from __future__ import division, print_function, absolute_import
import tflearn

# Data loading and preprocessing
import tflearn.datasets.mnist as mnist
X, Y, testX, testY = mnist.load_data(one_hot=True)

input_layer = tflearn.input_data(shape=[None, 784])
softmax = tflearn.fully_connected(input_layer, 10, activation='softmax', name="fully")

net = tflearn.regression(softmax, optimizer="adam",
                         loss='categorical_crossentropy')

# Training
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(X, Y, n_epoch=5, validation_set=(testX, testY),
          show_metric=True, run_id="dense_model")

with model.session.as_default():
    fully = tflearn.variables.get_layer_variables_by_name('fully')
    A = tflearn.variables.get_value(fully[0]).T
    b = tflearn.variables.get_value(fully[1])

print(A.shape)
print(b.shape)

import array as ar
import numpy as np

with open("A.bin", "wb") as f:
    f.write(ar.array("f", A.flatten()))

with open("b.bin", "wb") as f:
    f.write(ar.array("f", b))

with open("X.bin", "wb") as f:
    images = np.vstack((X, testX))
    print(images.shape)
    f.write(ar.array("f", images.flatten()))

with open("Y.bin", "wb") as f:
    labels = np.vstack((Y, testY))
    print(labels.shape)
    f.write(ar.array("f", labels.flatten()))
