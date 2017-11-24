#####################################################################
# run __ONE__ of the following commands:
#
# pip install --user tensorflow (if you have no CUDA-enabled GPU)
# pip install --user tensorflow-gpu
#
# Numpy should come bundled with tensorflow. Run this file et voila!
#####################################################################

import array as ar
import numpy as np

# everyone has tensorflow installed nowadays :D
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

merge = np.vstack(( mnist.train.images, mnist.test.images))

with open("mnist_all_65000_28_28_32.bin", "wb") as f:
    f.write(ar.array("f", merge.flatten()))
