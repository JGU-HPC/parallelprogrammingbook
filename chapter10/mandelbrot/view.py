# use as follows:
# # $ ./mandel > mandel.txt
# $ python2 view.py mandel.txt

# install numpy and matplotlib from standard repositories
# or locally with pip
# pip install --user numpy
# pip install --user matplotlib

import numpy as np
import matplotlib.pyplot as plt
import math
import sys

# Extract points from specified file
im = np.loadtxt( sys.argv[1] )

# Display
plt.imshow(im,cmap=plt.cm.flag)
plt.show()
