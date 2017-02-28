# Force matplotlib to not use any Xwindows backend.
import matplotlib # Pyplot rendering does not work in Cloud 9 IDE
matplotlib.use('Agg') # These 2 lines can be removed when working locally
# ///
import re
import cv2
import numpy as np
from matplotlib import image as mpimg, pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cpe_modules as cpe
from cpe_modules import Arrow3D

# Set Figure, Axes, and Image Variables
fig = plt.figure()
ax = fig.gca(projection='3d')
src = cv2.imread("../static/input/pattern.jpg")
imgPath = ["../static/img/IMG_6719.JPG",
           "../static/img/IMG_6720.JPG",
           "../static/img/IMG_6721.JPG",
           "../static/img/IMG_6723.JPG",
           "../static/img/IMG_6725.JPG",
           "../static/img/IMG_6726.JPG",
           "../static/img/IMG_6727.JPG"]


# Set Axis Labels and Limits
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_xlim(-50, 50)
ax.set_ylim(-50, 50)
ax.set_zlim(0, 100)     # Z-axis lower limit == 0

# Read SRC Pattern and Img (test1)
img = cv2.imread(imgPath[0])

# Find Marks from QR Code
bounds = cpe.findMarks(img)
P, V = cpe.reconstruct(bounds, (img.shape))


size = 4.4
X, Y = np.meshgrid(np.linspace(-size, size, src.shape[0]), np.linspace(-size, size, src.shape[0]))
Z = 0
ax.plot_surface(X, Y, Z, rstride=10, cstride=10, facecolors=src / 255., shade=False)


imgName = (imgPath[0].split("/"))[-1]
imgName = re.sub('\.JPG$', '', imgName)
label = '%s (%d, %d, %d)' % (imgName, P[0], P[1], P[2])
ax.scatter([P[0]], [P[1]], [P[2]])
ax.text(P[0], P[1], P[2], label)

# Display Output
# plt.show()        # Disabled for non-Xwindow use
plt.savefig('../static/output/display.svg')