# Force matplotlib to not use any Xwindows backend.
import matplotlib # Pyplot rendering does not work in Cloud 9 IDE
matplotlib.use('Agg') # These 2 lines can be removed when working locally
# ///
import cv2
import numpy as np
from matplotlib import image as mpimg, pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cpe_modules as cpe

# Set Figure and Axes
fig = plt.figure()
ax = fig.gca(projection='3d')

# Set Axis Labels and Limits
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
# ax.set_xlim(-50, 50)
# ax.set_ylim(-50, 50)
# ax.set_zlim(0, 100)

# Read SRC Pattern and Img (test1)
src = cv2.imread("../static/input/pattern.jpg")
img = cv2.imread("../static/img/IMG_6719.jpg")

# Find Marks from QR Code
bounds = cpe.findMarks(src)
print bounds

# Display Output
# plt.show()        # Disabled for non-Xwindow use
# plt.savefig('../static/output/display.svg')

