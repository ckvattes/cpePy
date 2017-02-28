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

# Dummy Demo
# theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
# z = np.linspace(-2, 2, 100)
# r = z**2 + 1
# x = r * np.sin(theta)
# y = r * np.cos(theta)
# ax.plot(x, y, z, label='Parametric Curve Demo')
# ax.legend()
#///

# Convert pattern to greyscale
src_grey = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(src_grey, 127, 255, 0)
# src_edges = cv2.Canny(img,100,200)
_, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


# Find Marks
mark = []
for i in range(len(contours)):
    j, c = i, 0
    while hierarchy[j][2] != -1:
		j = hierarchy[j][2]
		c += 1
    if c >= 5:
	    mark.append(i)

# Lable Marks for Legibility	    
A = mark[0]
B = mark[1]
C = mark[2]
top, right, bttm = None, None, None
	    
# Find Mass Centers
mc = []
for i in contours:
    mu = cv2.moments(i)
    mc.append(np.array((int(mu["m10"] / mu["m00"]), int(mu["m01"] / mu["m00"]))))

# Find Mark Positions
if len(mark) >= 3:
    AB = cpe.p2pDistance(mc[A],mc[B])
    BC = cpe.p2pDistance(mc[B],mc[C])
    CA = cpe.p2pDistance(mc[C],mc[A])
    if AB > BC and AB > CA:
    	top = C
    	p1, p2 = A, B
    elif CA > AB and CA > BC:
    	top = B
    	p1, p2 = A, C
    elif BC > AB and BC > CA:
    	top = A
    	p1, p2 = B, C

# Display Output
# plt.show()        # Disabled for non-Xwindow use
# plt.savefig('../static/output/display.svg')

