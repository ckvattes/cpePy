# Force matplotlib to not use any Xwindows backend.
import matplotlib # Pyplot rendering does not work in Cloud 9 IDE
matplotlib.use('Agg') # These 2 lines can be removed when working locally
# ///
# import cv2
import numpy as np
from matplotlib import image as mpimg, pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.gca(projection='3d')


# ax.set_xlim(-50, 50)
# ax.set_ylim(-50, 50)
# ax.set_zlim(0, 100)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
z = np.linspace(-2, 2, 100)
r = z**2 + 1
x = r * np.sin(theta)
y = r * np.cos(theta)
ax.plot(x, y, z, label='parametric curve demo')
ax.legend()

# plt.show()        # Disabled for non-Xwindow use
plt.savefig('../static/output/display.svg')