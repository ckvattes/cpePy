# Force matplotlib to not use any Xwindows backend.
import matplotlib # Pyplot rendering does not work in Cloud 9 IDE
matplotlib.use('Agg') # These 2 lines can be removed when working locally
# ///
import cv2
import numpy as np
from matplotlib import image as mpimg, pyplot as plt
import cpe_modules as cpe

# Set Figure, Axes, and Image Variables
fig = plt.figure()
ax = fig.gca(projection='3d')
size = 8.8  # This is kept at full-scale for visibility
srcPath = "../static/input/pattern.jpg"
outPath = "../static/output/display.svg"
imgPaths = ["../static/img/IMG_6719.JPG",
            "../static/img/IMG_6720.JPG",
            "../static/img/IMG_6721.JPG",
            "../static/img/IMG_6723.JPG",
            "../static/img/IMG_6725.JPG",
            "../static/img/IMG_6726.JPG",
            "../static/img/IMG_6727.JPG"
           ]

# Set Axis Labels and Limits, Plot SRC Image
print "START"
print "Loading source image: %s" % (srcPath)
src = cv2.flip(cv2.imread(srcPath), 0)    #Flip fixes rotation problem
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zticklabels([])
ax.set_zticks([])
# ax.set_zlabel('Z-axis')   # Z-axis labels gets in the way of scatter labels
ax.set_xlim(-50, 50)
ax.set_ylim(-50, 50)
ax.set_zlim(0, 100)     # Z-axis lower limit == 0
X, Y = np.meshgrid(np.linspace(-size, size, src.shape[0]), np.linspace(-size, size, src.shape[0]))
Z = 0
print "Plotting source image (QR Code)"
ax.plot_surface(X, Y, Z, facecolors=src/255.0, shade=False)


for imgPath in imgPaths:
    # Read SRC Pattern and Img (test1)
    print ">>>>>>>>>>>>>>>>>>>>>>>>>"
    print "Reading Image File: %s" % (imgPath)
    img = cv2.flip(cv2.transpose(cv2.imread(imgPath)), 1)   #Flip fixes rotation problem
    
    # Find Marks from QR Code
    print "Detecting QR Code Markers"
    bounds = cpe.findMarks(img)
    print "Detecting Camera Pose and Orientation"
    P, V = cpe.reconstruct(bounds, (img.shape))
    print "Pose = " + np.array_str(P)
    print "Orientation = " + np.array_str(V)

    # Plot Camera Pose (P) and Orientation Vector (V)
    print "Plotting Pose and Orientation"
    cpe.plot(P, V, imgPath, ax)
    print "<<<<<<<<<<<<<<<<<<<<<<<<<\n"
    
# Display Output
# plt.show()        # Disabled for non-Xwindow use

# Save output to file
plt.savefig("../static/output/display.svg")
# ///
print "Output saved as %s" % (outPath)
print "DONE"