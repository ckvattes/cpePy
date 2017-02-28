import cv2
import numpy as np


# Find distance between two points M and N
def p2pDistance(p1, p2):
    p1 = p1.astype(float)
    p2 = p2.astype(float)
    return np.linalg.norm(p1 - p2)
    
# Find distance from point A perpendicular to line MN
def p2lDistance(p1, p2, k):
    p1 = p1.astype(float)
    p2 = p2.astype(float)
    k = k.astype(float)
    return np.linalg.norm(np.cross(p2-p1, p1-k))/np.linalg.norm(p2-p1)
