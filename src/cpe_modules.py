import cv2
import numpy as np
import operator

# Find Marks on QR Code from Source
def findMarks(src):
    src_grey = bgrConvert(src)
    _, thresh = cv2.threshold(src_grey, 200, 255, 0)    # 127 too low for lower threshold
    cannyImg = cv2.Canny(thresh, 100 , 200) # Use cannyImg to avoid OOR Index
    _, contours, hierarchy = cv2.findContours(cannyImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = hierarchy[0]    # Use 0th Array to avoid Value Ambiguity

    # Find Marks
    mark = []
    for i in range(len(contours)):
        j, c = i, 0
        while hierarchy[j][2] != -1:
    		j = hierarchy[j][2]
    		c += 1
        if c >= 5:
    	    mark.append(i)
    # Label Marks for Legibility	    
    A = mark[0]
    B = mark[1]
    C = mark[2]
    top, right, bttm = None, None, None
    	    
    # Find Mass Centers
    mc = []
    for i in contours:
        mu = cv2.moments(i)
        if mu["m00"] == 0:
            mc.append((0,0))    #Avoid dividing by 0
        else:
            mc.append(np.array((int(mu["m10"] / mu["m00"]), int(mu["m01"] / mu["m00"]))))

    # Find Mark Positions
    if len(mark) >= 3:
        AB = p2pDistance(mc[A],mc[B])
        BC = p2pDistance(mc[B],mc[C])
        CA = p2pDistance(mc[C],mc[A])
        if AB > BC and AB > CA:
        	top = C
        	p1, p2 = A, B
        elif CA > AB and CA > BC:
        	top = B
        	p1, p2 = A, C
        elif BC > AB and BC > CA:
        	top = A
        	p1, p2 = B, C
        	
        d = p2lDistance(mc[p1], mc[p2], mc[top])
        m = slope(mc[p1], mc[p2])
        
        if m == None:
            right, bttm = p2, p1
        if m < 0 and d < 0:
            right, bttm = p2, p1
        elif m > 0 and d < 0:
            right, bttm = p1, p2
        elif m < 0 and d > 0:
            right, bttm = p1, p2
        elif m > 0 and d > 0:
            right, bttm = p2, p1
    
    # Find the center of SRC Pattern
    c1 = (mc[p1][0] + mc[p2][0]) // 2
    c2 = (mc[p1][1] + mc[p2][1]) // 2
    center = np.array(c1, c2)

	# Find Vertices of SRC Pattern Markers
    topV = getVertices(contours[top], mc[top])
    rightV = getVertices(contours[right], mc[right])
    bttmV = getVertices(contours[bttm], mc[bttm])
    
    topV = updateVertOrder(topV, mc[top], center)
    rightV = updateVertOrder(rightV, mc[right], center)
    bttmV = updateVertOrder(bttmV, mc[bttm], center)

    # Find fourth corner of SRC QR Code
    e1, e2 = None, None
    if p2pDistance(rightV[1], mc[top]) > p2pDistance(rightV[-1], mc[top]):
        e1 = rightV[1]
    else:
        e1 = rightV[-1]
    if p2pDistance(bttmV[1], mc[top]) > p2pDistance(bttmV[-1], mc[top]):
        e2 = bttmV[1]
    else:
        e2 = bttmV[-1]
    N = brCorner(rightV[0], e1, bttmV[0], e2)
    bounds = np.array([topV[0], rightV[0], N, bttmV[0]])
    return bounds       #Bounds are ordered [UL, UR, LR, LL]
    
# Convert pattern to greyscale
def bgrConvert(src):
    src_grey = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    return src_grey

# Find distance between two points p1 and p2
def p2pDistance(p1, p2):
    p1 = p1.astype(float)
    p2 = p2.astype(float)
    return np.linalg.norm(p1 - p2)
    
# Find distance from point k perpendicular to line p1p2
def p2lDistance(p1, p2, k):
    p1 = p1.astype(float)
    p2 = p2.astype(float)
    k = k.astype(float)
    return np.linalg.norm(np.cross(p2-p1, p1-k))/np.linalg.norm(p2-p1)

# Find slope of line p1p2
def slope(p1, p2):
    p1 = p1.astype(float)
    p2 = p2.astype(float)
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    if dy == 0:
    	return None
    else:
		return (dy / dx)
    
# Find vertices of given mark
def getVertices(c, mc):
	minBox = cv2.minAreaRect(c)
	minBoxVerts = cv2.boxPoints(minBox)
	mcDists = np.zeros((4, 1))
	vertices = np.array([None] * 4)
	for x in c:
		x = x[0]
		boundDist = []
		for vertex in minBoxVerts:
			vertex = np.array(vertex)
			boundDist.append(p2pDistance(x, vertex))
		quadrant = np.argmin(boundDist)
		mcDist = p2pDistance(x, mc)
		if mcDist > mcDists[quadrant]:
			mcDists[quadrant] = mcDist
			vertices[quadrant] = x
	return vertices
	
# Reorder vertices to be outer bound of mark
def updateVertOrder(vertices, mc, center):
    d = []
    for i in range(len(vertices)):
    	d.append((i, abs(p2lDistance(mc, center, vertices[i]))))
    d = sorted(d, key=operator.itemgetter(1))
    if p2pDistance(vertices[d[0][0]], center) > p2pDistance(vertices[d[1][0]], center):
        outer = d[0][0]
    else:
        outer = d[1][0]
    return np.append(vertices[outer:], vertices[:outer])

def brCorner(r1, r2, b1, b2):
    r1 = r1.astype(float)
    r2 = r2.astype(float)
    b1 = b1.astype(float)
    b2 = b2.astype(float)
    
    v1 = r2 - r1
    v2 = b2 - b1
    s = np.cross(v1, v2)
    t = np.cross((b1 - r1), v2) / s
    N = np.array(np.int0(r1 + (t * v1)))
    return N

