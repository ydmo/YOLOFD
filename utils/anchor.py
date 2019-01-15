import math

def genAnchor( areas = (1.0, 0.5, 0.25), aspects = (2.0, 1.0, 0.5) ):
    anchors = []
    for area in areas:
        for aspect in aspects:
            anchors.append((math.sqrt(area / aspect), math.sqrt(area * aspect)))
    return anchors
     
if __name__ == '__main__':     
    anchors = genAnchor(areas=(16*9, 4*9, 1*9))
    print(anchors)