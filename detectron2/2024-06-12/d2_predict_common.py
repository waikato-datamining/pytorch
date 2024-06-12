import cv2

from planar import Polygon, Vec2


def polygon_to_minrect(poly):
    """
    Determines the minimal rectangle around the provided polygon.
    Uses OpenCV's cv2.minAreaRect method for this.
    http://opencvpython.blogspot.com/2012/06/contours-2-brotherhood.html

    :param poly: the polygon describing a detected object: each contour is an ndarray of shape (n, 2),
                 consisting of n (x, y) coordinates along the contour
    :param poly: np.ndarray
    :return: the minimal rectangle, a tuple of width and height
    :rtype: tuple
    """

    rect = cv2.minAreaRect(np.float32(poly))
    w = rect[1][0]
    h = rect[1][1]
    return w, h


def lists_to_polygon(x, y):
    """
    Converts the list of x and y coordinates to a planar.Polygon.

    :param x: the list of X coordinates (float)
    :type x: list
    :param y: the list of Y coordinates (float)
    :type y: list
    :return: the polygon
    :rtype: Polygon
    """

    points = []
    for i in range(len(x)):
        points.append(Vec2(float(x[i]), float(y[i])))
    return Polygon(points)


def polygon_to_bbox(p):
    """
    Returns the x0,y0,x1,y1 coordinates of the bounding box surrounding the polygon.

    :param p: the polygon to get the bbox for
    :type p: Polygon
    :return: the bounding box coordinates (x0,y0,x1,y1)
    :rtype: tuple
    """

    bbox = p.bounding_box
    x0 = bbox.min_point.x
    y0 = bbox.min_point.y
    x1 = bbox.max_point.x
    y1 = bbox.max_point.y

    return x0,y0,x1,y1
