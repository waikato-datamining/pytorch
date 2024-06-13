from planar import Polygon, Vec2
from shapely import simplify, Polygon


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

    return x0, y0, x1, y1


def simplify_polygon(px, py, tolerance):
    """
    Simplifies the polygon using the specified tolerance parameter.

    :param px: the list of X coordinates
    :type px: list
    :param py: the list of Y coordinates
    :type py: list
    :param tolerance: the tolerance parameter, e.g., 0.01
    :type tolerance: float
    :return: the tuple of (potentially) updated lists of X and Y coordinates
    :rtype: tuple
    """
    points = []
    for x, y in zip(px, py):
        points.append((x, y))
    points.append((px[0], py[0]))
    poly = Polygon(points)
    poly_new = simplify(poly, tolerance)
    if isinstance(poly_new, Polygon) and (len(poly_new.exterior.coords) < len(poly.exterior.coords)):
        px, py = poly_new.exterior.coords.xy
    return px, py
