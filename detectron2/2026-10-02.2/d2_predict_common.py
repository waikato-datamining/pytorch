from shapely import simplify, Polygon


def lists_to_polygon(px, py):
    """
    Converts the list of x and y coordinates to a planar.Polygon.

    :param px: the list of X coordinates (float)
    :type px: list
    :param py: the list of Y coordinates (float)
    :type py: list
    :return: the polygon generated from the points
    :rtype: Polygon
    """
    points = []
    for x, y in zip(px, py):
        points.append((x, y))
    points.append((px[0], py[0]))
    return Polygon(points)


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
