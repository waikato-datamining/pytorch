from __future__ import annotations

import argparse
import cv2
import json
import numpy as np
import os
from PIL import Image
from dataclasses import dataclass
from datetime import datetime
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.engine.defaults import DefaultPredictor
from opex import ObjectPredictions, ObjectPrediction, BBox
from opex import Polygon as OpexPolygon
from sahi.slicing import slice_image
from shapely import simplify
from shapely.affinity import translate
from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry import Polygon, MultiPolygon, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.strtree import STRtree
from typing import List, Optional


@dataclass
class Detection:
    bbox: np.ndarray
    score: float
    category_id: int
    polygon: Optional[BaseGeometry] = None

    @property
    def geometry(self):
        if self.polygon is not None:
            return self.polygon
        return box(*self.bbox)


def bbox_ios(box_a, box_b) -> float:
    """
    Intersection over Smaller (IoS).

    :param box_a: first bounding box
    :param box_b: second bounding box
    :return: IoS
    :rtype: float
    """
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    ix1 = max(xa1, xb1)
    iy1 = max(ya1, yb1)
    ix2 = min(xa2, xb2)
    iy2 = min(ya2, yb2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    inter = iw * ih

    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)

    smaller = min(area_a, area_b)

    if smaller <= 0:
        return 0.0

    return inter / smaller


def polygon_ios(geom_a: BaseGeometry, geom_b: BaseGeometry) -> float:
    """
    Polygon Intersection over Smaller (IoS).

    :param geom_a: first polygon/bbox
    :param geom_b: second polygon/bbox
    :return: the IoS value
    :rtype: float
    """
    if geom_a.is_empty or geom_b.is_empty:
        return 0.0

    inter_area = geom_a.intersection(geom_b).area
    smaller_area = min(geom_a.area, geom_b.area)

    if smaller_area <= 0:
        return 0.0

    return inter_area / smaller_area


def mask_to_polygon(mask: np.ndarray) -> Optional[Union[Polygon | MultiPolygon]]:
    """
    Convert a binary mask into a valid Shapely geometry.

    :param mask: the mask to convert
    :type mask: np.ndarray
    :return: the polygon/multi-polygon or None
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    geoms = []

    for contour in contours:
        if len(contour) < 3:
            continue

        poly = Polygon(contour[:, 0, :])

        # repair invalid geometries
        if not poly.is_valid:
            poly = poly.buffer(0)

        if poly.is_empty:
            continue

        # poly may now be Polygon OR MultiPolygon
        if isinstance(poly, Polygon):
            geoms.append(poly)

        elif isinstance(poly, MultiPolygon):
            geoms.extend(poly.geoms)

    if not geoms:
        return None

    # safest way to combine everything
    result = unary_union(geoms)

    if result.is_empty:
        return None

    result = result.buffer(0)

    return result


def predict_slice(predictor, image_slice, offset_x: int, offset_y: int, score_threshold: float = 0.5) -> List[Detection]:
    """
    Run Detectron2 on a tile and shift outputs to full-image coordinates.

    :param predictor: the Detectron2 model
    :param image_slice: the slice to generate predictions for
    :param offset_x: the x offset for the slice
    :type offset_x: int
    :param offset_y: the y offset for the slice
    :type offset_y: int
    :param score_threshold: the minimum score for predictions
    :type score_threshold: float
    :return: the detections
    :rtype: list
    """
    outputs = predictor(image_slice)
    instances = outputs["instances"].to("cpu")
    boxes = instances.pred_boxes.tensor.numpy()
    scores = instances.scores.numpy()
    classes = instances.pred_classes.numpy()
    has_masks = instances.has("pred_masks")
    detections = []

    for idx in range(len(boxes)):
        score = float(scores[idx])
        if score < score_threshold:
            continue

        bbox = boxes[idx].astype(np.float32).copy()
        bbox[0] += offset_x
        bbox[2] += offset_x
        bbox[1] += offset_y
        bbox[3] += offset_y

        geom = None
        if has_masks:
            mask = instances.pred_masks[idx].numpy()
            geom = mask_to_polygon(mask)
            if geom is not None:
                geom = translate(geom, xoff=float(offset_x), yoff=float(offset_y))

        detections.append(
            Detection(
                bbox=bbox,
                score=score,
                category_id=int(classes[idx]),
                polygon=geom,
            )
        )

    return detections


def sliced_inference(image, predictor, score_threshold: float = 0.5, slice_height: int = 1024, slice_width: int = 1024,
                     overlap_height_ratio: float = 0.2, overlap_width_ratio: float = 0.2) -> List[Detection]:
    """
    Run inference across all SAHI slices.

    :param image: the image to slice and then predict on
    :param predictor: the Detectron2 model to use
    :param score_threshold: the minimum score for predictions
    :type score_threshold: float
    :param slice_height: the height of the slices in pixels
    :type slice_height: int
    :param slice_width: the width of the slices in pixels
    :type slice_width: int
    :param overlap_height_ratio: the ratio for the height overlaps
    :type overlap_height_ratio: float
    :param overlap_width_ratio: the ratio for the width overlaps
    :type overlap_width_ratio: float
    :return: the list of predictions
    :rtype: list
    """
    sliced = slice_image(
        image=image,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
    )

    all_predictions = []
    for slice_info in sliced.sliced_image_list:
        tile = slice_info.image
        start_x = slice_info.starting_pixel[0]
        start_y = slice_info.starting_pixel[1]
        detections = predict_slice(predictor, tile, start_x, start_y, score_threshold=score_threshold)
        all_predictions.extend(detections)

    return all_predictions


class UnionFind:

    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def build_clusters(detections: List[Detection], ios_threshold: float = 0.5):
    """
    Build connected components using STRtree and IoS.

    :param detections: the list of detections
    :type detections: list
    :param ios_threshold: the IoS threshold to use
    :type ios_threshold: float
    """
    geometries = [det.geometry for det in detections]
    tree = STRtree(geometries)
    uf = UnionFind(len(detections))

    for idx, det in enumerate(detections):
        candidate_indices = tree.query(det.geometry)
        for j in candidate_indices:
            j = int(j)
            if j <= idx:
                continue

            other = detections[j]
            if det.category_id != other.category_id:
                continue

            if (det.polygon is not None) and (other.polygon is not None):
                overlap = polygon_ios(det.polygon, other.polygon)
            else:
                overlap = bbox_ios(det.bbox, other.bbox)
            if overlap >= ios_threshold:
                uf.union(idx, j)

    clusters = {}
    for idx in range(len(detections)):
        root = uf.find(idx)
        if root not in clusters:
            clusters[root] = []
        clusters[root].append(idx)

    return list(clusters.values())


def merge_cluster(cluster_indices: List[int], detections: List[Detection]) -> Detection:
    """
    Merge detections in one connected component.

    :param cluster_indices: the indices of the detection to merge
    :type cluster_indices: list
    :param detections: the full list of detections
    :type: list
    :return: the merged detection
    :rtype: Detection
    """
    cluster = [detections[i] for i in cluster_indices]
    best = max(cluster, key=lambda d: d.score)
    polygons = [
        det.polygon
        for det in cluster
        if det.polygon is not None
    ]

    merged_geometry = None
    if len(polygons) > 0:
        merged_geometry = unary_union(polygons)
        minx, miny, maxx, maxy = merged_geometry.bounds
        merged_bbox = np.array([minx, miny, maxx, maxy], dtype=np.float32)
    else:
        boxes = np.stack([det.bbox for det in cluster])
        merged_bbox = np.array(
            [
                boxes[:, 0].min(),
                boxes[:, 1].min(),
                boxes[:, 2].max(),
                boxes[:, 3].max(),
            ],
            dtype=np.float32,
        )

    return Detection(
        bbox=merged_bbox,
        score=best.score,
        category_id=best.category_id,
        polygon=merged_geometry,
    )


def merge_detections(detections: List[Detection], ios_threshold: float = 0.5):
    """
    Merge all overlapping detections.

    :param detections: the detections to merge
    :type detections: list
    :param ios_threshold: the IoS threshold to use
    :type ios_threshold: float
    :return: the merged detections
    :rtype: list
    """
    if len(detections) == 0:
        return []

    clusters = build_clusters(detections, ios_threshold=ios_threshold)
    merged = [merge_cluster(cluster, detections) for cluster in clusters]

    return merged


def polygon_nms(detections: List[Detection], ios_threshold: float = 0.7) -> List[Detection]:
    """
    Removes overlapping polygons.

    :param detections: the detections to process
    :type detections: list
    :param ios_threshold: the IoS threshold to use
    :type ios_threshold: float
    :return: the updated list of detections
    :rtype: list
    """
    detections = sorted(
        detections,
        key=lambda d: d.score,
        reverse=True,
    )

    keep = []
    while detections:
        best = detections.pop(0)
        keep.append(best)
        remaining = []
        for det in detections:
            if (det.polygon is not None) and (best.polygon is not None):
                overlap = polygon_ios(best.polygon, det.polygon)
            else:
                overlap = bbox_ios(best.bbox, det.bbox)
            if overlap < ios_threshold:
                remaining.append(det)

        detections = remaining

    return keep


def predict(image, predictor, score_threshold: float = 0.5, slice_height: int = 1024, slice_width: int = 1024,
            overlap_height_ratio: float = 0.2, overlap_width_ratio: float = 0.2,
            merge_ios_threshold: float = 0.5, nms_ios_threshold: float = 0.7) -> List[Detection]:
    """
    Generates the detections for the slices and combines them before returning them.
    
    :param image: the BGR image
    :param predictor: the Detectron2 model
    :param score_threshold: the minimum score for predictions
    :type score_threshold: float
    :param slice_height: the height of the slices in pixels
    :type slice_height: int
    :param slice_width: the width of the slices in pixels
    :type slice_width: int
    :param overlap_height_ratio: the ratio for the height overlaps
    :type overlap_height_ratio: float
    :param overlap_width_ratio: the ratio for the width overlaps
    :type overlap_width_ratio: float
    :param merge_ios_threshold: the IoS threshold for merging
    :type merge_ios_threshold: float
    :param nms_ios_threshold: the IoS threshold for removing overlapping polygons
    :type nms_ios_threshold: float
    :return: the list of detections
    :rtype: list
    """
    raw = sliced_inference(
        image=image,
        predictor=predictor,
        score_threshold=score_threshold,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio
    )
    merged = merge_detections(raw, ios_threshold=merge_ios_threshold)
    no_overlaps = polygon_nms(merged, ios_threshold=nms_ios_threshold)
    return no_overlaps


def geometry_to_points(geom: BaseGeometry, simplify_tolerance: float = 1.0) -> List:
    """
    Converts each polygon into a list of [x,y] pairs.
    Only the largest polygon of multi-polygons is kept.

    :param geom: the polygon to convert
    :type geom: BaseGeometry
    :param simplify_tolerance: the tolerance parameter for polygon simplification (the higher, the fewer vertices)
    :type simplify_tolerance: float
    :return: the list of lists of x/y pairs
    :rtype: list
    """
    result = []

    if geom is None:
        return result

    if isinstance(geom, Polygon):
        geom = simplify(geom, simplify_tolerance)
        result.append([[float(x), float(y)] for x, y in geom.exterior.coords[:-1]])

    elif isinstance(geom, MultiPolygon):
        # find largest polygon
        largest_poly = None
        largest_area = 0
        for poly in geom.geoms:
            current_area = poly.area
            if poly.area > largest_area:
                largest_area = current_area
                largest_poly = poly
        # add largest polygon
        if largest_poly is not None:
            largest_poly = simplify(largest_poly, simplify_tolerance)
            result.append([[float(x), float(y)] for x, y in largest_poly.exterior.coords[:-1]])

    return result


def detection_to_opex(detection: Detection, class_mapping: Dict = None, simplify_tolerance: float = 1.0) -> List[ObjectPrediction]:
    """
    Converts a Detection into ObjectPrediction object(s).

    :param detection: the detection to convert
    :type detection: Detection
    :param class_mapping: the mapping of class index/label
    :type class_mapping: dict
    :param simplify_tolerance: the tolerance parameter for simplifying the polygons
    :type simplify_tolerance: float
    :return: the list of objects
    :rtype: list
    """
    result = []
    label = str(detection.category_id)
    if class_mapping is not None:
        label = class_mapping.get(detection.category_id, label)

    if detection.polygon is not None:
        polygons = geometry_to_points(detection.polygon, simplify_tolerance=simplify_tolerance)
        x0, y0, x1, y1 = detection.bbox
        bbox = BBox(left=int(x0), top=int(y0), right=int(x1), bottom=int(y1))
        for polygon in polygons:
            poly = OpexPolygon(points=polygon)
            obj = ObjectPrediction(score=float(detection.score), label=label, bbox=bbox, polygon=poly)
            result.append(obj)
    else:
        x0, y0, x1, y1 = detection.bbox
        px = [int(x0), int(x1), int(x1), int(x0)]
        py = [int(y0), int(y0), int(y1), int(y1)]
        bbox = BBox(left=int(x0), top=int(y0), right=int(x1), bottom=int(y1))
        points = []
        for x, y in zip(px, py):
            points.append((int(x), int(y)))
        poly = OpexPolygon(points=points)
        obj = ObjectPrediction(score=float(detection.score), label=label, bbox=bbox, polygon=poly)
        result.append(obj)

    return result


def detections_to_opex(detections: List[Detection], image_id: str, class_mapping: Dict = None, timestamp: str = None, simplify_tolerance: float = 1.0) -> ObjectPredictions:
    """
    Converts merged detections into a Fast-OPEX annotation.
    """
    objs = []
    if timestamp is None:
        timestamp = str(datetime.now())
    for detection in detections:
        objs.extend(detection_to_opex(detection, class_mapping=class_mapping, simplify_tolerance=simplify_tolerance))
    return ObjectPredictions(id=image_id, timestamp=timestamp, objects=objs)
