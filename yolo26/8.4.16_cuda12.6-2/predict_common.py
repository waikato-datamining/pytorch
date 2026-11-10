import traceback
from datetime import datetime
from typing import List

import torch
from opex import ObjectPredictions, ObjectPrediction, BBox, Polygon
from sahi import AutoDetectionModel
from sahi.models.base import DetectionModel
from sahi.predict import get_sliced_prediction, get_prediction

OUTPUT_OPEX = "opex"
OUTPUT_FORMATS = [OUTPUT_OPEX]


class ModelParams:
    """
    Container for the model and additional parameters.
    """
    def __init__(self):
        self.model = None
        self.device = None
        self.names = None
        self.confidence_threshold = None


def load_model(model_path, device: str = "cuda", confidence_threshold: float = 0.0, text_prompt: List[str] = None):
    """
    Loads the model from disk.

    :param model_path: the model to load
    :type model_path: str
    :param device: the device to use, eg cuda or cuda:0
    :type device: str
    :param confidence_threshold: the threshold to use for predictions
    :type confidence_threshold: float
    :param text_prompt: the classes to use as text prompt for yoloe models
    :type text_prompt: list
    :return: the model
    :rtype: ModelParams
    """
    if (text_prompt is not None) and (len(text_prompt) > 0):
        model = AutoDetectionModel.from_pretrained(
            model_type="yoloe",
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            device=device,
        )
    else:
        model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            device=device,
        )

    result = ModelParams()
    result.model = model
    result.device = device
    result.names = model.model.names
    result.confidence_threshold = confidence_threshold

    return result


def predict_image_opex(model_params, id, img,
                       classes: List[str] = None,
                       slice_height: int = None, slice_width: int = None,
                       overlap_height_ratio: float = 0.2, overlap_width_ratio: float = 0.2):
    """
    Generates predictions for an image.
    
    :param model_params: the model to use for making predictions
    :param model_params: ModelParams
    :param id: the ID to use for the predictions
    :type id: str
    :param img: the PIL image to push through the model
    :param classes: the classes to filter by (list of labels)
    :type classes: list
    :param slice_height: the height of the slices in pixels
    :type slice_height: int
    :param slice_width: the width of the slices in pixels
    :type slice_width: int
    :param overlap_height_ratio: the ratio for the height overlaps
    :type overlap_height_ratio: float
    :param overlap_width_ratio: the ratio for the width overlaps
    :type overlap_width_ratio: float
    :return: the generated predictions
    :rtype: ObjectPredictions
    """
    if (slice_height is not None) and (slice_width is not None):
        pred_result = get_sliced_prediction(
            img,
            model_params.model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
            verbose=0
        )
    else:
        pred_result = get_prediction(img, model_params.model)

    if classes is None:
        classes_set = set(model_params.names.values())
    else:
        classes_set = set(classes)

    # Process detections
    opex_preds = []
    for i, pred in enumerate(pred_result.object_prediction_list):
        minx, miny, maxx, maxy = pred.bbox.box
        if pred.score.value < model_params.confidence_threshold:
            continue
        if pred.category.name not in classes_set:
            continue
        bbox = BBox(left=int(minx), top=int(miny), right=int(maxx), bottom=int(maxy))
        p = [[bbox.left, bbox.top],
             [bbox.right, bbox.top],
             [bbox.right, bbox.bottom],
             [bbox.left, bbox.bottom]]
        poly = Polygon(points=p)
        opex_pred = ObjectPrediction(score=float(pred.score.value), label=pred.category.name, bbox=bbox, polygon=poly)
        opex_preds.append(opex_pred)

    result = ObjectPredictions(id=id, timestamp=str(datetime.now()), objects=opex_preds)
    return result
