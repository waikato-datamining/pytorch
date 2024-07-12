from datetime import datetime

import torch
from opex import ObjectPredictions, ObjectPrediction, BBox, Polygon

from ultralytics import YOLOv10

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


def load_model(model_path):
    """
    Loads the model from disk.

    :param model_path: the model to load
    :type model_path: str
    :return: the model parameters
    :rtype: ModelParams
    """
    device = "cpu"
    model = YOLOv10(model_path).to(device)

    result = ModelParams()
    result.model = model
    result.device = device
    result.names = model.names

    return result


def predict_image_opex(model_params, id, img, confidence_threshold=0.25,
                       classes=None, augment=False, agnostic_nms=False):
    """
    Generates predictions for an image.
    
    :param model_params: the model to use for making predictions
    :param model_params: ModelParams
    :param id: the ID to use for the predictions
    :type id: str
    :param img: the PIL image to push through the model
    :param confidence_threshold: the confidence threshold (0-1)
    :type confidence_threshold: float
    :param classes: the classes to filter by (list of 0-based label indices)
    :type classes: list
    :param agnostic_nms: whether to use class-agnostic NMS
    :type agnostic_nms: bool
    :param augment: whether to use augmented inference
    :type augment: bool
    :return: the generated predictions
    :rtype: ObjectPredictions
    """

    # Inference
    with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
        preds = model_params.model.predict(source=img, augment=augment, agnostic_nms=agnostic_nms)

    width, height = img.size

    if classes is None:
        classes_set = set(model_params.names.values())
    else:
        classes_set = set(classes)

    # Process detections
    opex_preds = []
    for i, pred in enumerate(preds):
        for box in pred.boxes:
            conf = float(box.conf)
            if conf < confidence_threshold:
                continue
            cls = float(box.cls)
            label = model_params.names[cls]
            if label not in classes_set:
                continue
            xyxyn = box.xyxyn.numpy()
            bbox = BBox(left=int(xyxyn[0][0]*width), top=int(xyxyn[0][1]*height), right=int(xyxyn[0][2]*width), bottom=int(xyxyn[0][3]*height))
            p = [[bbox.left, bbox.top],
                 [bbox.right, bbox.top],
                 [bbox.right, bbox.bottom],
                 [bbox.left, bbox.bottom]]
            poly = Polygon(points=p)
            opex_pred = ObjectPrediction(score=conf, label=label, bbox=bbox, polygon=poly)
            opex_preds.append(opex_pred)

    result = ObjectPredictions(id=id, timestamp=str(datetime.now()), objects=opex_preds)
    return result
