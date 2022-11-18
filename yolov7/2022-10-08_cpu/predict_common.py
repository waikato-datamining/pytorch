import numpy as np
import torch
from datetime import datetime
from utils.general import check_img_size
from opex import ObjectPredictions, ObjectPrediction, BBox, Polygon
from wai.annotations.roi import ROIObject
from utils.torch_utils import TracedModel
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords


class ModelParams:
    """
    Container for the model and additional parameters.
    """
    def __init__(self):
        self.model = None
        self.device = None
        self.stride = None
        self.imgsz = None
        self.half = None
        self.names = None


def load_model(model_path, image_size, no_trace):
    """
    Loads the model from disk.

    :param model_path: the ONNX model to load
    :type model_path: str
    :param image_size: the maximum size for the image (width and height)
    :type image_size: int
    :return: the model instance
    :param no_trace: don't trace model
    :type no_trace: bool
    :return: the model parameters
    :rtype: ModelParams
    """
    device = torch.device("cpu")
    half = device.type != 'cpu'  # half precision only supported on CUDA
    model = attempt_load(model_path, map_location=device)
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(image_size, s=stride)  # check image size
    if not no_trace:
        model = TracedModel(model, device, imgsz)
    if half:
        model.half()  # to FP16
    names = model.module.names if hasattr(model, 'module') else model.names
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    result = ModelParams()
    result.model = model
    result.device = device
    result.stride = stride
    result.half = half
    result.imgsz = imgsz
    result.names = names

    return result


def warmup_model_if_necessary(model, device, img, old_img_b, old_img_w, old_img_h, augment=False):
    """
    Warms up the model if shapes a different (only on CUDA).

    :param model: the model to warm up
    :param device: the torch device to use
    :param old_img_b: the old image #channels
    :type old_img_b: int
    :param old_img_w: the old image width
    :type old_img_w: int
    :param old_img_h: the old image height
    :type old_img_h: int
    :param augment: whether to use augmented inference
    :type augment: bool
    :return: tuple of (old_img_b, old_img_w, old_img_h)
    :rtype: tuple
    """
    if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
        old_img_b = img.shape[0]
        old_img_h = img.shape[2]
        old_img_w = img.shape[3]
        for i in range(3):
            model(img, augment=augment)[0]
    return old_img_b, old_img_w, old_img_h


def prepare_image(image, image_size, stride, device, half):
    """
    Prepares the image for inference.

    :param image: the image as loaded with OpenCV
    :param image_size: the maximum size (used for width and height)
    :type image_size: int
    :param stride: the stride to use
    :type stride: int
    :param device: the torch device to use
    :param half: whether to use half precision
    :type half: bool
    :return: the prepared image as tensor
    """
    # Padded resize
    img = letterbox(image, image_size, stride=stride)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    # to tensor
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    return img


def predict_image_opex(model, id, img_scaled, image_orig,
                       confidence_threshold=0.25, iou_threshold=0.45,
                       classes=None, augment=False, agnostic_nms=False):
    """
    Generates predictions for an image.
    
    :param model: the model to use for making predictions
    :param id: the ID to use for the predictions
    :type id: str
    :param img_scaled: the scaled image to push through the model
    :param image_orig: the orignal image (for scaling the predictions)
    :param confidence_threshold: the confidence threshold (0-1)
    :type confidence_threshold: float
    :param iou_threshold: the threshold for IoU (intersect over union)
    :type iou_threshold: float
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
        pred = model(img_scaled, augment=augment)[0]

    # Apply NMS
    pred = non_max_suppression(pred, confidence_threshold, iou_threshold,
                               classes=classes, agnostic=agnostic_nms)

    # Process detections
    opex_preds = []
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img_scaled.shape[2:], det[:, :4], image_orig.shape).round()
            # generate output
            for d in det:
                score = float(d[4])
                label = model.names[int(d[5])]
                bbox = BBox(left=int(d[0]), top=int(d[1]), right=int(d[2]), bottom=int(d[3]))
                p = [(bbox.left, bbox.top),
                     (bbox.right, bbox.top),
                     (bbox.right, bbox.bottom),
                     (bbox.left, bbox.bottom)]
                poly = Polygon(points=p)
                opex_pred = ObjectPrediction(score=score, label=label, bbox=bbox, polygon=poly)
                opex_preds.append(opex_pred)

    result = ObjectPredictions(id=id, timestamp=str(datetime.now()), objects=opex_preds)
    return result


def predict_image_rois(model, img_scaled, image_orig, old_img_w, old_img_h,
                       confidence_threshold=0.25, iou_threshold=0.45,
                       classes=None, augment=False, agnostic_nms=False):
    """
    Generates predictions for an image, generating rois.

    :param model: the model to use for making predictions
    :param img_scaled: the scaled image to push through the model
    :param image_orig: the original image (for scaling the predictions)
    :param old_img_w: the old image width
    :type old_img_w: int
    :param old_img_h: the old image height
    :type old_img_h: int
    :param confidence_threshold: the confidence threshold (0-1)
    :type confidence_threshold: float
    :param iou_threshold: the threshold for IoU (intersect over union)
    :type iou_threshold: float
    :param classes: the classes to filter by (list of 0-based label indices)
    :type classes: list
    :param agnostic_nms: whether to use class-agnostic NMS
    :type agnostic_nms: bool
    :param augment: whether to use augmented inference
    :type augment: bool
    :return: the generated predictions, list of ROIObject
    :rtype: list
    """
    result = []

    # Inference
    with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
        pred = model(img_scaled, augment=augment)[0]

    # Apply NMS
    pred = non_max_suppression(pred, confidence_threshold, iou_threshold,
                               classes=classes, agnostic=agnostic_nms)

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img_scaled.shape[2:], det[:, :4], image_orig.shape).round()
            # generate output
            for d in det:
                score = float(d[4])
                label = int(d[5])
                label_str = model.names[label]
                x0 = int(d[0])
                y0 = int(d[1])
                x1 = int(d[2])
                y1 = int(d[3])
                x0n = x0 / old_img_w
                y0n = y0 / old_img_h
                x1n = x1 / old_img_w
                y1n = y1 / old_img_h
                px = [x0, x1, x1, x0]
                py = [y0, y0, y1, y1]
                pxn = [x / old_img_w for x in px]
                pyn = [y / old_img_h for y in py]

                roiobj = ROIObject(x0, y0, x1, y1, x0n, y0n, x1n, y1n, label, label_str, score=score,
                                   poly_x=px, poly_y=py, poly_xn=pxn, poly_yn=pyn)
                result.append(roiobj)

    return result