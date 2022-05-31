import numpy as np
import torch
from datetime import datetime
from models.common import DetectMultiBackend, non_max_suppression
from utils.general import check_img_size
from utils.general import scale_coords
from utils.augmentations import letterbox
from opex import ObjectPredictions, ObjectPrediction, BBox, Polygon
from wai.annotations.roi import ROIObject
from utils.torch_utils import select_device


def load_model(model_path, data_path, image_size):
    """
    Loads the model from disk.

    :param model_path: the ONNX model to load
    :type model_path: str
    :param data_path: the YAML describing the dataset (for loading labels; https://github.com/ultralytics/yolov5/blob/master/data/coco128.yaml)
    :type data_path: str
    :param image_size: the maximum size for the image (width and height)
    :type image_size: int
    :return: the model instance
    """
    device = select_device("cpu")
    result = DetectMultiBackend(model_path, device=device, dnn=False, data=data_path)
    imgsz = [image_size, image_size]
    imgsz = check_img_size(imgsz, s=result.stride)  # check image size
    result.warmup(imgsz=(1, 3, *imgsz), half=False)  # warmup
    return result


def prepare_image(image, image_size, stride, device):
    """
    Prepares the image for inference.

    :param image: the image as loaded with OpenCV
    :param image_size: the maximum size (used for width and height)
    :type image_size: int
    :param stride: the stride to use
    :type stride: int
    :param device: the torch device to use
    :return: the prepared image
    """
    im = letterbox(image, image_size, stride=stride, auto=False)[0]

    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)
    im = torch.from_numpy(im).to(device)
    im = im.float()  # uint8 to fp32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    return im


def predict_image_opex(model, id, img_scaled, image_orig, confidence_threshold=0.25, iou_threshold=0.45, max_detection=1000):
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
    :param max_detection: the maximum number of detections to use
    :type max_detection: int
    :return: the generated predictions
    :rtype: ObjectPredictions
    """
    pred = model(img_scaled, augment=False, visualize=False)
    pred = non_max_suppression(pred, confidence_threshold, iou_threshold,
                               None, False, max_det=max_detection)
    opex_preds = []
    for i, det in enumerate(pred):
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img_scaled.shape[2:], det[:, :4], image_orig.shape).round()
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

    results = ObjectPredictions(id=id, timestamp=str(datetime.now()), objects=opex_preds)
    return results


def predict_image_rois(model, img_scaled, image_orig, confidence_threshold=0.25, iou_threshold=0.45, max_detection=1000):
    """
    Generates predictions for an image, generating rois.

    :param model: the model to use for making predictions
    :param img_scaled: the scaled image to push through the model
    :param image_orig: the original image (for scaling the predictions)
    :param confidence_threshold: the confidence threshold (0-1)
    :type confidence_threshold: float
    :param iou_threshold: the threshold for IoU (intersect over union)
    :type iou_threshold: float
    :param max_detection: the maximum number of detections to use
    :type max_detection: int
    :return: the generated predictions, list of ROIObject
    :rtype: list
    """
    pred = model(img_scaled, augment=False, visualize=False)
    pred = non_max_suppression(pred, confidence_threshold, iou_threshold,
                               None, False, max_det=max_detection)
    result = []
    image_width = image_orig.shape[1]
    image_height = image_orig.shape[0]
    for i, det in enumerate(pred):
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img_scaled.shape[2:], det[:, :4], image_orig.shape).round()
            for d in det:
                score = float(d[4])
                label = int(d[5])
                label_str = model.names[label]
                x0 = int(d[0])
                y0 = int(d[1])
                x1 = int(d[2])
                y1 = int(d[3])
                x0n = x0 / image_width
                y0n = y0 / image_height
                x1n = x1 / image_width
                y1n = y1 / image_height
                px = [x0, x1, x1, x0]
                py = [y0, y0, y1, y1]
                pxn = [x / image_width for x in px]
                pyn = [y / image_height for y in py]

                roiobj = ROIObject(x0, y0, x1, y1, x0n, y0n, x1n, y1n, label, label_str, score=score,
                                   poly_x=px, poly_y=py, poly_xn=pxn, poly_yn=pyn)
                result.append(roiobj)

    return result
