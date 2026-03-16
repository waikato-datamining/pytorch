# based on:
# https://github.com/YuanGongND/ast/blob/31088be8a3f6ef96416145c4b8d43c81f99eba7a/egs/audioset/inference.py

import csv
import json

import numpy as np
import torch
import torchaudio
from src.models import ASTModel


PREDICTION_FORMAT_JSON = "json"
PREDICTION_FORMAT_TEXT = "text"   # just the class with the highest score
PREDICTION_FORMATS = [
    PREDICTION_FORMAT_JSON,
    PREDICTION_FORMAT_TEXT,
]


def load_model(checkpoint_path, device, input_tdim):
    """
    Loads the model and returns it.

    :param checkpoint_path: the path of the pretrained model
    :type checkpoint_path: str
    :param device: the device to run the model on, eg cuda:0
    :type device: str
    :param input_tdim: the input dimensions
    :return: the model
    """
    ast_mdl = ASTModel(label_dim=527, input_tdim=input_tdim, imagenet_pretrain=False, audioset_pretrain=False)
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    audio_model = torch.nn.DataParallel(ast_mdl, device_ids=[0])
    audio_model.load_state_dict(checkpoint)
    audio_model = audio_model.to(torch.device(device))
    return audio_model


def make_features(wav, mel_bins, target_length=1024):
    """
    Generates the features from WAV data.

    :param wav: the wav file or byte like data
    :param mel_bins: the number of MEL spectrogram bins
    :type mel_bins: int
    :param target_length: the number to generate
    :return: the feature bank
    """
    waveform, sr = torchaudio.load(wav)

    fbank = torchaudio.compliance.kaldi.fbank(
        waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
        window_type='hanning', num_mel_bins=mel_bins, dither=0.0,
        frame_shift=10)

    n_frames = fbank.shape[0]

    p = target_length - n_frames
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]

    fbank = (fbank - (-4.2677393)) / (4.5689974 * 2)
    return fbank


def load_label(label_csv):
    """
    Loads the labels from the CSV file and returns them.

    :param label_csv: the CSV file with the label indices
    :type label_csv: str
    :return: the generated list
    :rtype: list
    """
    with open(label_csv, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        lines = list(reader)
    labels = []
    ids = []  # Each label has a unique id such as "/m/068hy"
    for i1 in range(1, len(lines)):
        id_ = lines[i1][1]
        label = lines[i1][2]
        ids.append(id_)
        labels.append(label)
    return labels


def predict(model, wav, class_names, top_k=10, prediction_format=PREDICTION_FORMAT_TEXT):
    """
    Makes a prediction on the audio.

    :param model: the model to use
    :param wav: the audio data to classify
    :param class_names: the class name lookup
    :param top_k: the top K labels to return
    :param prediction_format: the output format to generate
    :return: the predictions
    """
    feats = make_features(wav, mel_bins=128)           # shape(1024, 128)

    # assume each input spectrogram has 100 time frames
    input_tdim = feats.shape[0]

    feats_data = feats.expand(1, input_tdim, 128)     # reshape the feature

    model.eval()                                      # set the eval model
    with torch.no_grad():
        output = model.forward(feats_data)
        output = torch.sigmoid(output)
    result_output = output.data.cpu().numpy()[0]
    sorted_indexes = np.argsort(result_output)[::-1]

    result = dict()
    result["class"] = str(np.array(class_names)[sorted_indexes[0]])
    result["scores"] = dict()
    for k in range(top_k):
        result["scores"][str(np.array(class_names)[sorted_indexes[k]])] = float(result_output[sorted_indexes[k]])

    if prediction_format == PREDICTION_FORMAT_JSON:
        return json.dumps(result, indent=2)
    elif prediction_format == PREDICTION_FORMAT_TEXT:
        return result["class"]
    else:
        raise Exception("Unhandled prediction format: %s" % prediction_format)
