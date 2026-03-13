import csv
import json
import scipy
import tensorflow as tf
import tensorflow_hub as hub

from scipy.io import wavfile


PREDICTION_FORMAT_JSON = "json"
PREDICTION_FORMAT_TEXT = "text"   # just the class with the highest score
PREDICTION_FORMATS = [
    PREDICTION_FORMAT_JSON,
    PREDICTION_FORMAT_TEXT,
]


def load_model(url="https://www.kaggle.com/models/google/yamnet/TensorFlow2/yamnet/1"):
    """
    Loads the model and returns it.

    :param url: the tensorflow hub url
    :type url: str
    :return: the model
    """
    return hub.load(url)


def class_names_from_csv(class_map_csv_text):
    """
    Returns list of class names corresponding to score vector.

    :param class_map_csv_text: the CSV to read
    :return: the class names
    """
    class_names = []
    with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            class_names.append(row['display_name'])

    return class_names


def class_names_from_model(model):
    """
    Extracts the class names from the model.

    :param model: the model to get the class names from
    :return: the class names
    """
    class_map_path = model.class_map_path().numpy()
    return class_names_from_csv(class_map_path)


def ensure_sample_rate(original_sample_rate, waveform, desired_sample_rate=16000):
    """
    Resample waveform if required.

    :param original_sample_rate: the original sample rate of the audio data
    :param waveform: the audio data to process
    :param desired_sample_rate: the target sample rate
    :return: the tuple of desired reate and processed audio
    :rtype: tuple
    """
    if original_sample_rate != desired_sample_rate:
        desired_length = int(round(float(len(waveform)) /
                                   original_sample_rate * desired_sample_rate))
        waveform = scipy.signal.resample(waveform, desired_length)
    return desired_sample_rate, waveform


def load_audio(wav, desired_sample_rate=16000):
    """
    Loads the audio file and ensures the correct sample rate.

    :param wav: the audio file or file handle to load from
    :param desired_sample_rate: the sample rate to convert to
    :type desired_sample_rate: int
    :return: the tuple of sample rate and audio data
    :rtype: tuple
    """
    sample_rate, wav_data = wavfile.read(wav, 'rb')
    sample_rate, wav_data = ensure_sample_rate(sample_rate, wav_data, desired_sample_rate=desired_sample_rate)
    return sample_rate, wav_data


def predict(model, wav, class_names, prediction_format=PREDICTION_FORMAT_TEXT):
    """
    Makes a prediction on the audio.

    :param model: the model to use
    :param wav: the audio data to classify
    :param class_names: the class name lookup
    :param prediction_format: the output format to generate
    :return: the predictions
    """
    # normalize audio
    waveform = wav / tf.int16.max

    # Run the model, check the output.
    scores, embeddings, spectrogram = model(waveform)
    scores_np = scores.numpy()

    # generate output
    scores_mean = scores_np.mean(axis=0)
    result = dict()
    result["class"] = class_names[scores_mean.argmax()]
    result["scores"] = dict()
    for i, score in enumerate(scores_mean):
        result["scores"][class_names[i]] = float(score)

    if prediction_format == PREDICTION_FORMAT_JSON:
        return json.dumps(result, indent=2)
    elif prediction_format == PREDICTION_FORMAT_TEXT:
        return result["class"]
    else:
        raise Exception("Unhandled prediction format: %s" % prediction_format)
