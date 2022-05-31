import albumentations as albu


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callable): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


def get_augmentation(width, height):
    """
    Add paddings to make image shape divisible by 32.

    :param width: the width to use (divisible by 32)
    :type width: int
    :param height: the height to use (divisible by 32)
    :type height: int
    """
    test_transform = [
        albu.PadIfNeeded(height, width)
    ]
    return albu.Compose(test_transform)
