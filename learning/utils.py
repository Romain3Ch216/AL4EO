import torch
import numpy as np
from sklearn.decomposition import PCA
import torch.utils.data as data
import spectral
import itertools
import numpy as np
import torch
from scipy.special import softmax, expit
import dataclasses
import math
import warnings
from collections import defaultdict
from collections.abc import Mapping, Sequence
from functools import singledispatch

# From https://github.com/pytorch/ignite with slight changes

def to_prob(probabilities: np.ndarray):
    """
    If the probabilities array is not a distrubution will softmax it.

    Args:
        probabilities (array): [batch_size, num_classes, ...]

    Returns:
        Same as probabilities.
    """
    not_bounded = np.min(probabilities) < 0 or np.max(probabilities) > 1.0
    multiclass = probabilities.shape[1] > 1
    sum_to_one = np.allclose(probabilities.sum(1), 1)
    if not_bounded or (multiclass and not sum_to_one):
        if multiclass:
            probabilities = softmax(probabilities, 1)
        else:
            probabilities = expit(probabilities)
    return probabilities


def stack_in_memory(data, iterations):
    """
    Stack `data` `iterations` times on the batch axis.
    Args:
        data (Tensor): Data to stack
        iterations (int): Number of time to stack.

    Raises:
        RuntimeError when CUDA is out of memory.

    Returns:
        Tensor with shape [batch_size * iterations, ...]
    """
    input_shape = data.size()
    batch_size = input_shape[0]
    try:
        data = torch.stack([data] * iterations)
    except RuntimeError as e:
        raise RuntimeError(
            '''CUDA ran out of memory while BaaL tried to replicate data. See the exception above.
        Use `replicate_in_memory=False` in order to reduce the memory requirements.
        Note that there will be some speed trade-offs''') from e
    data = data.view(batch_size * iterations, *input_shape[1:])
    return data

@singledispatch
def to_cuda(data):
    """
    Move an object to CUDA.

    This function works recursively on lists and dicts, moving the values
    inside to cuda.

    Args:
        data (list, tuple, dict, torch.Tensor, torch.nn.Module):
            The data you'd like to move to the GPU. If there's a pytorch tensor or
            model in data (e.g. in a list or as values in a dictionary) this
            function will move them all to CUDA and return something that matches
            the input in structure.

    Returns:
        list, tuple, dict, torch.Tensor, torch.nn.Module:
            Data of the same type / structure as the input.
    """
    # the base case: if this is not a type we recognise, return it
    return data


@to_cuda.register(torch.Tensor)
@to_cuda.register(torch.nn.Module)
def _(data):
    return data.cuda()


@to_cuda.register
def _(data: Mapping):
    # use the type of the object to create a new one:
    return type(data)([(key, to_cuda(val)) for key, val in data.items()])


@to_cuda.register
def _(data: Sequence):
    """
        Move an object to CUDA.

        This function works recursively on lists and dicts, moving the values
        inside to cuda.

        Args:
            data (Sequence):
                The data you'd like to move to the GPU. If there's a pytorch tensor or
                model in data (e.g. in a list or as values in a dictionary) this
                function will move them all to CUDA and return something that matches
                the input in structure.

        Returns:
            Sequence, with the elements converted to cuda if possible
    """
    # use the type of this object to instantiate a new one:
    if hasattr(data, "_fields"):  # in case it's a named tuple
        return type(data)(*(to_cuda(item) for item in data))
    elif isinstance(data, str):
        # Special case
        return data
    else:
        return type(data)(to_cuda(item) for item in data)
from collections.abc import Sequence


def map_on_tensor(fn, val):
    """Map a function on a Tensor or a list of Tensors"""
    if isinstance(val, Sequence):
        return [fn(v) for v in val]
    elif isinstance(val, dict):
        return {k: fn(v) for k, v in val.items()}
    return fn(val)

# From https://github.com/nshaud/DeepHyperX

def hyper2rgb(img, bands):
    """Convert hyperspectral cube to a rgb image.
    Args:
        img: HS npy cube
        bands: tuple of rgb bands
    Returns:
        npy rgb array
    """
    rgb = spectral.get_rgb(img, bands)
    rgb /= np.max(rgb)
    rgb = np.asarray(255 * rgb, dtype='uint8')
    return rgb

def window(image, coordinates, window_size=(100, 100)):
    """Sliding window generator over an input image.
    Args:
        image: 2D+ image to slide the window on, e.g. RGB or hyperspectral
        coordinates: list of tuples coordinates of centered pixel
    Outputs:
        dict of img patches centered at pixel
    """
    # slide a window across the image

    patches = {}
    patches_coord = {'x': {}, 'y': {}}
    for patch_id, coordinate in enumerate(coordinates):
        x, y = coordinate
        x1, y1 = max(0, x - window_size[0] // 2), max(0, y - window_size[1] // 2)
        x2, y2 = min(x1 + window_size[0], image.shape[0]), min(y1 + window_size[1], image.shape[1])

        patches[patch_id] = image[x1:x2, y1:y2]

        if x1 == 0:
            patches_coord['x'][patch_id] = x
        else:
            patches_coord['x'][patch_id] = window_size[0] // 2

        if y1 == 0:
            patches_coord['y'][patch_id] = y
        else:
            patches_coord['y'][patch_id] = window_size[0] // 2

    return patches, patches_coord


def build_dataset(mat, gt, ignored_labels=None):
    """Create a list of training samples based on an image and a mask.

    Args:
        mat: 3D hyperspectral matrix to extract the spectrums from
        gt: 2D ground truth
        ignored_labels (optional): list of classes to ignore, e.g. 0 to remove
        unlabeled pixels
        return_indices (optional): bool set to True to return the indices of
        the chosen samples

    """
    samples = []
    labels = []
    # Check that image and ground truth have the same 2D dimensions
    assert mat.shape[:2] == gt.shape[:2]

    for label in np.unique(gt):
        if label in ignored_labels:
            continue
        else:
            indices = np.nonzero(gt == label)
            samples += list(mat[indices])
            labels += len(indices[0]) * [label]
    return np.asarray(samples), np.asarray(labels)


def sliding_window(image, step=10, window_size=(20, 20), with_data=True):
    """Sliding window generator over an input image.
    Args:
        image: 2D+ image to slide the window on, e.g. RGB or hyperspectral
        step: int stride of the sliding window
        window_size: int tuple, width and height of the window
        with_data (optional): bool set to True to return both the data and the
        corner indices
    Yields:
        ([data], x, y, w, h) where x and y are the top-left corner of the
        window, (w,h) the window size
    """
    # slide a window across the image
    w, h = window_size
    W, H = image.shape[:2]
    offset_w = (W - w) % step
    offset_h = (H - h) % step
    for x in range(0, W - w + offset_w, step):
        if x + w > W:
            x = W - w
        for y in range(0, H - h + offset_h, step):
            if y + h > H:
                y = H - h
            if with_data:
                yield image[x:x + w, y:y + h], x, y, w, h
            else:
                yield x, y, w, h

def count_sliding_window(top, step=10, window_size=(20, 20)):
    """ Count the number of windows in an image.
    Args:
        image: 2D+ image to slide the window on, e.g. RGB or hyperspectral, ...
        step: int stride of the sliding window
        window_size: int tuple, width and height of the window
    Returns:
        int number of windows
    """
    sw = sliding_window(top, step, window_size, with_data=False)
    return sum(1 for _ in sw)


def grouper(n, iterable):
    """ Browse an iterable by grouping n elements by n elements.
    Args:
        n: int, size of the groups
        iterable: the iterable to Browse
    Yields:
        chunk of n elements from the iterable
    """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk
