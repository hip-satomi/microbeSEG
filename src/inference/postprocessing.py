import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.segmentation import watershed
from skimage import measure


def distance_postprocessing(border_prediction, cell_prediction, th_seed, th_cell):
    """ Post-processing for distance label (cell + neighbor) prediction.

    :param border_prediction: Neighbor distance prediction.
    :type border_prediction:
    :param cell_prediction: Cell distance prediction.
    :type cell_prediction:
    :param th_seed: Threshold for seed/marker extraction
    :type th_seed: float
    :param th_cell: Threshold for cell size
    :type th_cell: float
    :return: Instance segmentation mask.
    """

    # Smooth predictions slightly + clip border prediction (to avoid negative values being positive after squaring)
    sigma_cell = 0.5
    # sigma_border = 0.5

    cell_prediction = gaussian_filter(cell_prediction, sigma=sigma_cell)
    # border_prediction = gaussian_filter(border_prediction, sigma=sigma_border)
    border_prediction = np.clip(border_prediction, 0, 1)

    # Get mask for watershed
    mask = cell_prediction > th_cell

    # Get seeds for marker-based watershed
    borders = np.tan(border_prediction ** 2)
    borders[borders < 0.05] = 0
    borders = np.clip(borders, 0, 1)
    cell_prediction_cleaned = (cell_prediction - borders)
    seeds = cell_prediction_cleaned > th_seed
    seeds = measure.label(seeds, background=0)

    # Remove very small seeds
    props = measure.regionprops(seeds)
    areas = []
    for i in range(len(props)):
        areas.append(props[i].area)
    if len(areas) > 0:
        min_area = 0.10 * np.mean(np.array(areas))
    else:
        min_area = 0
    min_area = np.maximum(min_area, 4)

    for i in range(len(props)):
        if props[i].area <= min_area:
            seeds[seeds == props[i].label] = 0
    seeds = measure.label(seeds, background=0)

    # Marker-based watershed
    prediction_instance = watershed(image=-cell_prediction, markers=seeds, mask=mask, watershed_line=False)

    return np.squeeze(prediction_instance.astype(np.uint16))


def boundary_postprocessing(prediction):
    """ Post-processing for boundary label prediction.

    :param prediction: Boundary label prediction.
    :type prediction:
    :return: Instance segmentation mask, binary raw prediction (0: background, 1: cell, 2: boundary).
    """

    # Binarize the channels
    prediction_bin = np.argmax(prediction, axis=-1).astype(np.uint16)

    # Get mask to flood with watershed
    mask = (prediction_bin == 1)  # only interior cell class belongs to cells

    # Get seeds for marker-based watershed
    seeds = (prediction[:, :, 1] * (1 - prediction[:, :, 2])) > 0.5
    seeds = measure.label(seeds, background=0)

    # Remove very small seeds
    props = measure.regionprops(seeds)
    for i in range(len(props)):
        if props[i].area <= 4:
            seeds[seeds == props[i].label] = 0
    seeds = measure.label(seeds, background=0)

    # Marker-based watershed
    prediction_instance = watershed(image=mask, markers=seeds, mask=mask, watershed_line=False)

    return np.squeeze(prediction_instance.astype(np.uint16))
