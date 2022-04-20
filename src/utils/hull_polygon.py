import numpy as np
import pandas as pd
import cv2

from shapely.geometry import Polygon


def get_indices_pandas(data, background_id=0):
    """
    Extracts for each mask id its positions within the array.
    Args:
        data: a np. array with masks, where all pixels belonging to the
            same masks have the same integer value
        background_id: integer value of the background

    Returns: data frame: indices are the mask id , values the positions of the mask pixels

    """
    if data.size < 1e9:  # aim for speed at cost of high memory consumption
        masked_data = data != background_id
        flat_data = data[masked_data]  # d: data , mask attribute
        dummy_index = np.where(masked_data.ravel())[0]
        df = pd.DataFrame.from_dict({"mask_id": flat_data, "flat_index": dummy_index})
        df = df.groupby("mask_id").apply(
            lambda x: np.unravel_index(x.flat_index, data.shape)
        )
    else:  # aim for lower memory consumption at cost of speed
        flat_data = data[(data != background_id)]  # d: data , mask attribute
        dummy_index = np.where((data != background_id).ravel())[0]
        mask_indices = np.unique(flat_data)
        df = {"mask_id": [], "index": []}
        data_shape = data.shape
        for mask_id in mask_indices:
            df["index"].append(
                np.unravel_index(dummy_index[flat_data == mask_id], data_shape)
            )
            df["mask_id"].append(mask_id)
        df = pd.DataFrame.from_dict(df)
        df = df.set_index("mask_id")
        df = df["index"].apply(lambda x: x)  # convert to same format as for other case
    return df


def cv2_countour(mask_idx):
    mask_idx = np.array(mask_idx)
    mask_img = np.zeros(
        tuple(np.max(mask_idx, axis=1) - np.min(mask_idx, axis=1) + 2 + 1)
    )
    mask_img[tuple(mask_idx - np.min(mask_idx, axis=1).reshape(-1, 1) + 1)] = 255
    mask_img = mask_img.astype(np.uint8)
    contour_points, hierachy = cv2.findContours(
        mask_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )
    if hierachy.squeeze().ndim > 1:

        # Check if hole is in mask
        outer_shape = None
        contour_points_list = []
        for idx in range(len(contour_points)-1):
            if len(contour_points[idx]) <= 2:
                continue
            if len(contour_points[idx+1]) <= 2:
                continue
            polygon_0 = Polygon(contour_points[idx].squeeze())
            polygon_1 = Polygon(contour_points[idx+1].squeeze())
            if polygon_0.covers(polygon_1):
                outer_shape = contour_points[idx]
            elif polygon_1.covers(polygon_0):
                outer_shape = contour_points[idx+1]
            if outer_shape is not None:
                contour_points = np.array(contour_points[idx]).squeeze().reshape(-1, 2)[:, ::-1]
                contour_points = contour_points + np.min(mask_idx, axis=1).reshape(1, -1) - 1
                contour_points_list.append(contour_points.T)
        if outer_shape is not None:
            contour_points = np.array(outer_shape).squeeze().reshape(-1, 2)[:, ::-1]
            contour_points = contour_points + np.min(mask_idx, axis=1).reshape(1, -1) - 1
            return [contour_points.T]
        else:
            contour_points = np.array(contour_points[-1]).squeeze().reshape(-1, 2)[:, ::-1]
            contour_points = contour_points + np.min(mask_idx, axis=1).reshape(1, -1) - 1
            contour_points_list.append(contour_points.T)
            return contour_points_list
    else:
        # N points, 2
        contour_points = np.array(contour_points).squeeze().reshape(-1, 2)[:, ::-1]

        contour_points = contour_points + np.min(mask_idx, axis=1).reshape(1, -1) - 1

        return [contour_points.T]
