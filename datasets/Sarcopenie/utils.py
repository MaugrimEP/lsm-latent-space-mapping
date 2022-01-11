from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pydicom

# https://towardsdatascience.com/medical-image-pre-processing-with-python-d07694852606


def transform_to_hu(values: np.ndarray, rescale_intercept: float, rescale_slope: float) -> np.ndarray:
    hu_image = values * rescale_slope + rescale_intercept
    return hu_image


def normalize_pixel(hu_image: np.ndarray, window_level: int, window_width: int) -> np.ndarray:
    """
    Normalize the pixel according to the dcm format:
    :param hu_image:
    :param window_level:
    :param window_width:
    :return:
    """
    hu_image = hu_image.astype(float)
    lowest_visible_value = window_level - window_width / 2
    highest_visible_value = window_level + window_width / 2

    inf_mask = (hu_image > lowest_visible_value).astype(float)
    sup_mask = (hu_image < highest_visible_value).astype(float)

    interpolated = (hu_image - lowest_visible_value) / (highest_visible_value - lowest_visible_value)

    final_pixels = interpolated * inf_mask * sup_mask

    return final_pixels


def crop_image(image: np.ndarray, return_coord: bool = False):
    # Create a mask with the background pixels
    mask = image == 0

    # Find the not background area
    coords = np.array(np.nonzero(~mask))
    top_left = np.min(coords, axis=-1)
    bottom_right = np.max(coords, axis=1)

    # Remove the background
    cropped_image = image[top_left[0]:bottom_right[0],
                          top_left[1]:bottom_right[1]]

    if return_coord:
        return cropped_image, top_left, bottom_right
    return cropped_image


def crop_image_with_coord(image: np.ndarray, top_left, bottom_right) -> np.ndarray:
    # Remove the background
    cropped_image = image[top_left[0]:bottom_right[0],
                          top_left[1]:bottom_right[1]]
    return cropped_image


def add_pad(image: np.ndarray, new_height: int, new_width: int, padding_value=0.,
            vertical_padding: str = 'even', horizontal_padding: str = 'even') -> np.ndarray:
    """
    'even', 'lower', 'higher'
    even: the image will be centered on the axis
    lower: the image will be at the lower coordinate on the axis
    higher: the image will be at the higher coordinate ont the axis
    :return:
    """
    height, width = image.shape

    if new_height < height or new_width < width:
        raise Exception(f"new_height < height or new_width < width: {new_height=}, {height=}, {new_width=}, {width=}")

    final_image = np.full(shape=[new_height, new_width], fill_value=padding_value)

    inf_padding_functions = {
        'even': lambda new, old: int((new-old)//2),
        'lower': lambda new, old: new-old,
        'right': lambda new, old: 0,
    }

    padding_top = inf_padding_functions[vertical_padding](new_height, height)
    padding_left = inf_padding_functions[horizontal_padding](new_width, width)

    final_image[padding_top:padding_top+height, padding_left:padding_left+width] = image

    return final_image
