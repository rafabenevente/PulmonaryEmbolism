from typing import List

import numpy as np
import os
import matplotlib.pyplot as plt
import pydicom
import cv2

def _transform_to_hu(slice, size):
    image = slice.pixel_array.astype(np.int16)

    if image.shape != size:
        image = cv2.resize(image, size)

    # Set air values to 0
    image[image <= - 1000] = 0

    # convert to HU
    intercept = slice.RescaleIntercept
    slope = slice.RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)

    return np.array(image, dtype=np.int16)


def _set_manual_window(hu_image, custom_center, custom_width):
    w_image = hu_image
    min_value = custom_center - (custom_width / 2)
    max_value = custom_center + (custom_width / 2)
    w_image[w_image < min_value] = min_value
    w_image[w_image > max_value] = max_value
    return w_image


def dicom_to_jpg(path: str,
                 size: List,
                 apply_custom_windowing: bool = True):
    dicom = pydicom.dcmread(path)
    image = _transform_to_hu(dicom, size)

    if apply_custom_windowing:
        channel_1 = _set_manual_window(hu_image=image.copy(), custom_center=-600, custom_width=1500)
        channel_2 = _set_manual_window(hu_image=image.copy(), custom_center=100, custom_width=700)
        channel_3 = _set_manual_window(hu_image=image.copy(), custom_center=40, custom_width=400)
        return np.dstack((channel_1, channel_2, channel_3))
    else:
        return image


if __name__ == "__main__":
    image_path = os.path.join(os.getcwd(), "data")
    scans = [f for f in os.listdir(image_path) if f != "train.csv"]
    slices = [dicom_to_jpg(os.path.join(image_path, scan)) for scan in scans]
    for slice in slices:
        ig, ax = plt.subplots(1, 3, figsize=(15, 8))
        ax[0].set_title("C:-600 W:1500")
        ax[0].imshow(slice[:, :, 0], cmap="gray")

        ax[1].set_title("C:100 W:700")
        ax[1].imshow(slice[:, :, 1], cmap="gray")

        ax[2].set_title("C:40 W:400")
        ax[2].imshow(slice[:, :, 2], cmap="gray")
        plt.show()
